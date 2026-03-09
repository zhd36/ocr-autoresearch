"""
OCR autoresearch training script. Single-GPU, single-file.

The benchmark, codec, dataloading, and evaluation live in prepare.py.
This file is the only one meant to be edited during autonomous research.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import math
import os
import random
import time
from dataclasses import asdict, dataclass

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import CODEC_PATH, TIME_BUDGET, OCRCodec, evaluate_cer, make_dataloader, prepare_cache


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def norm2d(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep_prob)
        mask = mask.div_(keep_prob)
        return x * mask


class ResidualTemporalRefine(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Temporal refine kernel size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.depthwise.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.pointwise.weight)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.depthwise(y)
        y = self.activation(y)
        y = self.pointwise(y)
        return x + y.transpose(1, 2)


class AsterBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes)
        squeeze_channels = max(planes // 8, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, squeeze_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, planes, kernel_size=1),
            nn.Sigmoid(),
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNetAsterEncoder(nn.Module):
    """
    Adapted from OpenOCR's ResNet_ASTER encoder.
    """

    def __init__(
        self,
        in_channels=3,
        lstm_hidden=256,
        lstm_layers=2,
        layer1_width_stride=2,
        layer2_width_stride=2,
        rnn_input_dim=512,
        lstm_proj_size=0,
        layer5_blocks=3,
    ):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
        )

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, layer1_width_stride])
        self.layer2 = self._make_layer(64, 4, [2, layer2_width_stride])
        self.layer3 = self._make_layer(128, 6, [2, 1])
        self.layer4 = self._make_layer(256, 6, [2, 1])
        self.layer5 = self._make_layer(512, layer5_blocks, [2, 1])
        self.pre_rnn_norm = nn.LayerNorm(512)
        self.rnn_input_dim = rnn_input_dim
        self.input_proj = None
        if rnn_input_dim != 512:
            self.input_proj = nn.Linear(512, rnn_input_dim)
        self.rnn_output_dim = lstm_proj_size if lstm_proj_size > 0 else lstm_hidden

        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=lstm_hidden,
            proj_size=lstm_proj_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else 0.1,
        )
        self.out_channels = 2 * self.rnn_output_dim

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm2d(planes),
            )
        layers = [AsterBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def encode_sequence(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.squeeze(2).transpose(1, 2).contiguous()
        x = self.pre_rnn_norm(x)
        if self.input_proj is not None:
            x = self.input_proj(x)
        return x

    def forward(self, x):
        x = self.encode_sequence(x)
        x, _ = self.rnn(x)
        return x


@dataclass
class CRNNConfig:
    in_channels: int = 3
    lstm_hidden: int = 256
    lstm_layers: int = 2
    layer1_width_stride: int = 2
    layer2_width_stride: int = 2
    rnn_input_dim: int = 512
    lstm_proj_size: int = 0
    layer5_blocks: int = 3
    use_rnn_skip: bool = False
    aux_ctc_weight: float = 0.0
    use_locked_dropout: bool = False
    num_dropout_samples: int = 1
    use_cosine_classifier: bool = False
    use_weight_norm_classifier: bool = False
    use_head_gate: bool = False
    use_post_rnn_refine: bool = False
    post_rnn_kernel: int = 3
    blank_logit_bias: float = 0.0
    cosine_scale_init: float = 12.0
    dropout: float = 0.1


class CRNN(nn.Module):
    def __init__(self, config: CRNNConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.encoder = ResNetAsterEncoder(
            in_channels=config.in_channels,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            layer1_width_stride=config.layer1_width_stride,
            layer2_width_stride=config.layer2_width_stride,
            rnn_input_dim=config.rnn_input_dim,
            lstm_proj_size=config.lstm_proj_size,
            layer5_blocks=config.layer5_blocks,
        )
        self.rnn_skip = None
        if config.use_rnn_skip:
            self.rnn_skip = nn.Linear(self.encoder.rnn_input_dim, self.encoder.out_channels)
        self.aux_classifier = None
        if config.aux_ctc_weight > 0.0:
            self.aux_classifier = nn.Linear(self.encoder.rnn_input_dim, num_classes)
        self.post_rnn_refine = None
        if config.use_post_rnn_refine:
            self.post_rnn_refine = ResidualTemporalRefine(self.encoder.out_channels, config.post_rnn_kernel)
        self.sequence_norm = nn.LayerNorm(self.encoder.out_channels)
        self.head_gate = None
        if config.use_head_gate:
            gate_hidden = max(self.encoder.out_channels // 8, 64)
            self.head_gate = nn.Sequential(
                nn.Linear(self.encoder.out_channels, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, self.encoder.out_channels),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.head_gate[2].weight)
            nn.init.zeros_(self.head_gate[2].bias)
        self.dropout = LockedDropout(config.dropout) if config.use_locked_dropout else nn.Dropout(config.dropout)
        self.classifier = None
        self.classifier_weight = None
        self.classifier_bias = None
        self.logit_scale = None
        self.blank_logit_offset = 0.0
        if config.use_cosine_classifier:
            self.classifier_weight = nn.Parameter(torch.empty(num_classes, self.encoder.out_channels))
            self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(config.cosine_scale_init), dtype=torch.float32))
            nn.init.normal_(self.classifier_weight, std=0.02)
        else:
            self.classifier = nn.Linear(self.encoder.out_channels, num_classes)
            if config.use_weight_norm_classifier:
                self.classifier = nn.utils.parametrizations.weight_norm(self.classifier, name="weight")
        if config.blank_logit_bias != 0.0:
            target_bias = self.classifier.bias if self.classifier is not None else self.classifier_bias
            if target_bias is not None:
                with torch.no_grad():
                    target_bias[0] = config.blank_logit_bias

    def set_dropout_rate(self, p):
        self.dropout.p = max(0.0, min(0.95, float(p)))

    def set_blank_logit_offset(self, offset):
        self.blank_logit_offset = float(offset)

    def project_logits(self, x):
        if self.classifier is not None:
            logits = self.classifier(x)
        else:
            features = F.normalize(x, dim=-1)
            weights = F.normalize(self.classifier_weight, dim=-1)
            scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
            logits = scale * F.linear(features, weights) + self.classifier_bias
        if self.blank_logit_offset != 0.0:
            logits[..., 0] = logits[..., 0] + self.blank_logit_offset
        return logits

    def classify(self, x):
        if self.training and self.config.num_dropout_samples > 1:
            logits = 0.0
            for _ in range(self.config.num_dropout_samples):
                logits = logits + self.project_logits(self.dropout(x))
            return logits / self.config.num_dropout_samples
        return self.project_logits(self.dropout(x))

    def forward_features(self, images):
        sequence = self.encoder.encode_sequence(images)
        x, _ = self.encoder.rnn(sequence)
        if self.rnn_skip is not None:
            x = x + self.rnn_skip(sequence)
        if self.post_rnn_refine is not None:
            x = self.post_rnn_refine(x)
        x = self.sequence_norm(x)
        if self.head_gate is not None:
            x = x * (2.0 * self.head_gate(x))
        return sequence, x

    def forward_with_aux(self, images):
        sequence, x = self.forward_features(images)
        logits = self.classify(x)
        aux_logits = self.aux_classifier(sequence) if self.aux_classifier is not None else None
        return logits, aux_logits

    def forward(self, images):
        _, x = self.forward_features(images)
        logits = self.classify(x)
        return logits


def ctc_loss(logits, flat_targets, target_lengths):
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    input_lengths = torch.full(
        size=(logits.size(0),),
        fill_value=logits.size(1),
        dtype=torch.long,
        device=logits.device,
    )
    return F.ctc_loss(
        log_probs,
        flat_targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )


def apply_width_mask(images, mask_width, mask_prob):
    if mask_width <= 0 or mask_prob <= 0.0:
        return images
    batch, _, _, width = images.shape
    if mask_width >= width:
        return images
    apply_mask = torch.rand(batch, device=images.device) < mask_prob
    if not apply_mask.any():
        return images
    starts = torch.randint(0, width - mask_width + 1, (batch,), device=images.device)
    masked = images.clone()
    for idx in torch.nonzero(apply_mask, as_tuple=False).flatten():
        start = int(starts[idx].item())
        masked[idx, :, :, start : start + mask_width] = 0.0
    return masked


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].lerp_(param.detach(), 1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def get_lr(progress):
    if progress < WARMUP_RATIO:
        return LR * (progress / max(WARMUP_RATIO, 1e-8))
    cosine_progress = (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-8)
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return LR * (FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine)


def env_int(name, default):
    value = os.getenv(f"OCR_AR_{name}")
    return int(value) if value is not None else default


def env_float(name, default):
    value = os.getenv(f"OCR_AR_{name}")
    return float(value) if value is not None else default


def env_bool(name, default):
    value = os.getenv(f"OCR_AR_{name}")
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean override for {name}: {value!r}")


def env_float_tuple(name, default):
    value = os.getenv(f"OCR_AR_{name}")
    if value is None:
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != len(default):
        raise ValueError(f"Invalid tuple override for {name}: {value!r}")
    return tuple(float(part) for part in parts)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Optimization
TOTAL_BATCH_SIZE = env_int("TOTAL_BATCH_SIZE", 32)
DEVICE_BATCH_SIZE = env_int("DEVICE_BATCH_SIZE", 32)
EVAL_BATCH_SIZE = env_int("EVAL_BATCH_SIZE", 128)
LR = env_float("LR", 6e-4)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.0)
BETAS = env_float_tuple("BETAS", (0.9, 0.99))
WARMUP_RATIO = env_float("WARMUP_RATIO", 0.0)
FINAL_LR_FRAC = env_float("FINAL_LR_FRAC", 0.1)
GRAD_CLIP = env_float("GRAD_CLIP", 5.0)
EMA_DECAY = env_float("EMA_DECAY", 0.0)

# Model
LSTM_HIDDEN = env_int("LSTM_HIDDEN", 256)
LSTM_LAYERS = env_int("LSTM_LAYERS", 2)
LAYER1_WIDTH_STRIDE = env_int("LAYER1_WIDTH_STRIDE", 2)
LAYER2_WIDTH_STRIDE = env_int("LAYER2_WIDTH_STRIDE", 2)
RNN_INPUT_DIM = env_int("RNN_INPUT_DIM", 512)
LSTM_PROJ_SIZE = env_int("LSTM_PROJ_SIZE", 0)
LAYER5_BLOCKS = env_int("LAYER5_BLOCKS", 3)
USE_RNN_SKIP = env_bool("USE_RNN_SKIP", False)
AUX_CTC_WEIGHT = env_float("AUX_CTC_WEIGHT", 0.0)
USE_LOCKED_DROPOUT = env_bool("USE_LOCKED_DROPOUT", False)
NUM_DROPOUT_SAMPLES = env_int("NUM_DROPOUT_SAMPLES", 1)
USE_COSINE_CLASSIFIER = env_bool("USE_COSINE_CLASSIFIER", False)
USE_WEIGHT_NORM_CLASSIFIER = env_bool("USE_WEIGHT_NORM_CLASSIFIER", False)
USE_HEAD_GATE = env_bool("USE_HEAD_GATE", False)
USE_POST_RNN_REFINE = env_bool("USE_POST_RNN_REFINE", False)
POST_RNN_KERNEL = env_int("POST_RNN_KERNEL", 3)
BLANK_LOGIT_BIAS = env_float("BLANK_LOGIT_BIAS", 0.0)
BLANK_LOGIT_OFFSET_START = env_float("BLANK_LOGIT_OFFSET_START", 0.0)
BLANK_LOGIT_OFFSET_END = env_float("BLANK_LOGIT_OFFSET_END", BLANK_LOGIT_OFFSET_START)
COSINE_SCALE_INIT = env_float("COSINE_SCALE_INIT", 12.0)
WIDTH_MASK = env_int("WIDTH_MASK", 0)
WIDTH_MASK_PROB = env_float("WIDTH_MASK_PROB", 0.0)
DROPOUT = env_float("DROPOUT", 0.1)
DROPOUT_START = env_float("DROPOUT_START", DROPOUT)
DROPOUT_END = env_float("DROPOUT_END", DROPOUT_START)

# Misc
SEED = env_int("SEED", 1337)
USE_AMP = env_bool("USE_AMP", False)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

prepare_cache()

t_start = time.time()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()

codec = OCRCodec.from_file(CODEC_PATH)
config = CRNNConfig(
    in_channels=3,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    layer1_width_stride=LAYER1_WIDTH_STRIDE,
    layer2_width_stride=LAYER2_WIDTH_STRIDE,
    rnn_input_dim=RNN_INPUT_DIM,
    lstm_proj_size=LSTM_PROJ_SIZE,
    layer5_blocks=LAYER5_BLOCKS,
    use_rnn_skip=USE_RNN_SKIP,
    aux_ctc_weight=AUX_CTC_WEIGHT,
    use_locked_dropout=USE_LOCKED_DROPOUT,
    num_dropout_samples=NUM_DROPOUT_SAMPLES,
    use_cosine_classifier=USE_COSINE_CLASSIFIER,
    use_weight_norm_classifier=USE_WEIGHT_NORM_CLASSIFIER,
    use_head_gate=USE_HEAD_GATE,
    use_post_rnn_refine=USE_POST_RNN_REFINE,
    post_rnn_kernel=POST_RNN_KERNEL,
    blank_logit_bias=BLANK_LOGIT_BIAS,
    cosine_scale_init=COSINE_SCALE_INIT,
    dropout=DROPOUT,
)

assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

model = CRNN(config, num_classes=codec.num_classes).to(device)
model.set_dropout_rate(DROPOUT_START)
model.set_blank_logit_offset(BLANK_LOGIT_OFFSET_START)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
ema_model = None
if EMA_DECAY > 0.0:
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
num_params = count_parameters(model)

train_loader = make_dataloader(codec, DEVICE_BATCH_SIZE, "train", infinite=True)
autocast_ctx = (
    torch.autocast(device_type="cuda", dtype=torch.float16)
    if USE_AMP and device.type == "cuda"
    else nullcontext()
)
scaler = torch.amp.GradScaler("cuda", enabled=True) if USE_AMP and device.type == "cuda" else None

print(f"Device: {device}")
print(f"Model config: {asdict(config)}")
print(f"Parameters: {num_params:,}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(
    "Optimization config: "
    f"total_batch={TOTAL_BATCH_SIZE}, device_batch={DEVICE_BATCH_SIZE}, "
    f"eval_batch={EVAL_BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}, "
    f"betas={BETAS}, warmup_ratio={WARMUP_RATIO}, final_lr_frac={FINAL_LR_FRAC}, "
    f"grad_clip={GRAD_CLIP}, ema_decay={EMA_DECAY}, use_amp={USE_AMP}, "
    f"aux_ctc_weight={AUX_CTC_WEIGHT}, width_mask={WIDTH_MASK}, width_mask_prob={WIDTH_MASK_PROB}, "
    f"use_cosine_classifier={USE_COSINE_CLASSIFIER}, use_weight_norm_classifier={USE_WEIGHT_NORM_CLASSIFIER}, "
    f"use_head_gate={USE_HEAD_GATE}, use_post_rnn_refine={USE_POST_RNN_REFINE}, "
    f"post_rnn_kernel={POST_RNN_KERNEL}, blank_logit_bias={BLANK_LOGIT_BIAS}, "
    f"blank_logit_offset_start={BLANK_LOGIT_OFFSET_START}, blank_logit_offset_end={BLANK_LOGIT_OFFSET_END}, "
    f"dropout_start={DROPOUT_START}, dropout_end={DROPOUT_END}, cosine_scale_init={COSINE_SCALE_INIT}"
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
smooth_loss = 0.0
step = 0
samples_seen = 0
epoch = 1

while True:
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr = get_lr(progress)
    current_dropout = DROPOUT_START + (DROPOUT_END - DROPOUT_START) * progress
    current_blank_logit_offset = BLANK_LOGIT_OFFSET_START + (BLANK_LOGIT_OFFSET_END - BLANK_LOGIT_OFFSET_START) * progress
    model.set_dropout_rate(current_dropout)
    model.set_blank_logit_offset(current_blank_logit_offset)
    if ema_model is not None:
        ema_model.set_dropout_rate(current_dropout)
        ema_model.set_blank_logit_offset(current_blank_logit_offset)
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()

    train_loss_value = 0.0
    for _ in range(grad_accum_steps):
        images, flat_targets, target_lengths, _, epoch = next(train_loader)
        images = images.to(device, non_blocking=True)
        flat_targets = flat_targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        images = apply_width_mask(images, WIDTH_MASK, WIDTH_MASK_PROB)

        with autocast_ctx:
            if AUX_CTC_WEIGHT > 0.0:
                logits, aux_logits = model.forward_with_aux(images)
                loss = ctc_loss(logits, flat_targets, target_lengths)
                loss = loss + AUX_CTC_WEIGHT * ctc_loss(aux_logits, flat_targets, target_lengths)
            else:
                logits = model(images)
                loss = ctc_loss(logits, flat_targets, target_lengths)

        train_loss_value = loss.detach().item()
        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        samples_seen += images.size(0)

    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    if ema_model is not None:
        update_ema_model(ema_model, model, EMA_DECAY)

    torch.cuda.synchronize() if device.type == "cuda" else None
    dt = time.time() - t0
    total_training_time += dt

    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss_value
    debiased_loss = smooth_loss / (1.0 - ema_beta ** (step + 1))
    remaining = max(0.0, TIME_BUDGET - total_training_time)
    samples_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-8)

    print(
        f"\rstep {step:05d} ({100.0 * progress:.1f}%) | "
        f"loss: {debiased_loss:.4f} | lr: {lr:.6f} | "
        f"dt: {dt*1000:.0f}ms | samp/sec: {samples_per_sec:.0f} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s   ",
        end="",
        flush=True,
    )

    step += 1
    if total_training_time >= TIME_BUDGET:
        break

print()


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

eval_model = ema_model if ema_model is not None else model
metrics = evaluate_cer(eval_model, codec, EVAL_BATCH_SIZE, device)
t_end = time.time()
peak_vram_mb = (
    torch.cuda.max_memory_allocated() / 1024 / 1024
    if device.type == "cuda"
    else 0.0
)

print("---")
print(f"val_cer:          {metrics['val_cer']:.6f}")
print(f"val_word_acc:     {metrics['val_word_acc']:.6f}")
print(f"val_1_minus_ned:  {metrics['val_1_minus_ned']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"samples_seen_K:   {samples_seen / 1e3:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
