"""
One-time data preparation and fixed evaluation utilities for ocr-autoresearch.

Downloads the ICDAR2015 cropped text-recognition benchmark from Hugging Face,
writes a small cached copy to ~/.cache/ocr-autoresearch/, freezes the codec,
and exposes fixed dataloading/evaluation helpers for train.py.
"""

from __future__ import annotations

import json
import os
import random
import string
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Fixed benchmark constants (do not modify during experiments)
# ---------------------------------------------------------------------------

IMG_HEIGHT = 32
IMG_WIDTH = 128
MAX_TEXT_LENGTH = 25
TIME_BUDGET = 300
DATASET_NAME = "MiXaiLL76/ICDAR2015_OCR"
SEED = 1337

# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "ocr-autoresearch"
DATA_DIR = CACHE_DIR / "icdar2015"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TRAIN_MANIFEST = DATA_DIR / "train.jsonl"
VAL_MANIFEST = DATA_DIR / "val.jsonl"
CODEC_PATH = DATA_DIR / "codec.json"
META_PATH = DATA_DIR / "meta.json"

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 2


def normalize_for_eval(text: str) -> str:
    normalized = text.replace(" ", "")
    normalized = "".join(
        ch for ch in normalized if ch in (string.digits + string.ascii_letters)
    )
    normalized = normalized.lower()
    return normalized if normalized else text.lower().replace(" ", "")


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


class OCRCodec:
    def __init__(self, characters: list[str]):
        self.characters = characters
        self.blank_index = 0
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    @property
    def num_classes(self) -> int:
        return self.vocab_size + 1

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx_to_char[idx] for idx in ids if idx in self.idx_to_char)

    def ctc_decode(self, logits: torch.Tensor) -> list[str]:
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        outputs = []
        for row in pred_ids:
            collapsed = []
            previous = self.blank_index
            for idx in row:
                if idx != self.blank_index and idx != previous:
                    collapsed.append(idx)
                previous = idx
            outputs.append(self.decode(collapsed))
        return outputs

    def to_file(self, path: Path) -> None:
        path.write_text(
            json.dumps({"characters": self.characters}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: Path) -> "OCRCodec":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["characters"])


def _artifacts_exist() -> bool:
    if not all(path.exists() for path in (TRAIN_MANIFEST, VAL_MANIFEST, CODEC_PATH, META_PATH)):
        return False
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return metadata.get("layout_version") == 2


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _dataset_rows():
    dataset = load_dataset(DATASET_NAME)
    rows = []
    for split_name in ("train", "train_numbers"):
        for sample in dataset[split_name]:
            text = sample["text"]
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Label exceeds MAX_TEXT_LENGTH={MAX_TEXT_LENGTH}: {text!r}"
                )
            rows.append({"image": sample["image"], "text": text})

    rng = random.Random(SEED)
    rng.shuffle(rows)
    val_size = max(256, int(round(len(rows) * 0.1)))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    return train_rows, val_rows


def _save_split(rows: list[dict[str, object]], image_dir: Path, split_name: str) -> list[dict[str, str]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for idx, row in enumerate(rows):
        image = row["image"].convert("RGB")
        filename = f"{idx:05d}.png"
        rel_path = f"{split_name}/{filename}"
        out_path = image_dir / filename
        image.save(out_path)
        manifest_rows.append({"image": rel_path, "text": row["text"]})
    return manifest_rows


def prepare_cache() -> None:
    if _artifacts_exist():
        return

    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = _dataset_rows()
    train_manifest = _save_split(train_rows, TRAIN_DIR, "train")
    val_manifest = _save_split(val_rows, VAL_DIR, "val")
    _write_manifest(TRAIN_MANIFEST, train_manifest)
    _write_manifest(VAL_MANIFEST, val_manifest)

    charset = sorted({char for row in train_manifest + val_manifest for char in row["text"]})
    codec = OCRCodec(charset)
    codec.to_file(CODEC_PATH)

    metadata = {
        "layout_version": 2,
        "dataset_name": DATASET_NAME,
        "train_samples": len(train_manifest),
        "val_samples": len(val_manifest),
        "max_text_length": MAX_TEXT_LENGTH,
        "codec_size": codec.vocab_size,
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "created_at_unix": time.time(),
    }
    META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(train_manifest)} train and {len(val_manifest)} val samples in {time.time() - t0:.1f}s")


class CachedICDARDataset(Dataset):
    def __init__(self, split: str, codec: OCRCodec):
        assert split in {"train", "val"}
        manifest_path = TRAIN_MANIFEST if split == "train" else VAL_MANIFEST
        self.rows = _read_manifest(manifest_path)
        self.codec = codec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(DATA_DIR / row["image"]).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = tensor.sub(0.5).div(0.5)
        target = self.codec.encode(row["text"])
        return tensor, target, row["text"]


def _collate_batch(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return images, flat_targets, target_lengths, list(texts)


def make_dataloader(codec: OCRCodec, batch_size: int, split: str, infinite: bool = False):
    dataset = CachedICDARDataset(split, codec)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=DEFAULT_NUM_WORKERS > 0,
        collate_fn=_collate_batch,
    )
    if not infinite:
        return loader

    def iterator():
        epoch = 1
        while True:
            for batch in loader:
                yield (*batch, epoch)
            epoch += 1

    return iterator()


@torch.no_grad()
def evaluate_cer(model, codec: OCRCodec, batch_size: int, device: torch.device):
    model.eval()
    loader = make_dataloader(codec, batch_size, "val", infinite=False)

    total_chars = 0
    total_words = 0
    total_edits = 0
    total_norm_edits = 0.0
    correct_words = 0

    for images, _, _, texts in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predictions = codec.ctc_decode(logits)
        for pred, target in zip(predictions, texts):
            pred = normalize_for_eval(pred)
            target = normalize_for_eval(target)
            edits = levenshtein_distance(pred, target)
            total_edits += edits
            total_chars += max(len(target), 1)
            total_words += 1
            correct_words += int(pred == target)
            total_norm_edits += edits / max(len(pred), len(target), 1)

    return {
        "val_cer": total_edits / max(total_chars, 1),
        "val_word_acc": correct_words / max(total_words, 1),
        "val_1_minus_ned": 1.0 - total_norm_edits / max(total_words, 1),
        "num_samples": total_words,
    }


def load_metadata() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def print_summary() -> None:
    meta = load_metadata()
    codec = OCRCodec.from_file(CODEC_PATH)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Benchmark dataset: {meta['dataset_name']}")
    print(f"Train samples: {meta['train_samples']}")
    print(f"Val samples:   {meta['val_samples']}")
    print(f"Image size:    {meta['img_height']}x{meta['img_width']}")
    print(f"Codec size:    {codec.vocab_size}")
    print(f"Max text len:  {meta['max_text_length']}")


if __name__ == "__main__":
    prepare_cache()
    print_summary()
    print("Done! Ready to train.")
