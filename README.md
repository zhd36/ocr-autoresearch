# ocr-autoresearch

`ocr-autoresearch` is an OCR adaptation of the `autoresearch` idea: give an AI agent a small but real training setup, let it edit a single `train.py`, train for a fixed 5-minute budget, measure whether the run improved, keep or discard, and repeat.

Instead of language modeling, this repo focuses on **scene text recognition**:

- Task: cropped text recognition on the **ICDAR2015** benchmark
- Dataset source: Hugging Face dataset [`MiXaiLL76/ICDAR2015_OCR`](https://huggingface.co/datasets/MiXaiLL76/ICDAR2015_OCR), derived from ICDAR2015 and marked as **CC BY 4.0**
- Baseline model: a compact **CRNN + CTC** recognizer inspired by OpenOCR's `crnn_ctc.yml` and `ResNet_ASTER` encoder
- Primary metric: **val_cer** (validation character error rate after OpenOCR-style text normalization), lower is better
- Secondary metric: **val_word_acc**

The repo is deliberately small. Only one file is meant to be edited during research:

- `prepare.py`: fixed constants, one-time data prep, codec, dataloader, evaluation
- `train.py`: the single file the agent edits
- `program.md`: instructions for the autonomous research loop

## Why This Task

This benchmark is small enough to iterate quickly on a single consumer GPU, but still real OCR:

- the labels are full text strings, not class IDs
- the model uses sequence recognition with CTC, not plain image classification
- the evaluation is OCR-native: edit distance and exact-match accuracy

Compared with document parsing, detection, or 0.1B-scale recognition models, this setup is much easier to compress into an `autoresearch`-style repository.

## Quick Start

Requirements:

- Python 3.10+
- one NVIDIA GPU is recommended
- `uv` is optional; plain `pip` also works

```bash
# 1. Install dependencies
uv sync

# 2. Download and cache the benchmark (~1-2 min)
uv run prepare.py

# 3. Run one baseline experiment (~5 min training + eval)
uv run train.py
```

If you prefer `pip`:

```bash
python -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
python prepare.py
python train.py
```

## How It Works

The benchmark is fixed in `prepare.py`:

- images are resized to `32x128`
- the character set is frozen from the benchmark data
- training uses a fixed 90% split of the official ICDAR2015 train data plus the numeric subset
- validation uses the remaining fixed 10% holdout split
- training always stops after `300` seconds of actual training loop time
- evaluation lowercases text, removes spaces, and filters non-alphanumeric symbols before computing metrics, matching the spirit of OpenOCR's `RecMetric`

The agent is supposed to modify only `train.py`. Fair game includes:

- model width and depth
- optimizer and schedule
- batch size and accumulation
- normalization, dropout, residual paths
- decoder head details

The agent should not change:

- dataset contents
- preprocessing
- evaluation logic
- the 5-minute research budget

## Output Format

At the end of each run, `train.py` prints a summary like:

```text
---
val_cer:          0.123456
val_word_acc:     0.654321
val_1_minus_ned:  0.812345
training_seconds: 300.1
total_seconds:    318.7
peak_vram_mb:     1834.2
samples_seen_K:   92.4
num_steps:        723
num_params_M:     8.7
```

Lower `val_cer` is better. Use it as the main criterion for keep/discard decisions.

## Repo Design

This repository follows the same design principles as `karpathy/autoresearch`:

- single editable training file
- fixed wall-clock training budget
- fixed benchmark and metric
- branch-based experiment loop
- simple enough for an autonomous coding agent to understand end to end

## Attribution

- The autonomous-research workflow is inspired by [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch).
- The OCR task and baseline are inspired by [`Topdu/OpenOCR`](https://github.com/Topdu/OpenOCR), especially the `crnn_ctc` configuration and `ResNet_ASTER` encoder.
