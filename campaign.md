# Mar 8 Campaign

This file tracks the current OCR autoresearch campaign on branch `autoresearch/mar8`.

## Scope

- Minimum total rounds: 36
- Current logged rounds: 12
- Next round index: 13
- Remote branch: `origin/autoresearch/mar8`
- Push cadence: push after every 4 newly completed rounds
- Local runtime command: use `python prepare.py` and `python train.py` in this workspace

## Current Best Result

- Commit: `a84f80a`
- Description: `remove warmup`
- `val_cer`: `0.754693`
- `word_acc`: `0.045852`
- `memory_gb`: `3.4`

## Setup Status

- Benchmark cache prepared at `C:\Users\ZHD-Y9000X\.cache\ocr-autoresearch`
- `train.py` currently supports `OCR_AR_*` environment overrides for fast experiments
- `pyproject.toml` is configured for editable install
- NumPy is pinned below 2 to avoid the current pandas ABI mismatch in this environment

## Per-Round Protocol

1. Propose exactly one experimental idea.
2. Commit the change or record the explicit `OCR_AR_*` override used for that round.
3. Run the experiment and log metrics to `results.tsv`.
4. Write a short round analysis before planning the next move.
5. The round analysis must state:
   - the metric delta versus the current best result
   - the likely keypoint behind the change
   - whether the evidence is clear or ambiguous
   - the next action and why it follows from the observed keypoint
6. Keep or discard the idea based on `val_cer`, with simplicity preferred when gains are close.
7. Push the branch after every 4 newly completed rounds.
