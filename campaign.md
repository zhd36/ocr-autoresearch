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

## Round Log

### Round 13 - `74d31dd` - total batch 16

- Result: `val_cer=0.770904`, `word_acc=0.024017`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.016211` CER worse
- Keypoint: the earlier gains from reducing total batch did not continue monotonically. Dropping from 32 to 16 doubled optimizer-step count within 5 minutes, but the higher update noise and lower total sample exposure did not improve recognition quality.
- Evidence: fairly clear. CER worsened materially and word accuracy also dropped, so this is not just metric noise.
- Next action: test `total batch 24` to map whether the useful regime ends near 32 or whether there is still a smaller-batch sweet spot between 16 and 32.

### Round 14 - `23959b9` - total batch 24

- Result: `val_cer=0.761519`, `word_acc=0.034934`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.006826` CER worse
- Keypoint: moving from 32 to 24 recovers much of the loss seen at 16, but still does not beat the current best. This suggests the main batch-size gain was already captured by getting down to 32, and further reduction now trades away more stability than it returns in useful extra updates.
- Evidence: clear enough to stop spending more rounds on smaller-batch search. The result sits between 16 and 32 exactly as expected and does not indicate a hidden win below 32.
- Next action: pivot away from batch size and test mild regularization at the current best geometry, starting with a small `weight_decay`, because that changes generalization pressure without undoing the proven 32-batch regime.
