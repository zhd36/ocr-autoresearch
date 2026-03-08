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

### Round 15 - `3de4fdb` - mild weight decay

- Result: `val_cer=0.765358`, `word_acc=0.056769`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.010665` CER worse
- Keypoint: adding `weight_decay=1e-4` improved exact-match word accuracy but hurt character error. The most likely explanation is that this amount of regularization nudges predictions toward cleaner all-or-nothing strings on easier samples while reducing fine-grained character fitting under the tight 5-minute budget.
- Evidence: moderately clear. The CER regression is too large to keep, but the word-accuracy gain suggests the regularization direction itself may still be viable at a weaker setting.
- Next action: try a smaller `weight_decay=5e-5` to see whether the word-level gain can be retained without paying as much CER cost.

### Round 16 - `41bad42` - lighter weight decay

- Result: `val_cer=0.774744`, `word_acc=0.050218`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.020051` CER worse
- Keypoint: reducing weight decay did not recover the CER loss. The direction itself appears mismatched to this short-budget setting: regularization is consuming fitting capacity faster than it buys useful generalization.
- Evidence: clear. Two consecutive weight-decay settings both miss the best by a meaningful margin, and the lighter setting gets even worse CER.
- Next action: stop exploring weight decay and move to base learning-rate tuning at the proven `batch=32`, since optimization speed is now a more plausible bottleneck than overfitting.

### Round 17 - `7ed342e` - lower lr

- Result: `val_cer=0.762799`, `word_acc=0.045852`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.008106` CER worse
- Keypoint: lowering the base learning rate to `5e-4` did not stabilize into a better solution; it mostly slowed progress within the fixed time budget. That points to optimization speed, not excessive aggressiveness, as the more relevant constraint here.
- Evidence: clear enough to bracket one side of the LR search. The regression is smaller than the weight-decay failures, but still comfortably worse than the best run.
- Next action: test a modestly higher LR (`7e-4`) to see whether the current best sits just below the useful aggressiveness limit or already near the optimum.

### Round 18 - `1884e34` - higher lr

- Result: `val_cer=0.764078`, `word_acc=0.045852`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.009385` CER worse
- Keypoint: increasing LR to `7e-4` also misses the best, so the current `6e-4` is not simply too conservative. The LR bracket now looks well centered: both lower and higher settings regress by a similar amount.
- Evidence: clear. With both sides of the bracket worse, the search should move on instead of spending more rounds on nearby scalar LR tweaks.
- Next action: pivot to dropout-strength tuning, because regularization via classifier dropout is cheaper and more targeted than weight decay and was not yet mapped around the current best regime.

### Round 19 - `d00bb29` - lighter dropout

- Result: `val_cer=0.779437`, `word_acc=0.039301`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.024744` CER worse
- Keypoint: reducing dropout from `0.1` to `0.05` sharply hurts CER, which suggests the current head is not over-regularized. In this setup, that dropout is likely doing real work against overconfident CTC decoding rather than merely slowing fitting.
- Evidence: very clear. The regression is large enough to rule out this side of the dropout search.
- Next action: test `dropout=0.15` to see whether the best region is centered around `0.1` or whether slightly stronger dropout can improve robustness without repeating the weight-decay failure mode.

### Round 20 - `a6d2344` - stronger dropout

- Result: `val_cer=0.783703`, `word_acc=0.039301`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.029010` CER worse
- Keypoint: stronger dropout is even worse than lighter dropout, so the optimum is not merely “some regularization”; it is specifically close to the current `0.1`. This axis is now well bracketed.
- Evidence: very clear. Both sides of the dropout sweep regress substantially, which makes the default value look robust rather than accidental.
- Next action: stop spending rounds on dropout and move to AdamW momentum settings (`betas`), since short-budget sequence training can be sensitive to adaptation timescales even when LR is already tuned.

### Round 21 - `e930dc0` - shorter beta2

- Result: `val_cer=0.781143`, `word_acc=0.037118`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.026450` CER worse
- Keypoint: shortening AdamW's second-moment timescale to `beta2=0.95` does not help this model exploit the 5-minute budget; it destabilizes enough to wipe out the hoped-for faster adaptation.
- Evidence: clear. The regression is large, and together with the earlier LR/dropout failures it suggests the current scalar optimization settings are already near a local optimum.
- Next action: stop pure hyperparameter search and switch to model-structure experiments, starting with a bidirectional GRU head that may trade some recurrent expressivity for more training speed under the fixed wall-clock budget.
