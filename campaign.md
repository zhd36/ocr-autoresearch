# Mar 8 Campaign

This file tracks the current OCR autoresearch campaign on branch `autoresearch/mar8`.

## Scope

- Minimum total rounds: 300
- Current logged rounds: 36
- Next round index: 37
- Remote branch: `origin/autoresearch/mar8`
- Push cadence: push after every 4 newly completed rounds
- Local runtime command: use `python prepare.py` and `python train.py` in this workspace

## Current Best Result

- Commit: `fe69bc3`
- Description: `GroupNorm encoder blocks`
- `val_cer`: `0.675341`
- `word_acc`: `0.104803`
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

### Round 22 - `1dfba6b` - GRU head instead of LSTM

- Result: `val_cer=0.772184`, `word_acc=0.030568`, `memory_gb=3.4`, `status=discard`
- Delta vs best `a84f80a`: `+0.017491` CER worse
- Keypoint: replacing the bidirectional LSTM with a bidirectional GRU reduced parameters (`16.2M -> 15.4M`) but did not increase useful throughput enough to matter, and the weaker sequence model hurt recognition quality. This task seems more sensitive to recurrent expressivity than to that small parameter reduction.
- Evidence: clear. CER drops materially, word accuracy drops, and step count did not improve enough to justify the trade.
- Next action: keep the LSTM backbone and try a lower-risk structural refinement on top of it, starting with a sequence `LayerNorm` before the classifier to improve feature conditioning without weakening temporal modeling.

### Round 23 - `7192ce2` - sequence LayerNorm before classifier

- Result: `val_cer=0.732509`, `word_acc=0.069869`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `a84f80a`: `-0.022184` CER better
- Keypoint: feature conditioning at the sequence head was the missing piece. Normalizing the bidirectional LSTM output before dropout and classification materially improves both CER and exact-match accuracy, which strongly suggests the classifier was previously seeing poorly scaled per-timestep features.
- Evidence: very clear. This is a large CER gain with matching word-accuracy improvement and no memory penalty.
- Next action: keep LayerNorm as the new base and test a lightweight temporal refinement block on top of it, because the success of head-side conditioning suggests further sequence-local smoothing may yield additional gains without replacing the strong LSTM backbone.

### Round 24 - `6435adc` - residual temporal refine after LayerNorm

- Result: `val_cer=0.773891`, `word_acc=0.054585`, `memory_gb=3.4`, `status=discard`
- Delta vs best `7192ce2`: `+0.041382` CER worse
- Keypoint: the gain from LayerNorm does not extend to adding extra local temporal mixing. The model likely already has enough sequence context from the bidirectional LSTM, and the added conv block introduces unnecessary transformation noise or optimization burden at the head.
- Evidence: very clear. CER regresses heavily despite similar memory use, so this is not a marginal tradeoff.
- Next action: keep LayerNorm as the new base, but stay closer to its apparent benefit by trying per-timestep channel refinement next instead of extra temporal mixing.

### Round 25 - `818e0ed` - residual MLP head after LayerNorm

- Result: `val_cer=0.765358`, `word_acc=0.074236`, `memory_gb=3.4`, `status=discard`
- Delta vs best `7192ce2`: `+0.032849` CER worse
- Keypoint: a richer pointwise head again improves word-level exact matches while hurting CER, which suggests the added head capacity is helping confident easy predictions without improving fine-grained character alignment. The useful signal from LayerNorm is therefore conditioning, not “make the head deeper.”
- Evidence: clear. CER worsens materially even though word accuracy reaches a new local high.
- Next action: move the next structural change upstream into the encoder with lightweight channel attention, because head-side complexity is no longer matching the main metric.

### Round 26 - `cbf0971` - add SE channel attention to Aster blocks

- Result: `val_cer=0.713737`, `word_acc=0.087336`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `7192ce2`: `-0.018772` CER better
- Keypoint: encoder-side channel reweighting is strongly beneficial. This confirms that the model was not mainly lacking head complexity; it was lacking better feature selection before the sequence model/classifier consumed the representation.
- Evidence: very clear. CER and word accuracy both improve substantially with only a small parameter increase and no meaningful memory change.
- Next action: keep SE as the new base and continue exploring encoder-side structural refinements that complement it, rather than adding more head-side complexity.

### Round 27 - `ee25656` - SiLU activations in encoder

- Result: `val_cer=0.766212`, `word_acc=0.067686`, `memory_gb=3.4`, `status=discard`
- Delta vs best `cbf0971`: `+0.052475` CER worse
- Keypoint: smoother activations did not complement the new SE-equipped encoder; they degraded the proven feature pipeline substantially. The useful encoder improvement appears to be selective channel emphasis, not a broad nonlinearity swap.
- Evidence: very clear. CER regresses heavily with no compensating gain elsewhere.
- Next action: revert to ReLU and stay focused on feature conditioning, testing another normalization point before the recurrent stack rather than changing the encoder's whole activation regime.

### Round 28 - `ae45646` - pre-RNN LayerNorm on encoder sequence

- Result: `val_cer=0.691126`, `word_acc=0.120087`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `cbf0971`: `-0.022611` CER better
- Keypoint: the conditioning story extends upstream. Normalizing the CNN sequence features before they enter the bidirectional LSTM gives a major boost, which implies the recurrent stack was previously spending capacity compensating for feature-scale variation from the encoder.
- Evidence: extremely clear. This is a large, clean improvement on both CER and word accuracy with negligible overhead.
- Next action: treat “careful feature conditioning across stage boundaries” as the main design principle for the next rounds. The next experiments should test whether both normalization points are needed and whether similar lightweight conditioning can simplify or further improve the model.

### Round 29 - `d91dab7` - late-stage SE only

- Result: `val_cer=0.824232`, `word_acc=0.024017`, `memory_gb=3.4`, `status=discard`
- Delta vs best `ae45646`: `+0.133106` CER worse
- Keypoint: the SE gain is not concentrated only in late semantic blocks. Removing attention from the shallow stages destroys performance, which means early channel selection is part of the successful representation pipeline rather than optional overhead.
- Evidence: decisive. The regression is massive even though training speed improves somewhat.
- Next action: keep SE across the full encoder and explore a different normalization family inside the CNN stack, since conditioning remains the strongest positive theme but pruning early SE clearly breaks it.

### Round 30 - `fe69bc3` - GroupNorm encoder blocks

- Result: `val_cer=0.675341`, `word_acc=0.104803`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `ae45646`: `-0.015785` CER better
- Keypoint: replacing `BatchNorm2d` with `GroupNorm` in the SE-equipped encoder gives another clear gain, which reinforces the idea that stable feature statistics matter more than batch-dependent normalization in this OCR setting.
- Evidence: very clear. CER improves materially again with essentially unchanged model size and memory.
- Next action: use this as the new base and test whether all downstream normalization points are still needed, starting with the post-LSTM/head LayerNorm.

### Round 31 - `6f969d8` - remove post-LSTM LayerNorm

- Result: `val_cer=0.812713`, `word_acc=0.030568`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.137372` CER worse
- Keypoint: the post-LSTM/head LayerNorm is not redundant. Even with `GroupNorm` in the encoder and pre-RNN conditioning, the classifier still depends heavily on normalized recurrent outputs.
- Evidence: decisive. Removing it causes a very large collapse in both CER and word accuracy.
- Next action: restore the head LayerNorm and run the symmetric ablation on the pre-RNN LayerNorm to verify whether both normalization boundaries are essential.

### Round 32 - `ef646c4` - remove pre-RNN LayerNorm

- Result: `val_cer=0.704352`, `word_acc=0.098253`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.029011` CER worse
- Keypoint: the pre-RNN LayerNorm also contributes meaningfully, but its role is secondary to the head LayerNorm. Removing it hurts, though not nearly as catastrophically as removing the post-LSTM normalization.
- Evidence: clear. The regression is large enough to keep the module, and the contrast with Round 31 gives a useful ranking of importance between the two normalization points.
- Next action: keep both normalization points in the base model and spend the final rounds testing whether alternative norm forms or modest capacity changes can improve on this conditioned backbone.

### Round 33 - `2a7885b` - RMSNorm head after LSTM

- Result: `val_cer=0.718003`, `word_acc=0.072052`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.042662` CER worse
- Keypoint: at the classifier boundary, pure scale normalization is not enough. The model benefits from full centering-and-scaling of `LayerNorm`, not just RMS-based rescaling.
- Evidence: clear. The regression is substantial, and throughput also falls rather than improving.
- Next action: restore head `LayerNorm` and test the same RMSNorm swap at the pre-RNN boundary, where the normalization role may be less tied to classifier calibration.

### Round 34 - `14a5413` - RMSNorm before LSTM

- Result: `val_cer=0.763652`, `word_acc=0.063319`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.088311` CER worse
- Keypoint: the pre-RNN boundary also wants full `LayerNorm`, not RMS-only rescaling. The recurrent stack appears highly sensitive to centered sequence features, not just normalized magnitudes.
- Evidence: decisive. This is a large regression and clearly worse than the existing conditioned base.
- Next action: stop norm-form exploration and use the last two rounds on modest architecture-capacity tests on top of the now well-validated normalization/attention backbone.

### Round 35 - `de87f00` - wider recurrent head

- Result: `val_cer=0.773464`, `word_acc=0.048035`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.098123` CER worse
- Keypoint: once the encoder and stage-boundary conditioning are fixed, simply increasing recurrent width is strongly counterproductive. The model no longer appears capacity-limited in the recurrent stack; the extra width mostly makes the fixed 5-minute budget less efficient.
- Evidence: very clear. CER degrades sharply with a noticeable parameter increase and no compensating accuracy gain.
- Next action: test the opposite direction next. A slightly narrower recurrent head may better match the now-stronger encoder and recover efficiency without losing too much expressivity.

### Round 36 - `22ad3e9` - narrower recurrent head

- Result: `val_cer=0.794369`, `word_acc=0.050218`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.119028` CER worse
- Keypoint: narrowing the recurrent head is also decisively bad. The current `256` hidden size is not an arbitrary default anymore; with the improved encoder and normalization stack, it is the balance point between enough temporal capacity and enough efficiency.
- Evidence: decisive. Both wider and narrower recurrent heads fail badly, so this axis is now well bracketed.
- Next action: stop spending rounds on recurrent capacity and continue along the stronger discovered theme: better early encoder feature selection and conditioning.

### Round 37 - `29acbe3` - add SE gate to stem

- Result: `val_cer=0.773464`, `word_acc=0.054585`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.098123` CER worse
- Keypoint: the success of early encoder attention does not mean “put SE everywhere as early as possible.” A single hard gate on the stem output is harmful; the useful behavior seems to come from progressive channel reweighting inside residual feature extraction blocks.
- Evidence: clear. CER regresses heavily with only a tiny parameter change.
- Next action: keep the early-encoder focus, but change the form of the intervention. The next experiment should strengthen stem feature extraction itself rather than gating it.

### Round 38 - `088183d` - deeper two-conv stem

- Result: `val_cer=0.755546`, `word_acc=0.078603`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.080205` CER worse
- Keypoint: strengthening the stem is less harmful than gating it, but still clearly inferior to the current base. The current early pipeline is already good enough once the downstream SE, GroupNorm, and dual LayerNorm structure is in place.
- Evidence: clear. This remains well short of the best run despite only a modest parameter increase.
- Next action: keep the stem simple and refocus future experiments on more targeted mid/late encoder conditioning rather than broad early-front-end changes.

### Round 39 - `a43c457` - fuse layer4 features into layer5

- Result: `val_cer=0.703925`, `word_acc=0.087336`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.028584` CER worse
- Keypoint: adding explicit mid-level feature fusion is less destructive than the stem changes, but it still underperforms the simpler top-stage-only representation. The current encoder seems to benefit more from clean conditioned features than from extra multi-scale mixing.
- Evidence: clear enough. The regression is moderate rather than catastrophic, but still large enough to reject.
- Next action: keep the encoder path simple and test learned feature alignment/compression before the recurrent stack, where there may still be wasted dimensionality under the 5-minute budget.
