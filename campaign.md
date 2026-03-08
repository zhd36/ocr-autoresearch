# Mar 8 Campaign

This file tracks the current OCR autoresearch campaign on branch `autoresearch/mar8`.

## Scope

- Minimum total rounds: 300
- Current logged rounds: 90
- Next round index: 91
- Remote branch: `origin/autoresearch/mar8`
- Push cadence: push after every 4 newly completed rounds
- Local runtime command: use `python prepare.py` and `python train.py` in this workspace

## Current Best Result

- Commit: `763771d`
- Description: `higher dropout after big gain`
- `val_cer`: `0.599403`
- `word_acc`: `0.155022`
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

### Round 40 - `9050c23` - pre-RNN projection bottleneck

- Result: `val_cer=0.682167`, `word_acc=0.113537`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.006826` CER worse
- Keypoint: this is the first recent miss that is close enough to be informative rather than simply bad. A learned projection before the recurrent stack seems to improve coarse word-level decisions, but compressing from `512` to `384` likely discards some fine character information, which shows up in CER.
- Evidence: moderately clear. The main metric is still worse, so it is not a keep, but the small gap plus the stronger word accuracy suggests the pre-RNN alignment idea itself may be sound.
- Next action: keep the idea but remove the hard compression. Test a residual pre-RNN adapter that preserves `512` dimensions while still giving the model a learned alignment stage before the LSTM.

### Round 41 - `a3f35e9` - residual pre-RNN adapter

- Result: `val_cer=0.692833`, `word_acc=0.080786`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.017492` CER worse
- Keypoint: the problem in Round 40 was not only the bottleneck width. Adding a learned adapter before the LSTM, even without reducing the final dimensionality, still hurts. That means the current pre-RNN normalization is already providing the useful alignment signal, and extra transform depth there is mostly wasted under the time budget.
- Evidence: clear. The adapter is materially worse than the best run and worse than the simpler bottleneck variant on the main metric.
- Next action: stop spending rounds on pre-RNN adapters and instead test optimization-friendly residual-block initialization, where the current deep conditioned encoder may still gain convergence speed within 5 minutes.

### Round 42 - `bfd58c2` - zero-init residual branch norms

- Result: `val_cer=0.825085`, `word_acc=0.067686`, `memory_gb=3.4`, `status=discard`
- Delta vs best `fe69bc3`: `+0.149744` CER worse
- Keypoint: on the current SE + GroupNorm + dual-LayerNorm backbone, identity-biased residual initialization is strongly harmful. This model is no longer optimization-limited in the way a plain residual stack might be; it needs the residual branches to contribute immediately.
- Evidence: decisive. The collapse is too large to interpret as noise.
- Next action: stop adding optimization crutches inside the architecture. The next targeted move should be to retune optimization for the new stronger backbone, because all earlier LR conclusions were measured on materially weaker models.

### Round 43 - `924a75b` - lower lr on conditioned backbone

- Result: `val_cer=0.663396`, `word_acc=0.124454`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `fe69bc3`: `-0.011945` CER better
- Keypoint: the stronger conditioned backbone changes the optimizer regime. Once SE, GroupNorm, and the two LayerNorm boundaries are in place, the old `6e-4` learning rate becomes too aggressive, and `5e-4` yields a clear gain in both CER and word accuracy.
- Evidence: very clear. This is a meaningful main-metric improvement plus the strongest word accuracy seen so far.
- Next action: stop treating optimization settings as settled. The next round should bracket around `5e-4` on the new backbone rather than jumping back to unrelated structural ideas.

### Round 44 - `ee39e2e` - lower lr bracket

- Result: `val_cer=0.648464`, `word_acc=0.135371`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `924a75b`: `-0.014932` CER better
- Keypoint: the new backbone still prefers even gentler updates. Lowering LR again to `4.5e-4` gives another substantial jump, so the critical insight now is that architectural conditioning shifted the whole optimization sweet spot downward.
- Evidence: extremely clear. CER and word accuracy both improve strongly again.
- Next action: continue a tight optimizer bracket around this region, starting with `4e-4`, before resuming broader structural exploration.

### Round 45 - `fd7ffef` - lower lr further

- Result: `val_cer=0.673208`, `word_acc=0.131004`, `memory_gb=3.4`, `status=discard`
- Delta vs best `ee39e2e`: `+0.024744` CER worse
- Keypoint: the LR sweet spot has a real floor. Dropping to `4e-4` loses too much optimization speed, so the earlier gains were not from “lower is always better,” but from moving into a narrower optimum around `4.5e-4`.
- Evidence: clear. This is meaningfully worse than the current best while keeping strong word accuracy.
- Next action: stop pushing base LR downward and instead test the decay shape around the now-plausible optimum, starting with a lower final LR floor.

### Round 46 - `02522af` - lower final lr floor

- Result: `val_cer=0.677901`, `word_acc=0.133188`, `memory_gb=3.4`, `status=discard`
- Delta vs best `ee39e2e`: `+0.029437` CER worse
- Keypoint: the improved backbone wants a lower base LR, but it does not want overly aggressive late annealing. The model still benefits from meaningful movement late in training.
- Evidence: clear. Main-metric regression is too large to justify keeping the change.
- Next action: test the opposite side of the schedule shape with a slightly higher final LR floor, to see whether the best point is a bit flatter than the current `0.1`.

### Round 47 - `9fb53d1` - higher final lr floor

- Result: `val_cer=0.665956`, `word_acc=0.120087`, `memory_gb=3.4`, `status=discard`
- Delta vs best `ee39e2e`: `+0.017492` CER worse
- Keypoint: the late-training floor is now bracketed. The current backbone wants `FINAL_LR_FRAC=0.1` closely enough that moving either direction hurts.
- Evidence: clear. The regression is smaller than some failures but still too large to keep, and both sides of the floor sweep are now worse than the center.
- Next action: leave the decay shape alone and test whether a small warmup helps the new lower-LR regime enter training more cleanly.

### Round 48 - `4a7ad3c` - small warmup on conditioned backbone

- Result: `val_cer=0.747440`, `word_acc=0.091703`, `memory_gb=3.4`, `status=discard`
- Delta vs best `ee39e2e`: `+0.098976` CER worse
- Keypoint: even on the new, lower-LR regime, warmup is still decisively harmful. This backbone wants to start learning immediately rather than spending the early budget ramping up.
- Evidence: decisive. The regression is far too large to treat as a minor schedule mismatch.
- Next action: keep `WARMUP_RATIO=0` fixed going forward. Future optimizer work should focus on momentum/regularization interactions around the new best LR, not on slower starts.

### Round 49 - `d94fa70` - higher beta2 on conditioned backbone

- Result: `val_cer=0.646758`, `word_acc=0.137555`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `ee39e2e`: `-0.001706` CER better
- Keypoint: the new low-LR regime benefits from a slightly longer second-moment memory. That means the remaining noise is not from under-updating; it is from overly reactive adaptive scaling.
- Evidence: good but smaller than the earlier LR gains. The improvement is real on both CER and word accuracy, though now the increments are tighter.
- Next action: bracket the upper side of `beta2` next, because the signal points toward smoother adaptation, but we still need to know where it starts becoming too inert for a 5-minute budget.

### Round 50 - `fe3e291` - even higher beta2

- Result: `val_cer=0.652730`, `word_acc=0.144105`, `memory_gb=3.4`, `status=discard`
- Delta vs best `d94fa70`: `+0.005972` CER worse
- Keypoint: `beta2=0.999` is past the useful smoothing point. It increases word-level exact matches further, but slightly oversmooths the adaptive scaling and gives back character-level precision.
- Evidence: clear enough to stop this bracket. `0.995` now looks like the center of the useful range.
- Next action: stop micro-tuning `beta2` and test EMA next, because the current pattern suggests evaluation-time parameter smoothing may help without further slowing online adaptation.

### Round 51 - `4b623ba` - EMA on conditioned backbone

- Result: `val_cer=0.656143`, `word_acc=0.109170`, `memory_gb=3.5`, `status=discard`
- Delta vs best `d94fa70`: `+0.009385` CER worse
- Keypoint: EMA is not helping this 5-minute regime. The likely reason is that the best gains are coming from relatively recent updates on a fast-improving trajectory, so averaging parameters backward dilutes useful late-stage specialization.
- Evidence: clear. The regression is meaningful and the memory cost goes up slightly.
- Next action: stop exploring heavy parameter smoothing and test a much lighter regularization change next, such as tiny weight decay, which may control overfitting without blurring the online solution.

### Round 52 - `db9b06d` - tiny weight decay on conditioned backbone

- Result: `val_cer=0.666809`, `word_acc=0.117904`, `memory_gb=3.4`, `status=discard`
- Delta vs best `d94fa70`: `+0.020051` CER worse
- Keypoint: even extremely small weight decay is still a net negative in this regime. The current backbone and low-LR optimizer appear to benefit more from preserving fitting capacity than from additional parameter shrinkage.
- Evidence: clear. The regression is too large to justify keeping the change.
- Next action: keep `weight_decay=0` fixed and move future optimizer experiments toward first-moment behavior or other lightweight stabilization ideas rather than explicit regularization.

### Round 53 - `51c91ba` - higher beta1 on conditioned backbone

- Result: `val_cer=0.669369`, `word_acc=0.124454`, `memory_gb=3.4`, `status=discard`
- Delta vs best `d94fa70`: `+0.022611` CER worse
- Keypoint: once `beta2` is already smoothed to `0.995`, raising `beta1` to `0.95` makes the optimizer too inert for a 5-minute budget. The remaining issue is not lack of momentum; it is balancing responsiveness against noisy scaling.
- Evidence: clear. The regression is meaningful and consistent with over-smoothing updates.
- Next action: keep `beta2=0.995`, but test the opposite side with a lower `beta1` to see whether more responsive first-moment updates pair better with the smoother second-moment estimate.

### Round 54 - `34fa0bd` - lower beta1 on conditioned backbone

- Result: `val_cer=0.641638`, `word_acc=0.126638`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `d94fa70`: `-0.005120` CER better
- Keypoint: the best optimizer shape on the new backbone is now clear: more responsive first-moment updates combined with smoother second-moment scaling. Lowering `beta1` to `0.85` helps the optimizer adapt faster without giving up the stabilizing effect of `beta2=0.995`.
- Evidence: clear. CER improves meaningfully again, even though word accuracy gives back a little from the previous best.
- Next action: continue the same bracket by testing whether `beta1` should go slightly lower still, starting with `0.8`.

### Round 55 - `066f31e` - lower beta1 bracket

- Result: `val_cer=0.651451`, `word_acc=0.152838`, `memory_gb=3.4`, `status=discard`
- Delta vs best `34fa0bd`: `+0.009813` CER worse
- Keypoint: reducing `beta1` further to `0.8` makes updates too reactive. It improves word-level decisiveness, but that extra aggressiveness hurts fine-grained character accuracy.
- Evidence: clear. The direction changes behavior in a consistent way, but not in the direction we want for CER.
- Next action: keep `beta1=0.85` as the center and try controlling update spikes with stronger gradient clipping instead of pushing momentum lower.

### Round 56 - `2bfc6e3` - tighter grad clip on conditioned backbone

- Result: `val_cer=0.673635`, `word_acc=0.128821`, `memory_gb=3.4`, `status=discard`
- Delta vs best `34fa0bd`: `+0.031997` CER worse
- Keypoint: stronger gradient clipping is not the right way to control the remaining noise. It constrains useful updates more than it suppresses harmful spikes.
- Evidence: clear. The regression is too large to justify keeping the change.
- Next action: leave `GRAD_CLIP=5.0` alone. The next most likely useful move is to revisit batch-size noise control on the new backbone, because earlier batch conclusions were collected under much weaker models and optimizer settings.

### Round 57 - `a9ee9eb` - larger batch on conditioned backbone

- Result: `val_cer=0.750853`, `word_acc=0.100437`, `memory_gb=3.4`, `status=discard`
- Delta vs best `34fa0bd`: `+0.109215` CER worse
- Keypoint: in the current regime, more parameter updates matter much more than lower gradient noise from larger batches. Even though sample throughput rises, losing optimizer steps is catastrophic.
- Evidence: decisive. This is a large regression despite higher samples seen.
- Next action: test the opposite side next. With the new optimizer settings, a somewhat smaller batch may finally be worthwhile because it buys more updates without the old instability.

### Round 58 - `252a617` - smaller batch on conditioned backbone

- Result: `val_cer=0.661263`, `word_acc=0.122271`, `memory_gb=3.4`, `status=discard`
- Delta vs best `34fa0bd`: `+0.019625` CER worse
- Keypoint: the batch-size optimum really is local around `32` even in the new regime. Extra optimizer steps from `batch=24` do not overcome the added stochasticity.
- Evidence: clear. This is worse than the current best by a useful margin, and together with Round 57 it brackets the batch optimum well.
- Next action: stop spending rounds on batch size and return to the momentum interaction that is still paying off, especially the possibility that `beta1=0.85` may pair best with a slightly higher `beta2`.

### Round 59 - `76f9bbb` - higher beta2 with lower beta1

- Result: `val_cer=0.634812`, `word_acc=0.144105`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `34fa0bd`: `-0.006826` CER better
- Keypoint: the optimizer still had real headroom. On the strengthened backbone, `beta1=0.85` wants a second-moment estimate that is smoother than `0.995` but not as sluggish as `0.999`. The useful regime is narrower and more delayed than the older optimizer searches suggested.
- Evidence: strong. CER improves meaningfully and word accuracy also rises, so this is not a tradeoff win; it is a cleaner optimizer match.
- Next action: stay in this neighborhood for one more bracket step. Test whether the gain peaks around `beta2=0.997` or still climbs slightly higher, starting with `0.998`.

### Round 60 - `5f1c9ca` - beta2 bracket above new best

- Result: `val_cer=0.622014`, `word_acc=0.159389`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `76f9bbb`: `-0.012798` CER better
- Keypoint: the previous optimizer search was materially under-smoothed for this backbone. The conditioned encoder is producing useful but still bursty gradients, and a much slower second-moment estimate is helping scale those updates cleanly while `beta1=0.85` preserves enough short-horizon responsiveness.
- Evidence: very strong. Both CER and word accuracy improve sharply again, which means this is not a fragile metric trade; it is a broad optimization improvement.
- Next action: keep pushing until the curve clearly turns. The immediate next test should be `beta2=0.999` with the same `beta1=0.85` to see whether the optimum has moved all the way to the upper edge or peaks just below it.

### Round 61 - `bd7ee6c` - upper beta2 edge on new optimizer

- Result: `val_cer=0.635239`, `word_acc=0.137555`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.013225` CER worse
- Keypoint: the second-moment smoothing sweet spot is real and sharp. `beta2=0.999` crosses from useful stabilization into harmful inertia, so the optimizer stops adapting quickly enough late in the 5-minute window.
- Evidence: strong. The regression is large enough that this is not just run noise, especially since word accuracy falls with it.
- Next action: refine inside the narrow high-performing window instead of pushing the edge further. The best next measurement is a midpoint such as `beta2=0.9985` to check whether the optimum sits slightly above `0.998` or whether `0.998` is already the peak.

### Round 62 - `c9e02f4` - midpoint beta2 refinement

- Result: `val_cer=0.647611`, `word_acc=0.161572`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.025597` CER worse
- Keypoint: the optimum is not a broad plateau; it is a narrow alignment. `beta2=0.998` works because it smooths bursty scaling just enough, but even a small extra increase to `0.9985` already makes the optimizer hold stale scale information for too long in this short training budget.
- Evidence: very strong. CER regresses sharply even though word accuracy ticks up, which indicates the model is becoming more decisive but less character-precise.
- Next action: freeze `beta2=0.998` as the new optimizer base. The next high-value search should move laterally rather than further up `beta2`, most likely retuning `LR` around this new optimum or re-evaluating lightweight structural changes under the now-correct optimizer.

### Round 63 - `4f56ca9` - retune lr on beta2 peak

- Result: `val_cer=0.770904`, `word_acc=0.067686`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.148890` CER worse
- Keypoint: once `beta2` is this slow, raising the nominal learning rate is catastrophic. The smoothed scale estimate is helping only when the underlying step size stays restrained; with `5e-4`, the optimizer appears to outrun its own variance calibration and never settles.
- Evidence: decisive. Both CER and word accuracy collapse, and the late-stage loss remains much too high.
- Next action: bracket in the opposite direction. The right question is no longer whether `LR` can go higher, but whether the `beta2=0.998` optimum actually wants a slightly smaller step such as `4.25e-4`.

### Round 64 - `42fbde7` - lower lr under slow beta2

- Result: `val_cer=0.637799`, `word_acc=0.155022`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.015785` CER worse
- Keypoint: `LR=4.5e-4` remains the local optimum even after the major `beta2` shift. Lowering the step size makes training safer than `5e-4`, but it also gives back too much progress inside the fixed 5-minute budget.
- Evidence: strong. The run is much healthier than Round 63, but still clearly behind the best.
- Next action: freeze both `LR=4.5e-4` and `beta2=0.998` as the new optimizer base, then return to structure. The most plausible remaining model-side bottleneck is the handoff from 2D features to the 1D recurrent sequence.

### Round 65 - `139bf83` - add pre-RNN local mixer

- Result: `val_cer=0.709471`, `word_acc=0.113537`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.087457` CER worse
- Keypoint: adding a local temporal convolution before the LSTM is not helping the current encoder. The CNN is already producing sufficiently mixed local width context, and the extra pre-RNN mixer mostly adds optimization burden and steals steps without improving the representation passed to the recurrent head.
- Evidence: strong. CER regresses heavily, memory rises slightly, and the run completes fewer optimization steps because of the added compute.
- Next action: stop pushing the pre-RNN convolution direction. A more promising lightweight structure change is to test explicit sequence position information at the same handoff point, because that changes the inductive bias without adding another heavy transformation block.

### Round 66 - `96a597f` - add pre-RNN positional bias

- Result: `val_cer=0.689420`, `word_acc=0.117904`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.067406` CER worse
- Keypoint: the model does not seem to be missing absolute position information at the CNN-to-sequence handoff. The recurrent stack already captures the needed order bias, and injecting a learned fixed positional signal likely interferes with the translation tolerance that scene-text crops still need.
- Evidence: strong. The regression is large and comes without any compensating benefit in memory or word accuracy.
- Next action: stop spending rounds on small handoff embellishments. The stronger inference now is that the current representation is basically good, and the remaining leverage may come from training-efficiency gains within the 5-minute budget, especially options that increase effective steps without changing the task, such as AMP.

### Round 67 - `73483ab` - enable amp on optimizer peak

- Result: `val_cer=0.793515`, `word_acc=0.054585`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.171501` CER worse
- Keypoint: plain `fp16` AMP is not a free speed win here; it is numerically harmful for this CTC training loop. The run gets almost no throughput benefit, while the optimization quality collapses badly.
- Evidence: decisive. `num_steps` and `samples_seen` are essentially unchanged from the fp32 best, but loss stays very high and both CER and word accuracy collapse.
- Next action: stop treating generic AMP as a likely improvement. If mixed precision is worth revisiting at all, it should be with `bfloat16` rather than `float16`; otherwise the more promising path is to search elsewhere.

### Round 68 - `c2aec4b` - enable bf16 amp

- Result: `val_cer=0.750000`, `word_acc=0.076419`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.127986` CER worse
- Keypoint: mixed precision is broadly the wrong direction for this setup, not just `fp16` specifically. `bfloat16` avoids the worst numerical collapse, but it still does not improve throughput and still degrades optimization badly enough to lose a large amount of CER.
- Evidence: strong. The run remains near baseline step count and sample count, but the loss is far above the fp32 best and final recognition quality is much worse.
- Next action: close the mixed-precision branch and return to simpler underfit-oriented changes. The next likely win is to revisit regularization under the now-correct optimizer, especially whether classifier dropout is still helping at all.

### Round 69 - `e3ebdda` - remove dropout on optimizer peak

- Result: `val_cer=0.692406`, `word_acc=0.120087`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.070392` CER worse
- Keypoint: even in this short-budget regime, the classifier-side dropout is still doing useful stabilization work. Removing it makes the model fit more noisily rather than more effectively, so the current bottleneck is not excessive regularization.
- Evidence: strong. CER regresses heavily and the run actually completes fewer steps, so there is no hidden throughput benefit offsetting the worse metric.
- Next action: stop revisiting dropout. The more plausible remaining lever is now the CTC interface itself, especially whether a small blank-logit bias could improve early alignment dynamics without changing the benchmark or decoder.

### Round 70 - `7792353` - negative blank bias for CTC

- Result: `val_cer=0.632253`, `word_acc=0.146288`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.010239` CER worse
- Keypoint: the CTC-interface hypothesis has real signal. A negative blank bias does help relative to many recent failed directions, which suggests that early blank dominance is part of the optimization story, but `-1.0` is too strong and starts to oversuppress the blank path that CTC still needs.
- Evidence: moderate-to-strong. The run lands close to the top tier instead of collapsing, so unlike AMP or the handoff tweaks, this direction is not obviously wrong; it looks mis-tuned.
- Next action: keep the blank-bias mechanism in the code and bracket a milder value next, such as `-0.5`, to see whether a softer push against blank dominance can capture the alignment benefit without overcorrecting.

### Round 71 - `7938f4e` - milder negative blank bias

- Result: `val_cer=0.655290`, `word_acc=0.148472`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.033276` CER worse
- Keypoint: the blank-bias direction is not peaking between `0` and `-1.0`; the milder bias is worse, which suggests the whole mechanism is at best a secondary effect and not a path to the next meaningful CER drop.
- Evidence: solid. It remains much healthier than the clearly bad directions, but it is still distinctly worse than both the best run and even the stronger `-1.0` variant.
- Next action: stop prioritizing blank-bias bracketing. The next search should return to optimizer-schedule interactions that may have shifted under `beta2=0.998`, or another lightweight head-side modification with a stronger theoretical case.

### Round 72 - `dbe505d` - higher lr floor with slow beta2

- Result: `val_cer=0.650171`, `word_acc=0.137555`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.028157` CER worse
- Keypoint: keeping a higher learning-rate floor does not compensate for the slower `beta2`; it mostly prevents the run from tightening late. The current best setup seems to benefit from very stable scaling early and genuinely small steps at the end.
- Evidence: solid. The run stays healthy but clearly underperforms the best, with no sign of a late-stage payoff from the higher floor.
- Next action: treat `FINAL_LR_FRAC=0.1` as still locked even under `beta2=0.998`. The next experiments should move away from global schedule tweaks and focus on another high-leverage part of the head or loss dynamics.

### Round 73 - `e431149` - preserve width at layer2

- Result: `val_cer=0.715017`, `word_acc=0.043668`, `memory_gb=2.6`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.093003` CER worse
- Keypoint: higher temporal resolution is not free information in a 5-minute budget. Doubling the sequence length from 32 to 64 without changing recurrent width cuts the optimization budget far too hard, and the model never gets close to the baseline fitting regime.
- Evidence: strong. `num_steps` drops from about `4190` to `2999`, and both CER and word accuracy collapse with it.
- Next action: do not reject the resolution hypothesis yet; reject the unbalanced version of it. The next useful test is a compute-rebalanced variant, preserving width at `layer2` while shrinking the recurrent head so the longer sequence can be trained with closer-to-baseline step count.

### Round 74 - `95cfc68` - preserve width with smaller recurrent head

- Result: `val_cer=0.814420`, `word_acc=0.032751`, `memory_gb=2.6`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.192406` CER worse
- Keypoint: the failure mode of the 64-step model is not just raw compute. Even after shrinking the recurrent width, the longer sequence remains much harder to optimize in this budget, and the weaker recurrent head compounds the problem instead of balancing it.
- Evidence: decisive. `num_steps` falls further to `2677`, loss stays extremely high, and recognition quality nearly collapses.
- Next action: stop prioritizing longer encoder sequences for now. The next higher-value structure test is the opposite efficiency move: simplify the recurrent stack itself and see whether the saved budget can be converted into better fitting on the proven 32-step backbone.

### Round 75 - `f934290` - shallower recurrent stack

- Result: `val_cer=0.692833`, `word_acc=0.122271`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.070819` CER worse
- Keypoint: the second recurrent layer is buying more than it costs. A one-layer bidirectional LSTM does train faster and sees more samples, but the lost sequence-modeling depth hurts character accuracy more than the extra optimization steps help.
- Evidence: strong. `num_steps` rises to `4404`, yet CER still regresses heavily, so the gap is not a throughput problem.
- Next action: do not abandon the depth-vs-efficiency trade yet. The next meaningful test is a wider one-layer LSTM, using the saved compute to recover expressive capacity without going back to two recurrent layers.

### Round 76 - `ca7fcdf` - wider one-layer LSTM

- Result: `val_cer=0.643345`, `word_acc=0.122271`, `memory_gb=3.4`, `status=discard`
- Delta vs best `5f1c9ca`: `+0.021331` CER worse
- Keypoint: the one-layer path has real signal once capacity is restored. Most of the Round 75 failure came from insufficient recurrent capacity rather than from shallowness alone, but a single wider layer still does not quite match the accuracy of the current two-layer best.
- Evidence: strong. CER recovers dramatically relative to Round 75 while still keeping high step count (`4331`) and modest parameter count (`16.2M`).
- Next action: keep this line alive for one more bracket step. The best next test is a slightly wider one-layer LSTM, such as `hidden=448` or `512`, to see whether it can fully convert the saved recurrent depth into enough single-layer capacity to challenge the current best.

### Round 77 - `10b4225` - even wider one-layer LSTM

- Result: `val_cer=0.619027`, `word_acc=0.159389`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `5f1c9ca`: `-0.002987` CER better
- Keypoint: the real bottleneck in the recurrent head was not strictly depth; it was the depth-capacity-budget balance. A single bidirectional LSTM layer with much larger hidden size (`512`) keeps throughput essentially intact while giving the model a stronger per-step sequence representation than the previous two-layer `256` stack.
- Evidence: strong. CER reaches a new best, word accuracy matches the previous best, and step count (`4213`) remains effectively unchanged.
- Next action: treat the one-layer-wide regime as a serious contender rather than a side branch. The next experiments should bracket recurrent width around `512` to see whether this improvement peaks here or continues upward.

### Round 78 - `b8968bd` - lower bracket for one-layer width

- Result: `val_cer=0.632253`, `word_acc=0.157205`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.013226` CER worse
- Keypoint: the one-layer solution is genuinely width-hungry. Dropping from `512` to `448` gives back too much sequence capacity, even though throughput stays strong.
- Evidence: solid. Step count remains high (`4263`) and word accuracy stays competitive, but CER clearly regresses.
- Next action: the next informative bracket is upward, not downward. Test a slightly larger one-layer hidden size above `512` to see whether the new best is still on the rising part of the curve or already at the top.

### Round 79 - `f8966d3` - upper bracket for one-layer width

- Result: `val_cer=0.645904`, `word_acc=0.139738`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.026877` CER worse
- Keypoint: the one-layer-wide curve has already turned downward above `512`. Extra capacity beyond this point costs enough training efficiency and/or over-sharpens the head that the net effect is negative.
- Evidence: strong. Parameters rise to `18.5M`, step count falls to `4102`, and CER moves well away from the new best.
- Next action: freeze `1-layer, hidden=512` as the best current recurrent geometry. Since this is a materially different architecture from the old two-layer best, the most justified next move is a very small optimizer re-check around the new structure rather than more width chasing.

### Round 80 - `d44d8c7` - retune beta2 on new one-layer best

- Result: `val_cer=0.632253`, `word_acc=0.150655`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.013226` CER worse
- Keypoint: the new one-layer `512` architecture still prefers the slower `beta2=0.998`. Its gradient statistics changed enough to justify re-checking, but not enough to move the optimum downward.
- Evidence: solid. Step count and memory stay essentially identical to Round 77, so the regression reflects optimization quality rather than throughput drift.
- Next action: lock `beta2=0.998` on the new architecture as well. The next rounds should search for the next structure move on top of `1-layer, hidden=512`, rather than reopening this optimizer axis.

### Round 81 - `7d32529` - add RNN skip fusion

- Result: `val_cer=0.669795`, `word_acc=0.128821`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.050768` CER worse
- Keypoint: the new one-layer head does not want a direct residual shortcut from pre-RNN features into the classifier space. The wide LSTM is already producing a good integrated representation, and bypassing it appears to reintroduce local features in a way that hurts character calibration more than it helps detail retention.
- Evidence: strong. Throughput stays similar, so the drop is about representation quality rather than a compute penalty.
- Next action: abandon post-RNN skip fusion. A more coherent follow-up is to make the one-layer LSTM cheaper at its input instead of adding output-side fusion, because the previous projection experiment had some signal and the new best likely still has room to improve its capacity/budget balance.

### Round 82 - `a26b47a` - project into one-layer LSTM

- Result: `val_cer=0.818259`, `word_acc=0.061135`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.199232` CER worse
- Keypoint: the one-layer `512` head is not bottlenecked by excessive input dimensionality. It needs the full 512-dimensional conditioned encoder sequence; compressing that interface before the LSTM destroys the representation even when step count stays unchanged.
- Evidence: decisive. `num_steps` remains at `4214`, so the collapse is purely representational rather than an optimization-budget issue.
- Next action: stop compressing the encoder-to-LSTM interface. The next stronger hypothesis is to keep the good `512` output width but increase internal recurrent state with LSTM projection, so capacity rises without repeating the failed “just widen the visible output” move.

### Round 83 - `3a5bd54` - projected wide one-layer LSTM

- Result: `val_cer=0.656143`, `word_acc=0.148472`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.037116` CER worse
- Keypoint: increasing hidden-state size behind an LSTM projection does not improve this regime. The projected wide LSTM pays extra recurrent cost without producing a better usable sequence representation than the plain visible-width `512` layer.
- Evidence: strong. Parameters rise to `19.3M`, steps fall to `4016`, and CER regresses materially.
- Next action: stop spending rounds on more elaborate recurrent geometry. The next promising way to use the fixed budget better is to accelerate alignment learning directly, for example with an auxiliary CTC head on the pre-RNN sequence.

### Round 84 - `22e9a8e` - add auxiliary CTC supervision

- Result: `val_cer=0.776451`, `word_acc=0.098253`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.157424` CER worse
- Keypoint: auxiliary supervision at the pre-RNN sequence is actively harmful here. Instead of accelerating useful alignment learning, it splits the optimization target across two representation levels that apparently want different invariances.
- Evidence: decisive. Step count stays healthy, yet CER collapses badly, so this is not a simple budget issue.
- Next action: stop pursuing deep supervision on this backbone. The next worthwhile test is to revisit a wider two-layer LSTM under the now-correct optimizer, because the older negative result on wider recurrent heads was measured in a materially different regime.

### Round 85 - `d720459` - revisit wider two-layer LSTM

- Result: `val_cer=0.686860`, `word_acc=0.131004`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.067833` CER worse
- Keypoint: the new one-layer `512` result was not an optimizer artifact hiding a better deep recurrent model. Even under the correct optimizer, a wider two-layer LSTM spends too much budget for the quality it returns.
- Evidence: strong. Parameters climb to `19.7M`, steps fall below `4000`, and CER regresses heavily.
- Next action: freeze the recurrent side more confidently around `1-layer, hidden=512`. The next likely structural gain is now on the encoder budget side, especially testing whether the stronger head allows a slightly cheaper late encoder without sacrificing feature quality.

### Round 86 - `502dfb3` - lighten final encoder stage

- Result: `val_cer=0.736775`, `word_acc=0.117904`, `memory_gb=3.4`, `status=discard`
- Delta vs best `10b4225`: `+0.117748` CER worse
- Keypoint: the last encoder stage is not redundant even with the stronger one-layer `512` head. The saved compute does buy many more steps, but the missing high-level visual refinement is far more damaging than the extra optimization budget is helpful.
- Evidence: decisive. `num_steps` jumps to `4402`, yet CER collapses badly, which cleanly separates feature-quality loss from optimization-speed gain.
- Next action: stop trimming proven encoder depth. The most justified next move is now a narrow retune on the new best head itself, especially classifier-side dropout, because the visible recurrent width has doubled and may have shifted the best regularization point.

### Round 87 - `c6ab25c` - stronger dropout on one-layer best

- Result: `val_cer=0.618601`, `word_acc=0.146288`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `10b4225`: `-0.000426` CER better
- Keypoint: the new one-layer `512` head does want slightly stronger classifier-side regularization. Once the visible recurrent representation doubled in width, the old `dropout=0.1` setting became a bit too weak, and `0.15` recovers a small but real CER gain.
- Evidence: moderate but credible. Throughput and memory are essentially unchanged, so this improvement is coming from better calibration rather than luck through altered training budget.
- Next action: keep the new base and bracket one step upward with `dropout=0.2` to see whether the regularization optimum moved broadly upward or whether `0.15` is already near the peak.

### Round 88 - `4e81844` - upper dropout bracket on one-layer best

- Result: `val_cer=0.599829`, `word_acc=0.176856`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `c6ab25c`: `-0.018772` CER better
- Keypoint: the dropout shift was not a minor cleanup; it was a major missing calibration. The one-layer `512` head is substantially more expressive than the old baseline, and it needs much stronger classifier-side noise injection to prevent overconfident CTC fitting under the fixed 5-minute budget.
- Evidence: very strong. CER improves sharply and word accuracy jumps with it, while throughput and memory remain effectively unchanged.
- Next action: stay on this line immediately. The next best experiment is another upward bracket, starting with `dropout=0.25`, to determine whether the optimum has moved broadly upward or whether `0.2` is already close to the peak.

### Round 89 - `763771d` - higher dropout after big gain

- Result: `val_cer=0.599403`, `word_acc=0.155022`, `memory_gb=3.4`, `status=keep`
- Delta vs previous best `4e81844`: `-0.000426` CER better
- Keypoint: the strong-dropout regime still has a little headroom above `0.2`, but the gain has become much smaller. This suggests the curve is flattening and we are approaching the new regularization peak rather than still climbing rapidly.
- Evidence: moderate. CER improves again with nearly identical compute, though word accuracy gives back some of the large gain from Round 88.
- Next action: take one more upward step to `dropout=0.3`. That should tell us whether the optimum continues upward or whether the curve has already started to turn.

### Round 90 - `6b91f9a` - upper edge of new dropout regime

- Result: `val_cer=0.627133`, `word_acc=0.150655`, `memory_gb=3.4`, `status=discard`
- Delta vs best `763771d`: `+0.027730` CER worse
- Keypoint: the new dropout optimum has already been crossed by `0.3`. The strong regularization regime is real, but there is still a narrow sweet spot; going too high quickly starts to suppress useful fitting rather than just calibrating confidence.
- Evidence: strong. Compute is unchanged, yet CER falls back substantially.
- Next action: lock the optimum to the low-to-mid `0.2x` range instead of exploring higher values. The best next move is a midpoint refinement such as `0.225` or `0.24`, or a return to structure on top of the now much stronger regularized one-layer head.

### Round 91 - `abf2518` - midpoint refinement in new dropout peak

- Result: `val_cer=0.633959`, `word_acc=0.141921`, `memory_gb=3.4`, `status=discard`
- Delta vs best `763771d`: `+0.034556` CER worse
- Keypoint: the high-performing dropout regime is not behaving like a smooth convex scalar sweep. A tiny move from `0.25` down to `0.24` causes a large regression, which means either the optimum is unusually sharp or this region has enough run-to-run instability that a direct reproducibility check is now more valuable than further interpolation guesses.
- Evidence: strong. Compute is unchanged, yet the metric drops far more than expected for such a small scalar change.
- Next action: rerun the current best `dropout=0.25` exactly once to verify whether that result is stable. Without that check, nearby scalar conclusions are too brittle to trust.

### Round 92 - `f36e094` - reproducibility check for new best

- Result: `val_cer=0.640785`, `word_acc=0.124454`, `memory_gb=3.4`, `status=discard`
- Delta vs best `763771d`: `+0.041382` CER worse
- Keypoint: the new `dropout=0.25` result is not stable enough yet to trust at face value. Re-running the exact same setup lands far away from the previous best, which means the recent high-dropout gains are sitting in a high-variance regime rather than a cleanly repeatable optimum.
- Evidence: very strong. Compute is essentially identical, so this is not a throughput artifact; it is optimization instability.
- Next action: stop fine interpolation around `0.24-0.25` until the variance is understood. The next most useful test is to re-run the stronger `dropout=0.2` setting once, to determine whether the whole high-dropout regime is unstable or whether `0.25` specifically is the brittle point.

### Round 93 - `cb1e715` - reproducibility check for dropout 0.2

- Result: `val_cer=0.618174`, `word_acc=0.155022`, `memory_gb=3.4`, `status=discard`
- Delta vs best `763771d`: `+0.018771` CER worse
- Keypoint: `dropout=0.2` is materially more stable than `0.25`, but it still does not consistently reproduce the very best lucky run. This points to a real high-dropout benefit combined with a still-noisy optimization regime, rather than pure measurement noise or pure scalar overfitting.
- Evidence: solid. The rerun stays near the stronger region instead of collapsing, but it remains clearly behind the single best result.
- Next action: test a stabilization move rather than another plain rerun. The most justified next experiment is `dropout=0.25` with a slightly lower learning rate, to see whether the strong-but-brittle high-dropout regime can be made reproducible.

### Round 94 - `60c00b0` - stabilize high dropout with lower lr

- Result: `val_cer=0.668515`, `word_acc=0.115721`, `memory_gb=3.4`, `status=discard`
- Delta vs best `763771d`: `+0.069112` CER worse
- Keypoint: the high-dropout instability is not caused by a nominal learning rate that is slightly too large. Lowering LR simply undercuts fitting while preserving the same unstable character of the regime.
- Evidence: strong. Compute stays similar, but recognition quality drops sharply.
- Next action: move away from step-size stabilization and test moment-based stabilization instead. The most plausible next experiment is a slightly slower second-moment estimate on the high-dropout configuration.
