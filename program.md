# ocr-autoresearch

This is an experiment to have the LLM do its own OCR research.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag: propose a tag based on today's date, for example `mar8`.
2. Create a fresh branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. Verify the benchmark cache exists under `~/.cache/ocr-autoresearch/`. If not, run `uv run prepare.py`.
5. Initialize `results.tsv` if it does not exist with the header:

```text
commit	val_cer	word_acc	memory_gb	status	description
```

6. Confirm the setup and start the first baseline run.

## Experimentation

Each experiment runs on a single GPU and gets a fixed 5-minute training budget:

```bash
uv run train.py
```

What you CAN do:

- Modify `train.py`
- Change model architecture, optimizer, batch size, schedule, normalization, regularization, and loss details inside that file

What you CANNOT do:

- Modify `prepare.py`
- Change the benchmark data
- Change the evaluation metric
- Add new dependencies

The goal is to minimize `val_cer`. Lower is better.

Metric note:

- `val_cer` and `val_word_acc` are computed after lowercasing, removing spaces, and filtering non-alphanumeric symbols, matching OpenOCR-style recognition evaluation

Secondary metrics:

- `val_word_acc`: higher is better
- `peak_vram_mb`: keep it reasonable

Simplicity criterion:

- Prefer simpler changes when gains are close
- A tiny gain is not worth a large pile of ugly code
- Deleting code and improving metrics is a very strong result

## Output Format

`train.py` prints a summary like:

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

## Logging Results

Append every experiment to `results.tsv` using tab-separated columns:

```text
commit	val_cer	word_acc	memory_gb	status	description
```

Rules:

- `commit`: short git hash
- `val_cer`: use `9.999999` for crashes
- `word_acc`: use `0.000000` for crashes
- `memory_gb`: `peak_vram_mb / 1024`, rounded to one decimal
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the attempted idea

Example:

```text
commit	val_cer	word_acc	memory_gb	status	description
a1b2c3d	0.284100	0.391200	1.8	keep	baseline
b2c3d4e	0.271900	0.407500	1.9	keep	increase batch size
c3d4e5f	0.286800	0.384100	1.8	discard	add dropout before classifier
d4e5f6g	9.999999	0.000000	0.0	crash	double hidden size OOM
```

## Experiment Loop

Loop forever:

1. Look at the current branch and commit.
2. Edit `train.py` with one experimental idea.
3. Commit the change.
4. Run the experiment with output redirected to `run.log`.
5. Extract `val_cer`, `val_word_acc`, and `peak_vram_mb`.
6. If the run crashed, inspect the traceback and decide whether to fix or discard.
7. Append the result to `results.tsv`.
8. If `val_cer` improved, keep the commit and continue from it.
9. If `val_cer` got worse or stayed the same, revert to the previous good commit.

Recommended command:

```bash
uv run train.py > run.log 2>&1
```

If the summary is missing, inspect the end of the log:

```bash
tail -n 50 run.log
```

Timeout rule:

- Normal runs should finish in about 5 minutes plus eval overhead
- If a run exceeds 10 minutes total, kill it and treat it as a failure

Never stop after the loop starts unless the human interrupts you.
