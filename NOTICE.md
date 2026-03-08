This repository is inspired by two upstream projects:

1. `karpathy/autoresearch` (MIT)
   - Repository structure
   - Fixed-budget autonomous experiment loop
   - `program.md`-driven workflow

2. `Topdu/OpenOCR` (Apache-2.0)
   - OCR task framing
   - CRNN + CTC baseline selection
   - ResNet_ASTER-style encoder design adapted in `train.py`

This repository is self-contained and does not import code from OpenOCR at runtime.

