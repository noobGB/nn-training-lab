# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A hands-on, local-first, CPU-only lab for learning neural network training, inference, and
fine-tuning mechanics through real runnable code rather than theory alone. See [README.md](README.md)
for the full learning narrative and per-module explanations — this file is for working in the repo,
not learning from it.

## Commands

Setup (once):
```bash
python -m venv venv
./venv/Scripts/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

Run a module (each is a standalone script, run from inside its own folder since checkpoint/data
paths are relative):
```bash
cd 01_train_from_scratch && ../venv/Scripts/python train.py
cd 02_inference && ../venv/Scripts/python predict.py       # requires 01's checkpoint to exist first
cd 03_finetune_pretrained && ../venv/Scripts/python prepare_dataset.py   # once, before finetune.py
cd 03_finetune_pretrained && ../venv/Scripts/python finetune.py
```

There is no test suite, linter, or CI configured — this is a learning repo, not a shipped package.
Verification is "run the script and check the printed loss/accuracy numbers look sane," done and
recorded in each module's PR description rather than automated.

## Architecture

Three independent, numbered modules, each a self-contained pair of a script + README:

- **`01_train_from_scratch/`** — the full training loop written out explicitly (zero_grad → forward
  → loss → backward → optimizer step), training a small CNN (`SmallCNN`) on FashionMNIST from
  random init. Saves `../checkpoints/fashion_cnn.pt`.
- **`02_inference/`** — loads Module 1's checkpoint and runs forward-pass-only predictions.
  Imports `SmallCNN` directly from `01_train_from_scratch/train.py` (via `sys.path.insert`) rather
  than redefining the architecture, so the loaded `state_dict` always matches — if Module 1's
  architecture changes, Module 2 picks it up automatically rather than silently drifting out of
  sync.
- **`03_finetune_pretrained/`** — the real-world "retraining" pattern: pretrained ResNet18, frozen
  early layers, new head, smaller learning rate. `prepare_dataset.py` must run first — it builds
  `custom_data/<split>/<class>/*.png` from a **local FashionMNIST subset** (not a fresh download;
  CIFAR-10 was tried first but its host proved too slow on this network — see git history on
  issue #3 / PR #6 for the full reasoning). This mimics the real workflow of fine-tuning against
  your own folder of class-labeled images via `torchvision.datasets.ImageFolder`.

**Shared conventions across modules**: `DATA_DIR`/`CHECKPOINT_PATH`/hyperparameter constants live
as module-level uppercase constants at the top of each script (not a shared config file — each
module is meant to be read start-to-end as a standalone teaching artifact, so duplication here is
intentional, not an oversight). `data/`, `custom_data/`, and `checkpoints/` are all git-ignored —
regenerable, not source.

## Extending this repo

Adding a new module: follow the existing issue → branch → PR pattern (file a GitHub issue first,
branch as `feat/issue-<N>-<slug>`, squash-merge after locally running the script and pasting real
output numbers into the PR description — see PRs #4–#6 for the established shape). Give it a
numbered folder (`04_.../`) with its own `README.md` explaining the *mechanism* it demonstrates,
not just how to run it, and add a row to the module table in the root `README.md`.
