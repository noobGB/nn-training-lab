# nn-training-lab

A hands-on, local-first lab for learning how neural networks are actually trained, how inference
works, and how to retrain (fine-tune) an existing model — starting small, on CPU, with real code
instead of theory alone.

## Why this repo exists

Understanding "training" and "inference" conceptually is easy; understanding them by watching a
loss curve move, then loading your own checkpoint, then fine-tuning a pretrained model on your own
tiny dataset, builds the kind of intuition that sticks. Each module below is a self-contained script
with its own README section explaining what it demonstrates and why it's built that way.

## Environment

- Python 3.12, CPU-only (no NVIDIA GPU on this machine — everything here is sized to train in
  minutes on CPU).
- PyTorch + torchvision, installed via the CPU wheel index (`--index-url
  https://download.pytorch.org/whl/cpu`) to avoid pulling unnecessary CUDA dependencies.

Setup:

```bash
python -m venv venv
./venv/Scripts/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

(`requirements.txt` pins the exact CPU-build versions used to develop this repo; the `+cpu` local
version suffix on torch/torchvision only resolves against PyTorch's own CPU wheel index, hence the
explicit `--index-url`.)

## Modules

| # | Module | Concept | Status |
|---|--------|---------|--------|
| 1 | [`01_train_from_scratch/`](01_train_from_scratch/) | Full training loop (forward → loss → backward → optimizer step) on a small CNN, from random init | done |
| 2 | [`02_inference/`](02_inference/) | Loading a trained checkpoint and running inference-only (no grad, eval mode) | planned |
| 3 | [`03_finetune_pretrained/`](03_finetune_pretrained/) | Fine-tuning a pretrained ResNet18 on a small custom dataset — the real-world "retraining" pattern | planned |

Each module folder gets its own short README once built, explaining the specific mechanics it's
demonstrating (not just how to run it).

## Conventions

This repo follows the standard solo-repo GitHub workflow: one GitHub issue per module, one branch
per issue, PR per branch, squash-merged after CI is green. See the repo's issues/PRs for the history
of how each module was built.
