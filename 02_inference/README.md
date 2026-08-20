# Module 2 — inference

`predict.py` loads the checkpoint Module 1 saved and runs predictions only. The point of this
module is the *code-path split*: training and inference are two genuinely different programs that
happen to share a model definition, not the same loop with a flag toggled.

## What's different from training, mechanically

| | Training (Module 1) | Inference (this module) |
|---|---|---|
| Weights | updated every batch | frozen, loaded from disk |
| Autograd graph | built every forward pass (needed for `.backward()`) | never built — wrapped in `torch.no_grad()` |
| `model.train()` / `model.eval()` | `.train()` — dropout/batchnorm use batch-time behavior | `.eval()` — dropout off, batchnorm uses stored running stats |
| Loss function | required, drives the weight update | not computed at all |
| Optimizer | required | not present |

This model architecture has no dropout or batch norm layers, so `.eval()` is a no-op here in
practice — it's included anyway because forgetting it is a real, easy-to-hit bug the moment a model
*does* have either layer type (predictions silently become nondeterministic or subtly wrong,
without raising an error).

## Run it

```bash
../venv/Scripts/python predict.py
```

Requires `../checkpoints/fashion_cnn.pt` to exist — run Module 1's `train.py` first if it doesn't.
Prints 10 random test images with predicted vs. true label.
