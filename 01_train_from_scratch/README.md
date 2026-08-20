# Module 1 — train from scratch

`train.py` trains a small CNN on FashionMNIST (28x28 grayscale clothing images, 10 classes)
starting from randomly initialized weights. This is the module for seeing the core training loop
with nothing hidden:

```
for each batch:
    zero_grad()               # clear old gradients
    predictions = model(x)    # forward pass
    loss = loss_fn(pred, y)   # how wrong are we, as one number
    loss.backward()           # backprop: compute d(loss)/d(weight) for every weight
    optimizer.step()          # nudge every weight opposite its gradient
```

Repeated over every batch (one full pass = one epoch) for several epochs. `Adam` is used as the
optimizer — it adapts the per-parameter step size using running averages of the gradient and its
square, which converges faster than plain SGD on small models like this without needing careful
learning-rate tuning.

## Why this architecture

Two `Conv2d -> ReLU -> MaxPool2d` blocks reduce the 28x28 image down to a 7x7 feature map with 32
channels, then two fully-connected layers map that down to 10 class scores. Small enough to train
several epochs on CPU in well under a minute per epoch, while still being a real conv net (not a
toy fully-connected-only model) — the point is to see genuine training dynamics, not just a proof
of concept.

## Run it

```bash
../venv/Scripts/python train.py
```

First run downloads FashionMNIST (~30MB) into `../data/` (git-ignored). Expect train loss to drop
from ~0.6 to ~0.2 and test accuracy to land around 90% after 5 epochs — exact numbers vary run to
run since weight init and batch shuffling are random.

The trained weights are saved to `../checkpoints/fashion_cnn.pt` (git-ignored — checkpoints are
regenerable artifacts, not something to commit) for Module 2 to load.
