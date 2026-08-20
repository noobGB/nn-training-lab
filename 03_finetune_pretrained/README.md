# Module 3 — fine-tune a pretrained model

This is the module closest to how "retraining" works in real practice. Almost no one trains a
vision model from random weights the way Module 1 did — instead you start from a model already
trained on a large dataset and adapt it to your own smaller one.

## Dataset

`prepare_dataset.py` carves a small "custom" dataset (3 classes — Sneaker, Sandal, Ankle boot —
~100 train / 20 test images each) out of FashionMNIST and writes it to
`custom_data/<split>/<class>/*.png` as RGB PNGs. This is exactly the folder layout you'd use for
your own photos: one subfolder per class, loaded via `torchvision.datasets.ImageFolder`. Using a
FashionMNIST subset (already downloaded locally by Module 1, so no new download needed) instead of
requiring your own photo collection keeps this reproducible, but the loading code is identical to
what you'd use with real custom images — swap `custom_data/` for your own folder and `finetune.py`
doesn't change. The three footwear classes were picked deliberately: they're visually more similar
to each other than, say, a shirt vs. a boot, making the 3-way split genuinely non-trivial rather
than something a barely-adapted head could ace immediately.

Run once first:
```bash
../venv/Scripts/python prepare_dataset.py
```

## What's different from Module 1

| | Module 1 (from scratch) | Module 3 (fine-tune) |
|---|---|---|
| Weight init | random | pretrained on ImageNet (1.2M images) |
| Layers trained | all | only `layer4` + new `fc` head (rest frozen) |
| Learning rate | 1e-3 | 1e-4 |
| Why | nothing to preserve, learn everything | early layers already encode generic edge/texture features worth keeping |

**Freezing**: `requires_grad = False` on most of the network means those weights never receive a
gradient update — the forward pass still runs through them (they still extract features), but
`.backward()` doesn't touch them and the optimizer only tracks the unfrozen parameters. This is
the mechanism, not just a policy: gradients literally don't flow into frozen weights.

**Why the smaller learning rate**: the pretrained weights already encode a good solution. A large
learning rate applied to a small new dataset can overwrite that solution faster than it learns
anything genuinely new — this is **catastrophic forgetting**, where fine-tuning makes the model
*worse* than either the original pretrained model or a model trained from scratch, because it
half-destroys useful pretrained features without fully replacing them with better ones. Freezing
most layers and using a small learning rate on the rest is the standard mitigation.

## Run it

```bash
../venv/Scripts/python finetune.py
```

Prints a baseline accuracy first (pretrained ResNet18 with an untrained random head — this landed
at 46.67% on a local run, above the 33% chance-level for 3 classes because even a *random* linear
projection of the pretrained backbone's genuinely useful features carries some separability), then
epoch-by-epoch accuracy as the head and last block adapt. A local run went 46.67% → 83.33% → 85.00%
→ 86.67% → 88.33% → 86.67% over 5 epochs.

Note the last epoch: train_loss kept falling (0.028 → 0.017) while test accuracy *dipped slightly*
(88.33% → 86.67%). That's overfitting, visible in real time — with only 300 training images, the
unfrozen layers start memorizing training-set specifics late in training rather than learning
anything more general. This is a real, common failure mode of fine-tuning on small datasets, not a
bug in the script; the fix in practice is early stopping (keep the best-val checkpoint rather than
the last one) or more aggressive freezing/regularization.
