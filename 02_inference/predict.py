"""Run inference-only predictions from a checkpoint trained by Module 1.

This script deliberately has no loss function, no backward pass, and no optimizer — inference is
just the forward pass, with two things turned off that only matter during training:

- `torch.no_grad()`: skips building the autograd graph. During training, every operation on a
  tensor gets recorded so `.backward()` can later compute gradients through it; at inference time
  we never call `.backward()`, so building that graph is wasted memory and compute.
- `model.eval()`: switches layers that behave differently in train vs. eval mode. This model has no
  dropout or batch norm, so it's a no-op here — but it's the correct habit for any model that does
  (dropout stops randomly zeroing activations; batch norm switches from batch statistics to its
  stored running mean/variance).

Run:
    ../venv/Scripts/python predict.py
"""

import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, "../01_train_from_scratch")
from train import SmallCNN  # reuse the exact architecture the checkpoint was trained with

DATA_DIR = "../data"
CHECKPOINT_PATH = "../checkpoints/fashion_cnn.pt"
NUM_SAMPLES = 10

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_model(checkpoint_path: str, device: torch.device) -> SmallCNN:
    """Rebuild the architecture and load trained weights into it."""
    model = SmallCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # inference mode: no dropout/batchnorm training behavior
    return model


@torch.no_grad()  # no gradient graph needed since we never call .backward() here
def predict(model: SmallCNN, images: torch.Tensor) -> torch.Tensor:
    """Forward pass only, returning the predicted class index per image."""
    logits = model(images)
    return logits.argmax(dim=1)


def main() -> None:
    device = torch.device("cpu")
    model = load_model(CHECKPOINT_PATH, device)

    test_set = datasets.FashionMNIST(
        DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(test_set, batch_size=NUM_SAMPLES, shuffle=True)
    images, labels = next(iter(loader))

    predictions = predict(model, images.to(device))

    correct = 0
    for i in range(NUM_SAMPLES):
        predicted_label = CLASS_NAMES[predictions[i].item()]
        true_label = CLASS_NAMES[labels[i].item()]
        match = "correct" if predictions[i] == labels[i] else "WRONG"
        correct += predictions[i] == labels[i]
        print(f"predicted={predicted_label:<12} true={true_label:<12} [{match}]")

    print(f"\n{correct}/{NUM_SAMPLES} correct")


if __name__ == "__main__":
    main()
