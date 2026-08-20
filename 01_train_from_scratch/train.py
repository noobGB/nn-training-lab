"""Train a small CNN on FashionMNIST from random weight initialization.

This script exists to make the training loop *visible* rather than hidden behind a
high-level `.fit()` call: every step from forward pass to gradient update is written out
explicitly below, with output showing the loss actually going down epoch over epoch.

Run:
    ../venv/Scripts/python train.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = "../data"
CHECKPOINT_PATH = "../checkpoints/fashion_cnn.pt"
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3


class SmallCNN(nn.Module):
    """Two conv blocks + two fully-connected layers, sized for 28x28 grayscale input."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Download (if needed) FashionMNIST and wrap it in train/test DataLoaders."""
    transform = transforms.ToTensor()  # scales pixel values to [0, 1]
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one full pass over the training set, updating weights after every batch."""
    model.train()  # enables dropout/batchnorm training behavior (this model has neither, but it's the correct habit)
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # clear gradients from the previous step
        predictions = model(images)  # forward pass
        loss = loss_fn(predictions, labels)  # how wrong are we
        loss.backward()  # backpropagation: compute d(loss)/d(weight) for every weight
        optimizer.step()  # nudge weights opposite their gradient

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Compute average loss and accuracy on a held-out set, with no gradient tracking."""
    model.eval()  # switches off training-only behavior (dropout/batchnorm)
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        total_loss += loss_fn(predictions, labels).item() * images.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main() -> None:
    device = torch.device("cpu")
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(
            f"epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} "
            f"| test_loss={test_loss:.4f} | test_acc={test_acc:.2%}"
        )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
