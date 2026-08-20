"""Fine-tune a pretrained ResNet18 on a small custom image dataset.

This is the module closest to how "retraining" actually works in practice — almost nobody trains a
vision model from random weights anymore. Instead:

1. Start from weights already trained on a large dataset (ImageNet, 1.2M images) instead of random
   init. Early conv layers have already learned generic, transferable features (edges, textures,
   color blobs) that are useful for almost any image task.
2. Replace the final classification layer, since the pretrained model's 1000 ImageNet classes
   don't match our 3 custom classes.
3. Freeze the early layers and only train the later layers + new head, so the training loop can't
   overwrite those generic features.
4. Use a much smaller learning rate than training-from-scratch (Module 1 used 1e-3; here we use
   1e-4). The pretrained weights are already close to a good solution — large updates risk
   "catastrophic forgetting," where the model overwrites its useful pretrained features faster than
   it learns anything new from the small dataset, ending up worse than either the original
   pretrained model or a from-scratch model.

Requires custom_data/ to exist — run prepare_dataset.py first.

Run:
    ../venv/Scripts/python finetune.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

DATA_DIR = "custom_data"
CHECKPOINT_PATH = "../checkpoints/resnet18_finetuned.pt"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4  # smaller than Module 1's 1e-3: we're nudging pretrained weights, not learning from scratch


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load the custom image-folder dataset, resized/normalized to match ImageNet pretraining."""
    # ResNet18 was pretrained on 224x224 ImageNet images normalized with these specific
    # per-channel statistics; fine-tuning must feed it inputs in the same distribution.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_set.classes


def build_model(num_classes: int) -> nn.Module:
    """Load pretrained ResNet18, freeze early layers, and swap in a new head."""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False  # freeze everything first...

    for param in model.layer4.parameters():
        param.requires_grad = True  # ...then unfreeze the last conv block (higher-level features)

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head is trainable by default
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
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
    train_loader, test_loader, classes = get_dataloaders(BATCH_SIZE)
    print(f"classes: {classes}")

    model = build_model(num_classes=len(classes)).to(device)

    # Only the unfrozen parameters (layer4 + fc) get passed to the optimizer — passing frozen
    # params would waste memory on optimizer state (e.g. Adam's per-parameter moment estimates)
    # for weights that never receive a gradient update anyway.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Baseline: how well does the untouched pretrained model do before any fine-tuning?
    baseline_loss, baseline_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"baseline (pretrained, unmodified head) | test_acc={baseline_acc:.2%}")

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
