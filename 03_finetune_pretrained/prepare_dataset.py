"""Build a small custom image-folder dataset from a FashionMNIST subset.

Real fine-tuning starts from your own folder of images, organized one subfolder per class
(exactly what `torchvision.datasets.ImageFolder` expects) — not from a built-in torchvision
dataset object. To keep this reproducible without requiring your own photo collection, this script
carves a small "custom" dataset (3 classes, ~100 train / 20 test images each) out of FashionMNIST
— already downloaded locally by Module 1, so this needs no new download — and writes it to disk in
that exact folder layout, so `finetune.py` loads it exactly the way it would load any real custom
dataset. Grayscale images are saved as 3-channel RGB PNGs since the pretrained ResNet18 in
Module 3 expects 3-channel input.

Run once before finetune.py:
    ../venv/Scripts/python prepare_dataset.py
"""

import os

from torchvision import datasets
from torchvision.transforms.functional import to_pil_image

DATA_DIR = "../data"
OUTPUT_DIR = "custom_data"
CLASSES = ["Sneaker", "Sandal", "Ankle boot"]  # visually similar footwear classes -> a genuinely non-trivial 3-way split
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 20

ALL_FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def export_split(dataset, split_name: str, images_per_class: int) -> None:
    """Walk a FashionMNIST dataset and save the first N images per target class as RGB PNGs."""
    counts = {c: 0 for c in CLASSES}
    class_to_idx = {name: ALL_FASHION_CLASSES.index(name) for name in CLASSES}
    idx_to_name = {idx: name for name, idx in class_to_idx.items()}

    for image, label in dataset:
        if label not in idx_to_name:
            continue
        class_name = idx_to_name[label]
        if counts[class_name] >= images_per_class:
            continue

        class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)
        to_pil_image(image).convert("RGB").save(
            os.path.join(class_dir, f"{counts[class_name]:03d}.png")
        )
        counts[class_name] += 1

        if all(c >= images_per_class for c in counts.values()):
            break

    print(f"{split_name}: {counts}")


def main() -> None:
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    train_source = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=to_tensor)
    test_source = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=to_tensor)

    export_split(train_source, "train", TRAIN_PER_CLASS)
    export_split(test_source, "test", TEST_PER_CLASS)
    print(f"wrote custom dataset to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
