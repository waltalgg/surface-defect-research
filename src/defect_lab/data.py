from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import Config


class ManifestDataset(Dataset):
    def __init__(self, items: list[dict[str, str]], class_to_index: dict[str, int], image_size: int, augment: bool) -> None:
        if augment:
            transform_steps = [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        else:
            transform_steps = [
                transforms.Resize(image_size + 16),
                transforms.CenterCrop(image_size),
            ]
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(transform_steps)
        self.items = items
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        sample = self.items[index]
        image = Image.open(sample["path"]).convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_index[sample["label"]]
        return tensor, label


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def create_dataloaders(config: Config) -> tuple[dict[str, DataLoader], list[str]]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    manifest = load_manifest(dataset_cfg["manifest_path"])
    classes = manifest["classes"]
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    image_size = int(dataset_cfg["image_size"])

    datasets = {
        split_name: ManifestDataset(
            items=items,
            class_to_index=class_to_index,
            image_size=image_size,
            augment=(split_name == "train"),
        )
        for split_name, items in manifest["splits"].items()
    }

    loaders = {}
    balanced_sampling = bool(training_cfg.get("balanced_sampling", False))
    for split_name, dataset in datasets.items():
        shuffle = split_name == "train"
        sampler = None
        if split_name == "train" and balanced_sampling:
            label_counts = Counter(item["label"] for item in manifest["splits"]["train"])
            sample_weights = [1.0 / label_counts[item["label"]] for item in manifest["splits"]["train"]]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(training_cfg["num_workers"]),
            pin_memory=bool(training_cfg.get("pin_memory", False)),
        )
    return loaders, classes
