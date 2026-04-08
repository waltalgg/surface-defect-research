from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from ..config import Config
from ..utils import ensure_dir, set_seed, write_json


@dataclass(slots=True)
class SegmentationPair:
    image_path: str
    mask_path: str

    def to_dict(self) -> dict[str, str]:
        return {"image_path": self.image_path, "mask_path": self.mask_path}


def _list_segmentation_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[SegmentationPair]:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    pairs: list[SegmentationPair] = []
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append(SegmentationPair(str(image_path), str(mask_path)))
    if not pairs:
        raise FileNotFoundError(f"No segmentation pairs found in {image_root} and {mask_root}")
    return pairs


def build_segmentation_manifest(config: Config) -> dict:
    dataset_cfg = config["dataset"]
    pairs = _list_segmentation_pairs(dataset_cfg["image_dir"], dataset_cfg["mask_dir"])
    set_seed(int(config["experiment"]["seed"]))
    random.shuffle(pairs)

    train_split = float(dataset_cfg.get("train_split", 0.7))
    val_split = float(dataset_cfg.get("val_split", 0.15))
    total = len(pairs)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    manifest = {
        "splits": {
            "train": [pair.to_dict() for pair in train_pairs],
            "val": [pair.to_dict() for pair in val_pairs],
            "test": [pair.to_dict() for pair in test_pairs],
        },
        "metadata": {
            "task": "segmentation",
            "image_dir": str(Path(dataset_cfg["image_dir"]).as_posix()),
            "mask_dir": str(Path(dataset_cfg["mask_dir"]).as_posix()),
            "total_pairs": total,
        },
    }

    manifest_path = Path(dataset_cfg["manifest_path"])
    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)
    return manifest


class SegmentationDataset(Dataset):
    def __init__(self, items: list[dict[str, str]], image_size: int, augment: bool) -> None:
        self.items = items
        self.image_size = image_size
        self.augment = augment
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.items[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=Image.Resampling.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.Resampling.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-12, 12)
            fill = tuple(int(v) for v in image.resize((1, 1)).getpixel((0, 0)))
            image = TF.rotate(image, angle, interpolation=Image.Resampling.BILINEAR, fill=fill)
            mask = TF.rotate(mask, angle, interpolation=Image.Resampling.NEAREST, fill=0)
            image = transforms.ColorJitter(brightness=0.12, contrast=0.15)(image)

        image_tensor = TF.to_tensor(image)
        image_tensor = self.image_normalize(image_tensor)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor


def create_segmentation_dataloaders(config: Config) -> dict[str, DataLoader]:
    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        build_segmentation_manifest(config)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    image_size = int(config["dataset"]["image_size"])
    training_cfg = config["training"]
    datasets = {
        split: SegmentationDataset(items, image_size=image_size, augment=(split == "train"))
        for split, items in manifest["splits"].items()
    }

    return {
        split: DataLoader(
            dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(training_cfg.get("num_workers", 0)),
            pin_memory=bool(training_cfg.get("pin_memory", False)),
        )
        for split, dataset in datasets.items()
    }


def estimate_pos_weight(config: Config) -> float:
    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        build_segmentation_manifest(config)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    positive_pixels = 0.0
    total_pixels = 0.0
    for item in manifest["splits"]["train"]:
        mask = Image.open(item["mask_path"]).convert("L")
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        positive_pixels += float(mask_tensor.sum().item())
        total_pixels += float(mask_tensor.numel())

    negative_pixels = max(1.0, total_pixels - positive_pixels)
    positive_pixels = max(1.0, positive_pixels)
    return negative_pixels / positive_pixels
