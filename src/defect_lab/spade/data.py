from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from ..config import Config
from ..utils import ensure_dir, set_seed, write_json


def build_spade_manifest(config: Config) -> dict:
    dataset_cfg = config["dataset"]
    image_root = Path(dataset_cfg["image_dir"])
    mask_root = Path(dataset_cfg["mask_dir"])
    pairs = []
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append({"image_path": str(image_path.as_posix()), "mask_path": str(mask_path.as_posix())})
    if not pairs:
        raise FileNotFoundError(f"No SPADE pairs found in {image_root} and {mask_root}")

    set_seed(int(config["experiment"]["seed"]))
    random.shuffle(pairs)

    train_split = float(dataset_cfg.get("train_split", 0.7))
    val_split = float(dataset_cfg.get("val_split", 0.15))
    total = len(pairs)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    manifest = {
        "splits": {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        },
        "metadata": {
            "task": "spade",
            "image_dir": str(image_root.as_posix()),
            "mask_dir": str(mask_root.as_posix()),
            "total_pairs": total,
        },
    }
    manifest_path = Path(dataset_cfg["manifest_path"])
    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)
    return manifest


class SpadeDataset(Dataset):
    def __init__(self, items: list[dict[str, str]], image_size: int, augment: bool) -> None:
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        image = Image.open(item["image_path"]).convert("RGB")
        mask = Image.open(item["mask_path"]).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=Image.Resampling.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.Resampling.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-8, 8)
            fill = tuple(int(v) for v in image.resize((1, 1)).getpixel((0, 0)))
            image = TF.rotate(image, angle, interpolation=Image.Resampling.BILINEAR, fill=fill)
            mask = TF.rotate(mask, angle, interpolation=Image.Resampling.NEAREST, fill=0)

        image_tensor = TF.to_tensor(image) * 2.0 - 1.0
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return mask_tensor, image_tensor


def create_spade_dataloaders(config: Config) -> dict[str, DataLoader]:
    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        build_spade_manifest(config)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    image_size = int(config["dataset"]["image_size"])
    training_cfg = config["training"]
    datasets = {
        split: SpadeDataset(items, image_size=image_size, augment=(split == "train"))
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
