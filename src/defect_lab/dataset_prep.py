from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .config import Config
from .utils import ensure_dir, set_seed, write_json


def _list_images(root: Path, allowed_extensions: set[str]) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in allowed_extensions:
                by_class[class_dir.name].append(image_path)
    return dict(by_class)


def _split_items(items: list[str], train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def prepare_dataset(config: Config) -> None:
    dataset_cfg = config["dataset"]
    experiment_cfg = config["experiment"]

    raw_dir = Path(dataset_cfg["raw_dir"])
    manifest_path = Path(dataset_cfg["manifest_path"])
    allowed_extensions = {ext.lower() for ext in dataset_cfg["allowed_extensions"]}
    max_images_per_class = dataset_cfg.get("max_images_per_class")
    train_images_per_class = dataset_cfg.get("train_images_per_class")
    include_classes = set(dataset_cfg.get("include_classes") or [])
    label_map = dataset_cfg.get("label_map") or {}

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    set_seed(int(experiment_cfg["seed"]))
    class_to_paths = _list_images(raw_dir, allowed_extensions)
    if include_classes:
        class_to_paths = {class_name: paths for class_name, paths in class_to_paths.items() if class_name in include_classes}
    if not class_to_paths:
        raise RuntimeError("No class folders with valid images were found in data/raw.")

    output_classes = sorted({label_map.get(class_name, class_name) for class_name in class_to_paths})
    manifest: dict[str, object] = {
        "classes": output_classes,
        "splits": {"train": [], "val": [], "test": []},
        "metadata": {
            "seed": experiment_cfg["seed"],
            "image_size": dataset_cfg["image_size"],
            "raw_dir": str(raw_dir.as_posix()),
            "max_images_per_class": max_images_per_class,
            "train_images_per_class": train_images_per_class,
            "include_classes": sorted(include_classes) if include_classes else None,
            "label_map": label_map or None,
            "counts_per_split": {},
        },
    }

    for class_name, paths in class_to_paths.items():
        target_label = label_map.get(class_name, class_name)
        sampled = list(paths)
        random.shuffle(sampled)
        if max_images_per_class:
            sampled = sampled[: int(max_images_per_class)]
        as_strings = [str(path.as_posix()) for path in sampled]
        split = _split_items(
            as_strings,
            float(dataset_cfg["train_split"]),
            float(dataset_cfg["val_split"]),
        )
        if train_images_per_class is not None:
            split["train"] = split["train"][: int(train_images_per_class)]

        split_counts = {split_name: len(split_items) for split_name, split_items in split.items()}
        manifest["metadata"]["counts_per_split"][class_name] = {"target_label": target_label, **split_counts}
        for split_name, split_items in split.items():
            for item in split_items:
                manifest["splits"][split_name].append({"path": item, "label": target_label})

    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)
    print(f"Saved dataset manifest to {manifest_path}")
