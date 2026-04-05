from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from .config import Config
from .data import load_manifest
from .utils import ensure_dir, set_seed


def _augment_image(image: Image.Image) -> Image.Image:
    augmented = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if random.random() > 0.5 else image.copy()
    augmented = augmented.rotate(random.uniform(-12, 12))
    augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.2)))
    contrast = ImageEnhance.Contrast(augmented)
    return contrast.enhance(random.uniform(0.9, 1.2))


def generate_synthetic_dataset(config: Config) -> None:
    synthetic_cfg = config["synthetic"]
    if not synthetic_cfg["enabled"]:
        print("Synthetic generation is disabled in config.")
        return

    manifest = load_manifest(config["dataset"]["manifest_path"])
    output_dir = ensure_dir(synthetic_cfg["output_dir"])
    set_seed(int(config["experiment"]["seed"]))

    generated = []
    multiplier = float(synthetic_cfg["multiplier"])
    train_items = manifest["splits"]["train"]
    target_count = max(0, int(round(len(train_items) * multiplier)))

    if target_count == 0:
        print("Synthetic multiplier produced zero samples; nothing to generate.")
        return

    sampled_items = [train_items[index % len(train_items)] for index in range(target_count)]
    random.shuffle(sampled_items)

    for sample_idx, item in enumerate(sampled_items):
        source_path = Path(item["path"])
        class_dir = ensure_dir(output_dir / item["label"])
        target_path = class_dir / f"{source_path.stem}_synthetic_{sample_idx}{source_path.suffix}"
        with Image.open(source_path).convert("RGB") as image:
            synthetic = _augment_image(image)
            synthetic.save(target_path)
        generated.append({"path": str(target_path.as_posix()), "label": item["label"]})

    summary = {"generated": generated}
    summary_path = Path(output_dir) / "synthetic_manifest.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    merged_manifest_path = synthetic_cfg.get("merged_manifest_path")
    if merged_manifest_path:
        merged_manifest = {
            "classes": manifest["classes"],
            "splits": {
                "train": manifest["splits"]["train"] + generated,
                "val": manifest["splits"]["val"],
                "test": manifest["splits"]["test"],
            },
            "metadata": {
                **manifest.get("metadata", {}),
                "synthetic_enabled": True,
                "synthetic_multiplier": multiplier,
                "synthetic_count": len(generated),
                "synthetic_output_dir": str(output_dir.as_posix()),
            },
        }
        merged_manifest_file = Path(merged_manifest_path)
        ensure_dir(merged_manifest_file.parent)
        with merged_manifest_file.open("w", encoding="utf-8") as fh:
            json.dump(merged_manifest, fh, indent=2)
        print(f"Saved merged manifest to {merged_manifest_file}")

    print(f"Generated {len(generated)} synthetic images in {output_dir}")
