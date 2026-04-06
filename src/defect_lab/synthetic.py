from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .config import Config
from .data import load_manifest
from .utils import ensure_dir, set_seed


def _basic_augment_image(image: Image.Image) -> Image.Image:
    augmented = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if random.random() > 0.5 else image.copy()
    augmented = augmented.rotate(random.uniform(-12, 12))
    augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.2)))
    contrast = ImageEnhance.Contrast(augmented)
    return contrast.enhance(random.uniform(0.9, 1.2))


def _strong_augment_image(image: Image.Image) -> Image.Image:
    augmented = image.copy()
    if random.random() > 0.5:
        augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() > 0.7:
        augmented = augmented.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    width, height = augmented.size
    crop_scale = random.uniform(0.82, 1.0)
    crop_width = max(16, int(width * crop_scale))
    crop_height = max(16, int(height * crop_scale))
    offset_x = random.randint(0, max(0, width - crop_width))
    offset_y = random.randint(0, max(0, height - crop_height))
    augmented = augmented.crop((offset_x, offset_y, offset_x + crop_width, offset_y + crop_height)).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )

    augmented = augmented.rotate(
        random.uniform(-18, 18),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    augmented = ImageEnhance.Contrast(augmented).enhance(random.uniform(0.8, 1.35))
    augmented = ImageEnhance.Brightness(augmented).enhance(random.uniform(0.9, 1.12))
    augmented = ImageEnhance.Sharpness(augmented).enhance(random.uniform(0.8, 1.4))
    if random.random() > 0.5:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.0)))
    else:
        augmented = augmented.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    if random.random() > 0.6:
        augmented = ImageOps.autocontrast(augmented)
    return augmented


def _blend_same_class_images(image: Image.Image, partner: Image.Image) -> Image.Image:
    base = _strong_augment_image(image)
    partner_aug = _strong_augment_image(partner).resize(base.size, resample=Image.Resampling.BILINEAR)

    width, height = base.size
    patch_scale = random.uniform(0.35, 0.7)
    patch_width = max(24, int(width * patch_scale))
    patch_height = max(24, int(height * patch_scale))
    offset_x = random.randint(0, max(0, width - patch_width))
    offset_y = random.randint(0, max(0, height - patch_height))

    base_patch = base.crop((offset_x, offset_y, offset_x + patch_width, offset_y + patch_height))
    partner_patch = partner_aug.crop((offset_x, offset_y, offset_x + patch_width, offset_y + patch_height))
    alpha = random.uniform(0.35, 0.65)
    blended_patch = Image.blend(base_patch, partner_patch, alpha=alpha)

    merged = base.copy()
    merged.paste(blended_patch, (offset_x, offset_y))
    return merged


def _estimate_defect_mask(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=random.uniform(5.0, 9.0)))
    difference = ImageChops.difference(gray, blurred)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    combined = ImageChops.add(difference, edges, scale=1.2)
    threshold = random.randint(18, 34)
    mask = combined.point(lambda px: 255 if px > threshold else 0, mode="L")
    mask = mask.filter(ImageFilter.MinFilter(size=3))
    mask = mask.filter(ImageFilter.MaxFilter(size=5))
    mask = mask.filter(ImageFilter.MaxFilter(size=5))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 2.4)))
    return mask.point(lambda px: min(255, int(px * 1.25)))


def _match_patch_to_background(patch: Image.Image, background_crop: Image.Image) -> Image.Image:
    patch_stats = ImageStat.Stat(patch.convert("L"))
    bg_stats = ImageStat.Stat(background_crop.convert("L"))
    patch_mean = patch_stats.mean[0] if patch_stats.mean else 128.0
    bg_mean = bg_stats.mean[0] if bg_stats.mean else 128.0
    patch_std = patch_stats.stddev[0] if patch_stats.stddev else 20.0
    bg_std = bg_stats.stddev[0] if bg_stats.stddev else 20.0

    brightness_ratio = max(0.75, min(1.25, bg_mean / max(1.0, patch_mean)))
    contrast_ratio = max(0.8, min(1.35, bg_std / max(1.0, patch_std)))

    patch = ImageEnhance.Brightness(patch).enhance(brightness_ratio)
    patch = ImageEnhance.Contrast(patch).enhance(contrast_ratio)
    return patch


def _random_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    crop_scale = random.uniform(0.35, 0.7)
    crop_width = max(24, int(width * crop_scale))
    crop_height = max(24, int(height * crop_scale))
    offset_x = random.randint(0, max(0, width - crop_width))
    offset_y = random.randint(0, max(0, height - crop_height))
    return offset_x, offset_y, offset_x + crop_width, offset_y + crop_height


def _extract_patch_with_mask(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    bbox = mask.getbbox()
    width, height = image.size
    if bbox is None:
        bbox = _random_crop_box(width, height)

    left, top, right, bottom = bbox
    pad_x = max(8, int((right - left) * random.uniform(0.12, 0.25)))
    pad_y = max(8, int((bottom - top) * random.uniform(0.12, 0.25)))
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return image.crop(crop_box), mask.crop(crop_box)


def _deform_patch(patch: Image.Image, patch_mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    angle = random.uniform(-20, 20)
    patch = patch.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
    patch_mask = patch_mask.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

    scale_x = random.uniform(0.7, 1.25)
    scale_y = random.uniform(0.7, 1.25)
    new_width = max(20, int(patch.width * scale_x))
    new_height = max(20, int(patch.height * scale_y))
    patch = patch.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    patch_mask = patch_mask.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

    if random.random() > 0.5:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        patch_mask = patch_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() > 0.4:
        shear = random.uniform(-0.22, 0.22)
        patch = patch.transform(
            patch.size,
            Image.Transform.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
        )
        patch_mask = patch_mask.transform(
            patch_mask.size,
            Image.Transform.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
        )

    patch = ImageEnhance.Contrast(patch).enhance(random.uniform(0.9, 1.25))
    patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.9, 1.12))
    patch = ImageEnhance.Sharpness(patch).enhance(random.uniform(0.9, 1.3))
    patch_mask = patch_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.5)))
    return patch, patch_mask


def _paste_patch_on_background(background: Image.Image, patch: Image.Image, patch_mask: Image.Image) -> Image.Image:
    canvas = background.copy()
    max_x = max(0, canvas.width - patch.width)
    max_y = max(0, canvas.height - patch.height)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)
    bg_crop = canvas.crop((offset_x, offset_y, offset_x + patch.width, offset_y + patch.height))
    patch = _match_patch_to_background(patch, bg_crop)
    patch_mask = ImageEnhance.Contrast(patch_mask).enhance(random.uniform(1.1, 1.5))
    patch_mask = patch_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5)))
    canvas.paste(patch, (offset_x, offset_y), patch_mask)
    if random.random() > 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))
    if random.random() > 0.6:
        canvas = ImageEnhance.Sharpness(canvas).enhance(random.uniform(0.95, 1.15))
    return canvas


def _choose_background_label(class_pools: dict[str, list[dict[str, str]]], label: str) -> str:
    if label != "no_defect" and "no_defect" in class_pools:
        return "no_defect"
    return label


def _composite_defect_images(
    image: Image.Image,
    partner: Image.Image,
    label: str,
) -> Image.Image:
    source = _strong_augment_image(image)
    background = _strong_augment_image(partner).resize(source.size, resample=Image.Resampling.BILINEAR)
    if label == "no_defect":
        return _blend_same_class_images(source, background)

    mask = _estimate_defect_mask(source)
    patch, patch_mask = _extract_patch_with_mask(source, mask)
    patch, patch_mask = _deform_patch(patch, patch_mask)
    return _paste_patch_on_background(background, patch, patch_mask)


def _build_class_pools(train_items: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in train_items:
        pools[item["label"]].append(item)
    return dict(pools)


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
    method = str(synthetic_cfg.get("method", "basic")).lower()
    class_pools = _build_class_pools(train_items)

    if target_count == 0:
        print("Synthetic multiplier produced zero samples; nothing to generate.")
        return

    sampled_items = [train_items[index % len(train_items)] for index in range(target_count)]
    random.shuffle(sampled_items)

    for sample_idx, item in enumerate(sampled_items):
        source_path = Path(item["path"])
        class_dir = ensure_dir(output_dir / item["label"])
        target_path = class_dir / f"{source_path.stem}_synthetic_{sample_idx}{source_path.suffix}"
        source_paths = [str(source_path.as_posix())]
        with Image.open(source_path).convert("RGB") as image:
            if method == "basic":
                synthetic = _basic_augment_image(image)
            elif method == "strong":
                synthetic = _strong_augment_image(image)
            elif method == "blend":
                pool = class_pools[item["label"]]
                partner_item = random.choice(pool)
                source_paths.append(partner_item["path"])
                with Image.open(partner_item["path"]).convert("RGB") as partner_image:
                    synthetic = _blend_same_class_images(image, partner_image)
            elif method == "composite":
                background_label = _choose_background_label(class_pools, item["label"])
                partner_item = random.choice(class_pools[background_label])
                source_paths.append(partner_item["path"])
                with Image.open(partner_item["path"]).convert("RGB") as partner_image:
                    synthetic = _composite_defect_images(image, partner_image, item["label"])
            else:
                raise ValueError(f"Unsupported synthetic method: {method}")
            synthetic.save(target_path)
        generated.append(
            {
                "path": str(target_path.as_posix()),
                "label": item["label"],
                "method": method,
                "source_paths": source_paths,
            }
        )

    summary = {"generated": generated, "method": method}
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
                "synthetic_method": method,
                "synthetic_output_dir": str(output_dir.as_posix()),
            },
        }
        merged_manifest_file = Path(merged_manifest_path)
        ensure_dir(merged_manifest_file.parent)
        with merged_manifest_file.open("w", encoding="utf-8") as fh:
            json.dump(merged_manifest, fh, indent=2)
        print(f"Saved merged manifest to {merged_manifest_file}")

    print(f"Generated {len(generated)} synthetic images in {output_dir}")
