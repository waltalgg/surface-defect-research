from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .config import Config
from .data import load_manifest
from .segmentation import SegmentationPredictor
from .spade.model import SpadeGenerator
from .utils import ensure_dir, set_seed


def _basic_augment_image(image: Image.Image) -> Image.Image:
    augmented = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if random.random() > 0.5 else image.copy()
    fill = _background_fill_color(augmented)
    augmented = augmented.rotate(
        random.uniform(-12, 12),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )
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

    fill = _background_fill_color(augmented)
    augmented = augmented.rotate(
        random.uniform(-18, 18),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
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


def _augment_background_image(image: Image.Image) -> Image.Image:
    augmented = image.copy()
    if random.random() > 0.5:
        augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() > 0.8:
        augmented = augmented.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    width, height = augmented.size
    crop_scale = random.uniform(0.88, 1.0)
    crop_width = max(16, int(width * crop_scale))
    crop_height = max(16, int(height * crop_scale))
    offset_x = random.randint(0, max(0, width - crop_width))
    offset_y = random.randint(0, max(0, height - crop_height))
    augmented = augmented.crop((offset_x, offset_y, offset_x + crop_width, offset_y + crop_height)).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
    augmented = ImageEnhance.Contrast(augmented).enhance(random.uniform(0.92, 1.18))
    augmented = ImageEnhance.Brightness(augmented).enhance(random.uniform(0.95, 1.08))
    augmented = ImageEnhance.Sharpness(augmented).enhance(random.uniform(0.95, 1.18))
    return augmented


def _background_fill_color(image: Image.Image) -> tuple[int, int, int]:
    stats = ImageStat.Stat(image.convert("RGB"))
    means = stats.mean if stats.mean else [128.0, 128.0, 128.0]
    return tuple(int(max(0, min(255, round(value)))) for value in means[:3])


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
    mask = Image.new("L", (patch_width, patch_height), 0)
    mask_draw = ImageDraw.Draw(mask)
    margin_x = max(4, int(patch_width * random.uniform(0.08, 0.2)))
    margin_y = max(4, int(patch_height * random.uniform(0.08, 0.2)))
    mask_draw.ellipse((margin_x, margin_y, patch_width - margin_x, patch_height - margin_y), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(6.0, 12.0)))

    merged = base.copy()
    merged.paste(blended_patch, (offset_x, offset_y), mask)
    return merged


def _estimate_defect_mask(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    blurred_large = gray.filter(ImageFilter.GaussianBlur(radius=random.uniform(5.0, 9.0)))
    blurred_small = gray.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 2.0)))
    local_anomaly = ImageChops.difference(gray, blurred_large)
    local_texture = ImageChops.difference(gray, blurred_small)
    edges = gray.filter(ImageFilter.FIND_EDGES)

    combined = ImageChops.add(local_anomaly, edges, scale=1.0)
    combined = ImageChops.add(combined, local_texture, scale=1.1)
    combined = ImageOps.autocontrast(combined)

    stats = ImageStat.Stat(combined)
    mean = stats.mean[0] if stats.mean else 32.0
    std = stats.stddev[0] if stats.stddev else 12.0
    threshold = max(18, min(72, int(mean + 0.8 * std)))

    mask = combined.point(lambda px: 255 if px > threshold else 0, mode="L")
    mask = mask.filter(ImageFilter.MinFilter(size=3))
    mask = mask.filter(ImageFilter.MaxFilter(size=5))
    mask = mask.filter(ImageFilter.MaxFilter(size=5))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.8)))
    mask = ImageEnhance.Contrast(mask).enhance(random.uniform(1.15, 1.45))
    return mask.point(lambda px: min(255, int(px * 1.2)))


def _mask_coverage(mask: Image.Image) -> float:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return float(arr.mean())


def _mask_is_usable(mask: Image.Image) -> bool:
    coverage = _mask_coverage(mask)
    return 0.01 <= coverage <= 0.45 and mask.getbbox() is not None


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


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _save_detection_debug_artifacts(
    source_image: Image.Image,
    synthetic_image: Image.Image,
    mask: Image.Image | None,
    bbox: tuple[int, int, int, int] | None,
    output_path: Path,
    label: str,
    debug_root: Path,
) -> None:
    label_dir = ensure_dir(debug_root / label)

    source_rgb = _ensure_rgb(source_image).copy()
    synthetic_rgb = _ensure_rgb(synthetic_image).copy()

    preview_size = (220, 220)
    source_preview = ImageOps.contain(source_rgb, preview_size)
    synthetic_preview = ImageOps.contain(synthetic_rgb, preview_size)
    source_canvas = Image.new("RGB", preview_size, "white")
    synthetic_canvas = Image.new("RGB", preview_size, "white")
    source_canvas.paste(
        source_preview,
        ((preview_size[0] - source_preview.width) // 2, (preview_size[1] - source_preview.height) // 2),
    )
    synthetic_canvas.paste(
        synthetic_preview,
        ((preview_size[0] - synthetic_preview.width) // 2, (preview_size[1] - synthetic_preview.height) // 2),
    )

    if mask is not None:
        mask_preview = ImageOps.contain(mask.convert("L"), preview_size)
        mask_canvas = Image.new("L", preview_size, 0)
        mask_canvas.paste(
            mask_preview,
            ((preview_size[0] - mask_preview.width) // 2, (preview_size[1] - mask_preview.height) // 2),
        )
        mask_canvas = ImageOps.colorize(mask_canvas, black=(0, 0, 0), white=(255, 120, 120))
    else:
        mask_canvas = Image.new("RGB", preview_size, (245, 245, 245))

    margin = 16
    title_h = 24
    caption_h = 24
    panel_w = preview_size[0] * 3 + margin * 4
    panel_h = preview_size[1] + title_h + caption_h + margin * 3
    panel = Image.new("RGB", (panel_w, panel_h), (250, 250, 250))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((margin, margin // 2), f"{label} | synthetic debug", fill=(20, 20, 20))

    top = margin + title_h
    xs = [margin, margin * 2 + preview_size[0], margin * 3 + preview_size[0] * 2]
    captions = ["Source image", "Detected mask", "Synthetic result"]
    images = [source_canvas, mask_canvas, synthetic_canvas]
    for x, caption, image in zip(xs, captions, images, strict=False):
        panel.paste(image, (x, top))
        panel_draw.rectangle((x, top, x + preview_size[0], top + preview_size[1]), outline=(220, 220, 220))
        panel_draw.text((x, top + preview_size[1] + 4), caption, fill=(20, 20, 20))

    stem = output_path.stem
    panel.save(label_dir / f"{stem}_debug.png")
    synthetic_rgb.save(label_dir / f"{stem}_synthetic.png")


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


def _extract_mask_crop(mask: Image.Image) -> Image.Image:
    bbox = mask.getbbox()
    width, height = mask.size
    if bbox is None:
        bbox = _random_crop_box(width, height)

    left, top, right, bottom = bbox
    pad_x = max(6, int((right - left) * random.uniform(0.18, 0.32)))
    pad_y = max(6, int((bottom - top) * random.uniform(0.18, 0.32)))
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return mask.crop(crop_box)


def _extract_crack_feature_maps(source_patch: Image.Image, patch_mask: Image.Image) -> list[Image.Image]:
    gray = source_patch.convert("L")
    alpha = np.asarray(patch_mask.convert("L"), dtype=np.float32) / 255.0
    gray_arr = np.asarray(gray, dtype=np.float32)
    source_arr = np.asarray(source_patch.convert("RGB"), dtype=np.float32)

    smooth_large = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=4.2)), dtype=np.float32)
    smooth_small = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=1.1)), dtype=np.float32)
    smooth_rgb = np.asarray(source_patch.convert("RGB").filter(ImageFilter.GaussianBlur(radius=3.2)), dtype=np.float32)
    detail = np.clip(smooth_large - gray_arr, 0.0, 255.0) / 255.0
    micro = np.clip(np.abs(gray_arr - smooth_small), 0.0, 255.0) / 255.0
    contrast = np.clip((smooth_large - smooth_small), -255.0, 255.0)
    contrast = np.abs(contrast) / 255.0
    residual_rgb = source_arr - smooth_rgb
    residual_rgb = np.where(residual_rgb < 0.0, residual_rgb * 1.4, residual_rgb * 0.7)
    residual_rgb = residual_rgb * alpha[..., None]
    grad_y, grad_x = np.gradient(gray_arr / 255.0)
    grad_x = np.clip(grad_x * alpha, -1.0, 1.0)
    grad_y = np.clip(grad_y * alpha, -1.0, 1.0)

    depth = np.clip(detail * (0.65 + 0.35 * alpha) + 0.3 * contrast * alpha, 0.0, 1.0)
    texture = np.clip(micro * (0.25 + 0.75 * alpha), 0.0, 1.0)

    depth_img = Image.fromarray((depth * 255.0).astype(np.uint8), mode="L")
    texture_img = Image.fromarray((texture * 255.0).astype(np.uint8), mode="L")
    residual_img = Image.fromarray(np.clip(residual_rgb + 128.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    grad_x_img = Image.fromarray(np.clip(grad_x * 127.0 + 128.0, 0.0, 255.0).astype(np.uint8), mode="L")
    grad_y_img = Image.fromarray(np.clip(grad_y * 127.0 + 128.0, 0.0, 255.0).astype(np.uint8), mode="L")
    return [depth_img, texture_img, residual_img, grad_x_img, grad_y_img]


def _quality_filter_mask_and_maps(mask: Image.Image, feature_maps: list[Image.Image] | None = None) -> bool:
    if not _mask_is_usable(mask):
        return False

    alpha = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    bbox = mask.getbbox()
    if bbox is None:
        return False

    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    area = float(width * height)
    coverage = float(alpha.mean())
    aspect = max(width / max(1.0, height), height / max(1.0, width))
    mask_energy = float(alpha[left:right, top:bottom].mean()) if False else float(alpha[top:bottom, left:right].mean())

    if coverage < 0.006 or coverage > 0.28:
        return False
    if area < 80:
        return False
    if aspect > 18.0:
        return False
    if mask_energy < 0.08:
        return False

    if feature_maps:
        depth = np.asarray(feature_maps[0].convert("L"), dtype=np.float32) / 255.0
        texture = np.asarray(feature_maps[1].convert("L"), dtype=np.float32) / 255.0 if len(feature_maps) > 1 else np.zeros_like(depth)
        residual = np.asarray(feature_maps[2].convert("RGB"), dtype=np.float32) - 128.0 if len(feature_maps) > 2 else np.zeros((*depth.shape, 3), dtype=np.float32)
        grad_x = (np.asarray(feature_maps[3].convert("L"), dtype=np.float32) - 128.0) / 127.0 if len(feature_maps) > 3 else np.zeros_like(depth)
        grad_y = (np.asarray(feature_maps[4].convert("L"), dtype=np.float32) - 128.0) / 127.0 if len(feature_maps) > 4 else np.zeros_like(depth)

        support = alpha > 0.12
        if support.sum() < 24:
            return False
        depth_mean = float(depth[support].mean())
        texture_mean = float(texture[support].mean())
        residual_energy = float(np.abs(residual[support]).mean())
        gradient_energy = float(np.sqrt(grad_x[support] ** 2 + grad_y[support] ** 2).mean())

        if depth_mean < 0.03:
            return False
        if texture_mean < 0.01:
            return False
        if residual_energy < 2.0:
            return False
        if gradient_energy < 0.015:
            return False

    return True


def _apply_seamless_like_blend(
    background_crop: np.ndarray,
    composite_crop: np.ndarray,
    patch_mask: Image.Image,
) -> np.ndarray:
    alpha = np.asarray(patch_mask.convert("L"), dtype=np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    soft_alpha = np.asarray(
        patch_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=random.uniform(1.4, 2.8))),
        dtype=np.float32,
    ) / 255.0
    soft_alpha = np.clip(soft_alpha, 0.0, 1.0)

    bg_img = Image.fromarray(np.clip(background_crop, 0, 255).astype(np.uint8), mode="RGB")
    comp_img = Image.fromarray(np.clip(composite_crop, 0, 255).astype(np.uint8), mode="RGB")
    bg_low = np.asarray(bg_img.filter(ImageFilter.GaussianBlur(radius=6.5)), dtype=np.float32)
    comp_low = np.asarray(comp_img.filter(ImageFilter.GaussianBlur(radius=6.5)), dtype=np.float32)
    low_freq_delta = comp_low - bg_low

    corrected = composite_crop - low_freq_delta * (0.55 * soft_alpha[..., None])
    edge_band = np.clip(soft_alpha - alpha, 0.0, 1.0)
    edge_mix = background_crop * edge_band[..., None] + corrected * (1.0 - edge_band[..., None])
    final = background_crop * (1.0 - soft_alpha[..., None]) + edge_mix * soft_alpha[..., None]
    return np.clip(final, 0, 255)


def _deform_patch(patch: Image.Image, patch_mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    fill = _background_fill_color(patch)
    angle = random.uniform(-35, 35)
    patch = patch.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=True,
        fillcolor=fill,
    )
    patch_mask = patch_mask.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

    scale_x = random.uniform(0.55, 1.45)
    scale_y = random.uniform(0.55, 1.45)
    new_width = max(20, int(patch.width * scale_x))
    new_height = max(20, int(patch.height * scale_y))
    patch = patch.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    patch_mask = patch_mask.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

    if random.random() > 0.5:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        patch_mask = patch_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() > 0.4:
        shear = random.uniform(-0.35, 0.35)
        patch = patch.transform(
            patch.size,
            Image.Transform.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill,
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
    if random.random() > 0.45:
        width, height = patch.size
        shift = max(3, int(min(width, height) * random.uniform(0.05, 0.14)))
        coeffs = (
            1,
            random.uniform(-0.12, 0.12),
            random.randint(-shift, shift),
            random.uniform(-0.12, 0.12),
            1,
            random.randint(-shift, shift),
        )
        patch = patch.transform(patch.size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR)
        patch_mask = patch_mask.transform(
            patch_mask.size,
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
        )
    patch_mask = patch_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 2.2)))
    return patch, patch_mask


def _deform_mask_only(
    mask: Image.Image,
    extra_maps: list[Image.Image] | None = None,
) -> Image.Image | tuple[Image.Image, list[Image.Image]]:
    maps = [feature.copy() for feature in (extra_maps or [])]
    angle = random.uniform(-40, 40)
    mask = mask.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
    maps = [feature.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True) for feature in maps]

    scale_x = random.uniform(0.5, 1.55)
    scale_y = random.uniform(0.5, 1.55)
    new_width = max(20, int(mask.width * scale_x))
    new_height = max(20, int(mask.height * scale_y))
    mask = mask.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    maps = [feature.resize((new_width, new_height), resample=Image.Resampling.BILINEAR) for feature in maps]

    if random.random() > 0.5:
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        maps = [feature.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for feature in maps]

    if random.random() > 0.45:
        shear = random.uniform(-0.38, 0.38)
        mask = mask.transform(
            mask.size,
            Image.Transform.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
        )
        maps = [
            feature.transform(
                feature.size,
                Image.Transform.AFFINE,
                (1, shear, 0, 0, 1, 0),
                resample=Image.Resampling.BILINEAR,
            )
            for feature in maps
        ]

    if random.random() > 0.45:
        width, height = mask.size
        shift = max(3, int(min(width, height) * random.uniform(0.05, 0.16)))
        coeffs = (
            1,
            random.uniform(-0.14, 0.14),
            random.randint(-shift, shift),
            random.uniform(-0.14, 0.14),
            1,
            random.randint(-shift, shift),
        )
        mask = mask.transform(
            mask.size,
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
        )
        maps = [
            feature.transform(
                feature.size,
                Image.Transform.AFFINE,
                coeffs,
                resample=Image.Resampling.BILINEAR,
            )
            for feature in maps
        ]

    mask = mask.filter(ImageFilter.MaxFilter(size=3))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.7, 1.8)))
    mask = ImageEnhance.Contrast(mask).enhance(random.uniform(1.15, 1.5))
    maps = [feature.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9))) for feature in maps]
    if extra_maps is None:
        return mask
    return mask, maps


def _build_structural_patch(
    background_crop: Image.Image,
    patch: Image.Image,
    patch_mask: Image.Image,
) -> Image.Image:
    patch = _match_patch_to_background(patch, background_crop)
    patch_arr = np.asarray(patch, dtype=np.float32)
    bg_arr = np.asarray(background_crop, dtype=np.float32)
    smooth_patch = np.asarray(
        patch.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.4, 2.6))),
        dtype=np.float32,
    )
    detail = patch_arr - smooth_patch
    alpha = np.asarray(patch_mask.convert("L"), dtype=np.float32) / 255.0
    alpha = np.clip(alpha * random.uniform(0.9, 1.15), 0.0, 1.0)[..., None]

    detail_strength = random.uniform(0.8, 1.2)
    stain_strength = random.uniform(0.08, 0.22)
    low_freq_residual = smooth_patch - smooth_patch.mean(axis=(0, 1), keepdims=True)
    blended = bg_arr + detail * detail_strength * alpha + low_freq_residual * stain_strength * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _build_mask_guided_crack_patch(
    background_crop: Image.Image,
    patch_mask: Image.Image,
    feature_maps: list[Image.Image] | None = None,
) -> Image.Image:
    bg_arr = np.asarray(background_crop, dtype=np.float32)

    bg_gray = bg_arr.mean(axis=2, keepdims=True)

    alpha_img = patch_mask.convert("L")
    alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
    alpha = np.clip(alpha * random.uniform(0.95, 1.15), 0.0, 1.0)

    center_img = alpha_img.filter(ImageFilter.MinFilter(size=3))
    center_img = center_img.filter(ImageFilter.MinFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=0.35))
    wall_img = alpha_img.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.05))
    outer_img = alpha_img.filter(ImageFilter.MaxFilter(size=11)).filter(ImageFilter.GaussianBlur(radius=1.8))

    center = np.asarray(center_img, dtype=np.float32) / 255.0
    wall = np.asarray(wall_img, dtype=np.float32) / 255.0
    outer = np.asarray(outer_img, dtype=np.float32) / 255.0
    wall = np.clip(wall - 0.62 * center, 0.0, 1.0)
    outer = np.clip(outer - 0.55 * wall - 0.35 * center, 0.0, 1.0)

    light_dx = random.choice([-2, -1, 1, 2])
    light_dy = random.choice([-2, -1, 1, 2])
    bright_shift = ImageChops.offset(alpha_img, light_dx, light_dy)
    dark_shift = ImageChops.offset(alpha_img, -light_dx, -light_dy)
    bright_rim = np.asarray(ImageChops.subtract(bright_shift, alpha_img), dtype=np.float32) / 255.0
    dark_rim = np.asarray(ImageChops.subtract(dark_shift, alpha_img), dtype=np.float32) / 255.0
    bright_rim = np.clip(bright_rim, 0.0, 1.0)
    dark_rim = np.clip(dark_rim, 0.0, 1.0)

    transferred_residual = None
    if feature_maps:
        transferred_depth = np.asarray(feature_maps[0].convert("L"), dtype=np.float32) / 255.0
        transferred_depth = np.clip(transferred_depth, 0.0, 1.0)
        if len(feature_maps) > 1:
            transferred_texture = np.asarray(feature_maps[1].convert("L"), dtype=np.float32) / 255.0
            transferred_texture = np.clip(transferred_texture, 0.0, 1.0)
        else:
            transferred_texture = np.zeros_like(alpha)
        if len(feature_maps) > 2:
            transferred_residual = np.asarray(feature_maps[2].convert("RGB"), dtype=np.float32) - 128.0
        else:
            transferred_residual = None
        if len(feature_maps) > 3:
            transferred_grad_x = (np.asarray(feature_maps[3].convert("L"), dtype=np.float32) - 128.0) / 127.0
        else:
            transferred_grad_x = np.zeros_like(alpha)
        if len(feature_maps) > 4:
            transferred_grad_y = (np.asarray(feature_maps[4].convert("L"), dtype=np.float32) - 128.0) / 127.0
        else:
            transferred_grad_y = np.zeros_like(alpha)
    else:
        transferred_depth = np.clip(alpha * 0.75 + center * 0.4, 0.0, 1.0)
        transferred_texture = np.zeros_like(alpha)
        transferred_grad_x = np.zeros_like(alpha)
        transferred_grad_y = np.zeros_like(alpha)

    grad_y, grad_x = np.gradient(transferred_depth)
    grad_x = 0.55 * grad_x + 0.45 * transferred_grad_x
    grad_y = 0.55 * grad_y + 0.45 * transferred_grad_y
    norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    light_dx_f = float(random.choice([-1.4, -0.9, 0.9, 1.4]))
    light_dy_f = float(random.choice([-1.4, -0.9, 0.9, 1.4]))
    directional = (grad_x * light_dx_f + grad_y * light_dy_f) / norm
    gradient_shadow = np.clip(-directional, 0.0, 1.0)
    gradient_highlight = np.clip(directional, 0.0, 1.0)

    crack_strength = random.uniform(0.48, 0.7)
    wall_strength = random.uniform(0.10, 0.18)
    shadow_strength = random.uniform(0.06, 0.13)
    highlight_strength = random.uniform(0.03, 0.08)
    texture_strength = random.uniform(0.015, 0.04)
    grain = np.random.normal(loc=0.0, scale=random.uniform(0.25, 0.7), size=bg_arr.shape[:2]).astype(np.float32)

    core_profile = np.clip(0.42 * alpha + 0.58 * transferred_depth, 0.0, 1.0)
    core_darkening = crack_strength * np.maximum(center, core_profile)
    wall_darkening = wall_strength * np.clip(wall + 0.45 * transferred_depth, 0.0, 1.0)
    directional_shadow = shadow_strength * np.clip(
        0.55 * dark_rim + 0.45 * gradient_shadow,
        0.0,
        1.0,
    ) * np.clip(wall + 0.45 * outer, 0.0, 1.0)
    total_darkening = np.clip(core_darkening + wall_darkening + directional_shadow, 0.0, 0.75)

    shading = bg_gray * total_darkening[..., None]
    highlight_weight = highlight_strength * np.clip(
        0.5 * bright_rim + 0.5 * gradient_highlight,
        0.0,
        1.0,
    ) * np.clip(wall + 0.4 * outer, 0.0, 1.0)
    highlight = bg_gray * highlight_weight[..., None]
    texture_profile = np.clip(0.3 * outer + 0.45 * wall + 0.7 * transferred_texture, 0.0, 1.0)
    texture = grain[..., None] * texture_strength * texture_profile[..., None]
    transferred_low = transferred_texture[..., None] * random.uniform(10.0, 22.0)
    residual_transfer = np.zeros_like(bg_arr)
    if transferred_residual is not None:
        residual_strength = random.uniform(0.95, 1.35)
        residual_transfer = transferred_residual * residual_strength * np.clip(0.35 + 0.65 * alpha[..., None], 0.0, 1.0)
    gradient_transfer = np.zeros_like(bg_arr)
    gradient_field = np.sqrt(grad_x**2 + grad_y**2)
    gradient_transfer[..., 0] -= 9.0 * gradient_field * np.clip(alpha + wall, 0.0, 1.0)
    gradient_transfer[..., 1] -= 7.0 * gradient_field * np.clip(alpha + 0.8 * wall, 0.0, 1.0)
    gradient_transfer[..., 2] -= 5.0 * gradient_field * np.clip(alpha + 0.5 * wall, 0.0, 1.0)

    tint = np.zeros_like(bg_arr)
    tint[..., 2] = 3.2 * center + 1.3 * wall + 1.4 * transferred_depth
    tint[..., 1] = 0.8 * outer + 0.7 * transferred_texture

    blended = bg_arr + residual_transfer + gradient_transfer - shading + highlight + texture - tint - transferred_low
    blended = _apply_seamless_like_blend(bg_arr, blended, patch_mask)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _paste_patch_on_background(background: Image.Image, patch: Image.Image, patch_mask: Image.Image) -> Image.Image:
    canvas = background.copy()
    max_x = max(0, canvas.width - patch.width)
    max_y = max(0, canvas.height - patch.height)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)
    bg_crop = canvas.crop((offset_x, offset_y, offset_x + patch.width, offset_y + patch.height))
    structured_patch = _build_structural_patch(bg_crop, patch, patch_mask)
    patch_mask = ImageEnhance.Contrast(patch_mask).enhance(random.uniform(1.1, 1.5))
    patch_mask = patch_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5)))
    canvas.paste(structured_patch, (offset_x, offset_y), patch_mask)
    if random.random() > 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))
    if random.random() > 0.6:
        canvas = ImageEnhance.Sharpness(canvas).enhance(random.uniform(0.95, 1.15))
    return canvas


def _paste_crack_mask_on_background(
    background: Image.Image,
    patch_mask: Image.Image,
    feature_maps: list[Image.Image] | None = None,
) -> Image.Image:
    canvas = background.copy()
    max_x = max(0, canvas.width - patch_mask.width)
    max_y = max(0, canvas.height - patch_mask.height)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)
    bg_crop = canvas.crop((offset_x, offset_y, offset_x + patch_mask.width, offset_y + patch_mask.height))
    structured_patch = _build_mask_guided_crack_patch(bg_crop, patch_mask, feature_maps=feature_maps)
    soft_mask = ImageEnhance.Contrast(patch_mask).enhance(random.uniform(1.25, 1.6))
    soft_mask = soft_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.8)))
    canvas.paste(structured_patch, (offset_x, offset_y), soft_mask)
    canvas = ImageEnhance.Sharpness(canvas).enhance(random.uniform(1.0, 1.15))
    return canvas


def _choose_background_label(class_pools: dict[str, list[dict[str, str]]], label: str) -> str:
    if label != "no_defect" and "no_defect" in class_pools:
        return "no_defect"
    return label


def _composite_defect_images(
    image: Image.Image,
    partner: Image.Image,
    label: str,
    predictor: SegmentationPredictor | None = None,
) -> tuple[Image.Image, Image.Image | None, tuple[int, int, int, int] | None]:
    source = _strong_augment_image(image)
    background_augment = _augment_background_image(partner) if predictor is not None else _strong_augment_image(partner)
    background = background_augment.resize(source.size, resample=Image.Resampling.BILINEAR)
    if label == "no_defect":
        return _strong_augment_image(image), None, None

    mask = predictor.predict_mask(source) if predictor is not None else _estimate_defect_mask(source)
    if not _mask_is_usable(mask):
        return _strong_augment_image(image), mask, mask.getbbox()
    if predictor is not None:
        source_patch, mask_crop = _extract_patch_with_mask(source, mask)
        feature_maps = _extract_crack_feature_maps(source_patch, mask_crop)
        patch_mask, feature_maps = _deform_mask_only(mask_crop, feature_maps)
        if not _quality_filter_mask_and_maps(patch_mask, feature_maps):
            return _strong_augment_image(image), mask, mask.getbbox()
        return _paste_crack_mask_on_background(background, patch_mask, feature_maps=feature_maps), mask, mask.getbbox()
    patch, patch_mask = _extract_patch_with_mask(source, mask)
    patch, patch_mask = _deform_patch(patch, patch_mask)
    if not _mask_is_usable(patch_mask):
        return _strong_augment_image(image), mask, mask.getbbox()
    return _paste_patch_on_background(background, patch, patch_mask), mask, mask.getbbox()


def _build_class_pools(train_items: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in train_items:
        pools[item["label"]].append(item)
    return dict(pools)


def _load_spade_generator(checkpoint_path: str, device: str, base_channels: int) -> SpadeGenerator:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = SpadeGenerator(base_channels=base_channels).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()
    return generator


def _resolve_spade_mask_path(source_path: Path, synthetic_cfg: dict) -> Path | None:
    mask_dir = synthetic_cfg.get("spade_mask_dir")
    if not mask_dir:
        return None
    candidate = Path(mask_dir) / f"{source_path.stem}.png"
    return candidate if candidate.exists() else None


def _transform_spade_mask(mask: Image.Image, canvas_size: int) -> Image.Image:
    transformed = mask.convert("L")
    if random.random() > 0.5:
        transformed = transformed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() > 0.7:
        transformed = transformed.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    angle = random.uniform(-22.0, 22.0)
    transformed = transformed.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    bbox = transformed.getbbox()
    if bbox is not None:
        transformed = transformed.crop(bbox)

    crop_w, crop_h = transformed.size
    scale = random.uniform(0.75, 1.3)
    target_w = max(16, min(canvas_size - 8, int(crop_w * scale)))
    target_h = max(16, min(canvas_size - 8, int(crop_h * scale)))
    transformed = transformed.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    max_x = max(0, canvas_size - target_w)
    max_y = max(0, canvas_size - target_h)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)
    canvas.paste(transformed, (offset_x, offset_y))

    canvas = canvas.filter(ImageFilter.MaxFilter(size=3))
    return canvas.point(lambda px: 255 if px > 20 else 0, mode="L")


def _generate_spade_image(
    generator: SpadeGenerator,
    mask: Image.Image,
    device: str,
    image_size: int,
) -> Image.Image:
    mask_array = (np.asarray(mask.convert("L").resize((image_size, image_size), resample=Image.Resampling.NEAREST)) > 0).astype(
        np.float32
    )
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        generated = generator(mask_tensor)[0].detach().cpu()
    generated = ((generated.clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0
    generated_array = generated.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(generated_array, mode="RGB")


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
    train_items = [item for item in manifest["splits"]["train"] if "source_paths" not in item]
    if not train_items:
        raise ValueError("Synthetic generation requires real train items, but only generated items were found.")
    target_count = max(0, int(round(len(train_items) * multiplier)))
    method = str(synthetic_cfg.get("method", "basic")).lower()
    class_pools = _build_class_pools(train_items)
    save_debug_artifacts = bool(synthetic_cfg.get("save_debug_artifacts", False))
    debug_root = ensure_dir(
        Path(synthetic_cfg.get("debug_output_dir", f"artifacts/synthetic_debug/{config['experiment']['name']}"))
    )
    mask_source = str(synthetic_cfg.get("mask_source", "heuristic")).lower()
    segmentation_predictor = None
    spade_generator = None
    if method == "composite" and mask_source == "segmentation":
        checkpoint_path = synthetic_cfg.get("segmentation_checkpoint_path")
        if not checkpoint_path:
            raise ValueError("synthetic.mask_source=segmentation requires synthetic.segmentation_checkpoint_path")
        segmentation_predictor = SegmentationPredictor(
            checkpoint_path=checkpoint_path,
            device=str(config["training"]["device"]),
            image_size=int(synthetic_cfg.get("segmentation_image_size", config["dataset"]["image_size"])),
            base_channels=int(synthetic_cfg.get("segmentation_base_channels", 32)),
        )
    elif method == "spade":
        checkpoint_path = synthetic_cfg.get("spade_checkpoint_path")
        if not checkpoint_path:
            raise ValueError("synthetic.method=spade requires synthetic.spade_checkpoint_path")
        spade_generator = _load_spade_generator(
            checkpoint_path=str(checkpoint_path),
            device=str(config["training"]["device"]),
            base_channels=int(synthetic_cfg.get("spade_base_channels", 64)),
        )

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
        debug_mask = None
        debug_bbox = None
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
                    synthetic, debug_mask, debug_bbox = _composite_defect_images(
                        image,
                        partner_image,
                        item["label"],
                        predictor=segmentation_predictor,
                    )
            elif method == "spade":
                if item["label"] == "defect":
                    mask_path = _resolve_spade_mask_path(source_path, synthetic_cfg)
                    if mask_path is None:
                        synthetic = _strong_augment_image(image)
                    else:
                        source_paths.append(str(mask_path.as_posix()))
                        with Image.open(mask_path).convert("L") as mask_image:
                            transformed_mask = _transform_spade_mask(
                                mask_image,
                                int(synthetic_cfg.get("spade_image_size", 256)),
                            )
                        debug_mask = transformed_mask.resize(image.size, resample=Image.Resampling.NEAREST)
                        debug_bbox = debug_mask.getbbox()
                        synthetic = _generate_spade_image(
                            generator=spade_generator,
                            mask=transformed_mask,
                            device=str(config["training"]["device"]),
                            image_size=int(synthetic_cfg.get("spade_image_size", 256)),
                        ).resize(image.size, resample=Image.Resampling.BILINEAR)
                else:
                    synthetic = _augment_background_image(image)
            else:
                raise ValueError(f"Unsupported synthetic method: {method}")
            synthetic.save(target_path)
            if save_debug_artifacts:
                _save_detection_debug_artifacts(
                    source_image=image,
                    synthetic_image=synthetic,
                    mask=debug_mask,
                    bbox=debug_bbox,
                    output_path=target_path,
                    label=item["label"],
                    debug_root=debug_root,
                )
        generated.append(
            {
                "path": str(target_path.as_posix()),
                "label": item["label"],
                "method": method,
                "source_paths": source_paths,
                "detected_bbox": list(debug_bbox) if debug_bbox is not None else None,
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
                "train": train_items + generated,
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
