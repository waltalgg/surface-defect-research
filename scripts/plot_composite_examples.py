from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


MANIFEST_PATH = Path("data/processed/synthetic_magnetic_tile_binary_small_composite_gpu/synthetic_manifest.json")
OUTPUT_DIR = Path("artifacts/plots_synthetic_composite")

BG_COLOR = (248, 248, 248)
TEXT_COLOR = (20, 20, 20)
LINE_COLOR = (220, 220, 220)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        canvas.paste(image, offset)
        return canvas


def build_gallery(items: list[dict], title: str, output_name: str, per_row: int = 3) -> None:
    margin = 16
    title_h = 30
    caption_h = 34
    cell_w, cell_h = 170, 170
    rows = (len(items) + per_row - 1) // per_row
    width = margin * (per_row + 1) + per_row * (cell_w * 2)
    height = margin * (rows + 2) + title_h + rows * (cell_h + caption_h)

    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=TEXT_COLOR)

    for idx, item in enumerate(items):
        row = idx // per_row
        col = idx % per_row
        top = margin + title_h + margin + row * (cell_h + caption_h + margin)
        left = margin + col * ((cell_w * 2) + margin)

        real_path = Path(item["source_paths"][0])
        synth_path = Path(item["path"])
        real_img = fit_image(real_path, (cell_w, cell_h))
        synth_img = fit_image(synth_path, (cell_w, cell_h))

        canvas.paste(real_img, (left, top))
        canvas.paste(synth_img, (left + cell_w, top))
        draw.rectangle((left, top, left + cell_w, top + cell_h), outline=LINE_COLOR)
        draw.rectangle((left + cell_w, top, left + cell_w * 2, top + cell_h), outline=LINE_COLOR)
        draw.text((left, top + cell_h + 4), "Real", fill=TEXT_COLOR)
        draw.text((left + cell_w, top + cell_h + 4), "Composite", fill=TEXT_COLOR)

    ensure_dir(OUTPUT_DIR)
    canvas.save(OUTPUT_DIR / output_name)


def main() -> None:
    manifest = load_manifest()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in manifest["generated"]:
        grouped[item["label"]].append(item)

    defect_items = grouped.get("defect", [])[:9]
    no_defect_items = grouped.get("no_defect", [])[:6]
    mixed_items = defect_items[:6] + no_defect_items[:3]

    build_gallery(defect_items, "Composite Defect Examples / Примеры composite defect", "01_defect_gallery.png")
    build_gallery(
        no_defect_items,
        "Composite No-Defect Examples / Примеры composite no_defect",
        "02_no_defect_gallery.png",
    )
    build_gallery(mixed_items, "Composite Mixed Gallery / Смешанная галерея composite", "03_mixed_gallery.png")
    print(f"Saved galleries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
