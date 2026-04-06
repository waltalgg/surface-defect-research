from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


RUNS_DIR = Path("artifacts/runs")
REPORTS_DIR = Path("artifacts/reports")
PLOTS_DIR = Path("artifacts/plots_synthetic")


NEU_RUNS = {
    "small": "neu_resnet18_small_gpu_100",
    "medium": "neu_resnet18_medium_gpu_100",
    "full": "neu_resnet18_full_gpu_100",
    "synth_half": "neu_resnet18_small_synth_half_gpu_100",
    "synth": "neu_resnet18_small_synth_gpu_100",
    "synth_double": "neu_resnet18_small_synth_double_gpu_100",
}

MAG_RUNS = {
    "small": "magnetic_tile_resnet18_binary_small_balanced_gpu_100",
    "medium": "magnetic_tile_resnet18_binary_medium_balanced_gpu_100",
    "full": "magnetic_tile_resnet18_binary_full_balanced_gpu_100",
    "synth_half": "magnetic_tile_resnet18_binary_small_synth_half_balanced_gpu_100",
    "synth": "magnetic_tile_resnet18_binary_small_synth_balanced_gpu_100",
    "synth_double": "magnetic_tile_resnet18_binary_small_synth_double_balanced_gpu_100",
}

NEU_SYNTH_MANIFEST = Path("data/processed/synthetic_neu_small_double_gpu_100/synthetic_manifest.json")
MAG_SYNTH_MANIFEST = Path("data/processed/synthetic_magnetic_tile_binary_small_double_balanced_gpu_100/synthetic_manifest.json")

FONT_COLOR = (20, 20, 20)
BG_COLOR = (250, 250, 250)
LINE_COLOR = (220, 220, 220)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metrics(run_name: str) -> dict:
    payload = load_json(RUNS_DIR / run_name / "test_metrics.json")
    return payload["metrics"] | {"test_loss": payload["test_loss"]}


def write_summary_tables() -> None:
    rows = [
        ("NEU", "small", load_metrics(NEU_RUNS["small"])),
        ("NEU", "medium", load_metrics(NEU_RUNS["medium"])),
        ("NEU", "full", load_metrics(NEU_RUNS["full"])),
        ("NEU", "small + 0.5x synth", load_metrics(NEU_RUNS["synth_half"])),
        ("NEU", "small + 1.0x synth", load_metrics(NEU_RUNS["synth"])),
        ("NEU", "small + 2.0x synth", load_metrics(NEU_RUNS["synth_double"])),
        ("Magnetic Tile", "small", load_metrics(MAG_RUNS["small"])),
        ("Magnetic Tile", "medium", load_metrics(MAG_RUNS["medium"])),
        ("Magnetic Tile", "full", load_metrics(MAG_RUNS["full"])),
        ("Magnetic Tile", "small + 0.5x synth", load_metrics(MAG_RUNS["synth_half"])),
        ("Magnetic Tile", "small + 1.0x synth", load_metrics(MAG_RUNS["synth"])),
        ("Magnetic Tile", "small + 2.0x synth", load_metrics(MAG_RUNS["synth_double"])),
    ]

    csv_path = REPORTS_DIR / "final_results_table.csv"
    md_path = REPORTS_DIR / "final_results_table.md"
    ensure_dir(REPORTS_DIR)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "regime", "accuracy", "precision", "recall", "f1", "test_loss"],
        )
        writer.writeheader()
        for dataset, regime, metrics in rows:
            writer.writerow(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "accuracy": f"{metrics['accuracy']:.6f}",
                    "precision": f"{metrics['precision']:.6f}",
                    "recall": f"{metrics['recall']:.6f}",
                    "f1": f"{metrics['f1']:.6f}",
                    "test_loss": f"{metrics['test_loss']:.6f}",
                }
            )

    neu_small = load_metrics(NEU_RUNS["small"])["f1"]
    mag_small = load_metrics(MAG_RUNS["small"])["f1"]
    lines = [
        "# Final Results Table",
        "",
        "| Dataset | Regime | Accuracy | Precision | Recall | F1 | Test loss | ΔF1 vs small |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, regime, metrics in rows:
        base = neu_small if dataset == "NEU" else mag_small
        delta = metrics["f1"] - base
        lines.append(
            f"| {dataset} | {regime} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['test_loss']:.4f} | {delta:+.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        canvas.paste(image, offset)
        return canvas


def _draw_caption(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, fill=FONT_COLOR)


def _build_pair_grid(
    pairs: list[tuple[str, Path, Path]],
    title: str,
    output_path: Path,
    cell_size: tuple[int, int] = (180, 180),
) -> None:
    margin = 18
    header_h = 42
    caption_h = 34
    cols = 2
    rows = len(pairs)
    width = margin * 3 + cols * cell_size[0]
    height = margin * (rows + 2) + header_h + rows * (cell_size[1] + caption_h)
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    _draw_caption(draw, margin, margin, title)

    for idx, (label, real_path, synth_path) in enumerate(pairs):
        top = margin + header_h + idx * (cell_size[1] + caption_h + margin)
        real_img = _fit_image(real_path, cell_size)
        synth_img = _fit_image(synth_path, cell_size)
        real_x = margin
        synth_x = margin * 2 + cell_size[0]
        canvas.paste(real_img, (real_x, top))
        canvas.paste(synth_img, (synth_x, top))
        draw.rectangle((real_x, top, real_x + cell_size[0], top + cell_size[1]), outline=LINE_COLOR)
        draw.rectangle((synth_x, top, synth_x + cell_size[0], top + cell_size[1]), outline=LINE_COLOR)
        _draw_caption(draw, real_x, top + cell_size[1] + 6, f"Real | {label}")
        _draw_caption(draw, synth_x, top + cell_size[1] + 6, f"Synthetic | {label}")

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def _select_examples(manifest_path: Path, per_label: int) -> list[tuple[str, Path, Path]]:
    payload = load_json(manifest_path)
    grouped: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for item in payload["generated"]:
        label = item["label"]
        real_path = Path(item["source_paths"][0])
        synth_path = Path(item["path"])
        grouped[label].append((label, real_path, synth_path))

    selected: list[tuple[str, Path, Path]] = []
    for label in sorted(grouped):
        selected.extend(grouped[label][:per_label])
    return selected


def build_synthetic_galleries() -> None:
    neu_pairs = _select_examples(NEU_SYNTH_MANIFEST, per_label=2)
    mag_pairs = _select_examples(MAG_SYNTH_MANIFEST, per_label=4)

    _build_pair_grid(
        neu_pairs,
        "NEU: Real vs Synthetic / Реальные и синтетические изображения",
        PLOTS_DIR / "01_neu_real_vs_synth.png",
    )
    _build_pair_grid(
        mag_pairs,
        "Magnetic Tile: Real vs Synthetic / Реальные и синтетические изображения",
        PLOTS_DIR / "02_magnetic_real_vs_synth.png",
    )

    combined = neu_pairs[:6] + mag_pairs[:6]
    _build_pair_grid(
        combined,
        "Combined Gallery / Сводная галерея real vs synthetic",
        PLOTS_DIR / "03_combined_real_vs_synth.png",
        cell_size=(170, 170),
    )


def main() -> None:
    write_summary_tables()
    build_synthetic_galleries()
    print(f"Saved summary tables to {REPORTS_DIR}")
    print(f"Saved synthetic galleries to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
