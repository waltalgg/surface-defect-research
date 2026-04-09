from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from defect_lab.utils import ensure_dir


RUNS_DIR = Path("artifacts/runs")
REPORTS_DIR = Path("artifacts/reports")
PLOTS_DIR = Path("artifacts/plots_compare")
SYNTH_PLOTS_DIR = Path("artifacts/plots_synthetic")
COMPOSITE_PLOTS_DIR = Path("artifacts/plots_synthetic_composite")

DATASETS = {
    "NEU": {
        "small": "neu_resnet18_small_gpu",
        "full": "neu_resnet18_full_gpu",
        "synth_half": "neu_resnet18_small_synth_half_gpu",
        "synth": "neu_resnet18_small_synth_gpu",
        "synth_double": "neu_resnet18_small_synth_double_gpu",
    },
    "Magnetic Tile": {
        "small": "magnetic_tile_resnet18_binary_small_balanced_gpu",
        "full": "magnetic_tile_resnet18_binary_full_balanced_gpu",
        "synth_half": "magnetic_tile_resnet18_binary_small_synth_half_balanced_gpu",
        "synth": "magnetic_tile_resnet18_binary_small_synth_balanced_gpu",
        "synth_double": "magnetic_tile_resnet18_binary_small_synth_double_balanced_gpu",
    },
    "PY-CrackDB": {
        "small": "py_crackdb_resnet18_binary_small_balanced_gpu",
        "full": "py_crackdb_resnet18_binary_full_balanced_gpu",
        "synth_half": "py_crackdb_resnet18_binary_small_synth_half_balanced_gpu",
        "synth": "py_crackdb_resnet18_binary_small_synth_balanced_gpu",
        "synth_double": "py_crackdb_resnet18_binary_small_synth_double_balanced_gpu",
    },
}

REGIME_LABELS = {
    "small": "small",
    "full": "full",
    "synth_half": "small + 0.5x synth",
    "synth": "small + 1.0x synth",
    "synth_double": "small + 2.0x synth",
}

PLOT_KEYS = {
    "train_size": "Compare small/full",
    "synthetic_ratio": "Compare synthetic ratios",
    "histories": "Plot best validation histories",
    "best_regimes": "Plot best regime bars",
    "synthetic_examples": "Generate real vs synthetic galleries",
    "composite_examples": "Generate composite galleries",
    "summary": "Write CSV and Markdown summaries",
}

SYNTHETIC_MANIFESTS = {
    "NEU": Path("data/processed/synthetic_neu_small_double_gpu/synthetic_manifest.json"),
    "Magnetic Tile": Path("data/processed/synthetic_magnetic_tile_binary_small_double_balanced_gpu/synthetic_manifest.json"),
    "PY-CrackDB": Path("data/processed/synthetic_py_crackdb_binary_small_double_balanced_gpu/synthetic_manifest.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latest plots and summaries.")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=list(PLOT_KEYS.keys()),
        help="Keys to generate. Example: --plots train_size synthetic_ratio",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all available plots and summaries.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metrics(run_name: str) -> dict:
    payload = load_json(RUNS_DIR / run_name / "test_metrics.json")
    return payload["metrics"] | {"test_loss": payload["test_loss"]}


def load_history(run_name: str) -> dict:
    return load_json(RUNS_DIR / run_name / "history.json")


def set_style() -> None:
    plt.style.use("tableau-colorblind10")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["axes.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 9


def save_figure(fig, name: str) -> None:
    ensure_dir(PLOTS_DIR)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        canvas.paste(image, offset)
        return canvas


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def collect_rows() -> list[dict]:
    rows = []
    for dataset_name, runs in DATASETS.items():
        for regime_key, run_name in runs.items():
            metrics = load_metrics(run_name)
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime_key": regime_key,
                    "regime": REGIME_LABELS[regime_key],
                    "run_name": run_name,
                    **metrics,
                }
            )
    return rows


def plot_train_size_comparison(rows: list[dict]) -> None:
    labels = ["small", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "f1", "Train Size vs F1 / Размер train и F1"),
        (axes[1], "accuracy", "Train Size vs Accuracy / Размер train и accuracy"),
    ]:
        for dataset_name in DATASETS:
            values = [
                next(row[metric] for row in rows if row["dataset"] == dataset_name and row["regime_key"] == label)
                for label in labels
            ]
            ax.plot(labels, values, marker="o", linewidth=2, label=dataset_name)
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, "01_train_size_comparison.png")


def plot_synthetic_comparison(rows: list[dict]) -> None:
    labels = ["small", "synth_half", "synth", "synth_double"]
    display_labels = ["real only", "0.5x synth", "1.0x synth", "2.0x synth"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "f1", "Synthetic Ratio vs F1 / Доля синтетики и F1"),
        (axes[1], "accuracy", "Synthetic Ratio vs Accuracy / Доля синтетики и accuracy"),
    ]:
        for dataset_name in DATASETS:
            values = [
                next(row[metric] for row in rows if row["dataset"] == dataset_name and row["regime_key"] == label)
                for label in labels
            ]
            ax.plot(display_labels, values, marker="o", linewidth=2, label=dataset_name)
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, "02_synthetic_ratio_comparison.png")


def plot_best_histories(rows: list[dict]) -> None:
    selected = {}
    for dataset_name in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        best_row = max(dataset_rows, key=lambda row: row["f1"])
        selected[f"{dataset_name}: {best_row['regime']}"] = best_row["run_name"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, run_name in selected.items():
        history = load_history(run_name)
        epochs = [item["epoch"] for item in history["epochs"]]
        val_f1 = [item["val_metrics"]["f1"] for item in history["epochs"]]
        val_acc = [item["val_metrics"]["accuracy"] for item in history["epochs"]]
        axes[0].plot(epochs, val_f1, linewidth=2, label=label)
        axes[1].plot(epochs, val_acc, linewidth=2, label=label)

    axes[0].set_title("Validation F1 by Epoch / F1 по эпохам")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("F1")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Accuracy by Epoch / Accuracy по эпохам")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    save_figure(fig, "03_validation_histories.png")


def plot_best_regime_bars(rows: list[dict]) -> None:
    best_rows = []
    for dataset_name in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        best_rows.append(max(dataset_rows, key=lambda row: row["f1"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(len(best_rows))
    f1_values = [row["f1"] for row in best_rows]
    ax.bar(positions, f1_values, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xticks(list(positions), [f"{row['dataset']}\n{row['regime']}" for row in best_rows])
    ax.set_ylabel("F1")
    ax.set_title("Best Regime per Dataset / Лучший режим для каждого датасета")
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(f1_values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    save_figure(fig, "04_best_regime_bars.png")


def select_examples(manifest_path: Path, per_label: int) -> list[tuple[str, Path, Path]]:
    payload = load_json(manifest_path)
    grouped: dict[str, list[tuple[str, Path, Path]]] = {}
    for item in payload["generated"]:
        label = item["label"]
        grouped.setdefault(label, [])
        grouped[label].append((label, Path(item["source_paths"][0]), Path(item["path"])))

    selected: list[tuple[str, Path, Path]] = []
    for label in sorted(grouped):
        selected.extend(grouped[label][:per_label])
    return selected


def build_pair_grid(
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
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(20, 20, 20))

    for idx, (label, real_path, synth_path) in enumerate(pairs):
        top = margin + header_h + idx * (cell_size[1] + caption_h + margin)
        real_img = fit_image(real_path, cell_size)
        synth_img = fit_image(synth_path, cell_size)
        real_x = margin
        synth_x = margin * 2 + cell_size[0]
        canvas.paste(real_img, (real_x, top))
        canvas.paste(synth_img, (synth_x, top))
        draw.rectangle((real_x, top, real_x + cell_size[0], top + cell_size[1]), outline=(220, 220, 220))
        draw.rectangle((synth_x, top, synth_x + cell_size[0], top + cell_size[1]), outline=(220, 220, 220))
        draw.text((real_x, top + cell_size[1] + 6), f"Real | {label}", fill=(20, 20, 20))
        draw.text((synth_x, top + cell_size[1] + 6), f"Synthetic | {label}", fill=(20, 20, 20))

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def build_composite_gallery(manifest_path: Path, output_dir: Path) -> None:
    payload = load_json(manifest_path)
    grouped: dict[str, list[dict]] = {}
    for item in payload["generated"]:
        grouped.setdefault(item["label"], []).append(item)

    def _save_gallery(items: list[dict], title: str, output_name: str, per_row: int = 3) -> None:
        margin = 16
        title_h = 30
        caption_h = 34
        cell_w, cell_h = 170, 170
        rows = (len(items) + per_row - 1) // per_row
        width = margin * (per_row + 1) + per_row * (cell_w * 2)
        height = margin * (rows + 2) + title_h + rows * (cell_h + caption_h)
        canvas = Image.new("RGB", (width, height), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, margin), title, fill=(20, 20, 20))

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
            draw.rectangle((left, top, left + cell_w, top + cell_h), outline=(220, 220, 220))
            draw.rectangle((left + cell_w, top, left + cell_w * 2, top + cell_h), outline=(220, 220, 220))
            draw.text((left, top + cell_h + 4), "Real", fill=(20, 20, 20))
            draw.text((left + cell_w, top + cell_h + 4), "Composite", fill=(20, 20, 20))

        ensure_dir(output_dir)
        canvas.save(output_dir / output_name)

    defect_items = grouped.get("defect", [])[:9]
    no_defect_items = grouped.get("no_defect", [])[:6]
    mixed_items = defect_items[:6] + no_defect_items[:3]

    _save_gallery(defect_items, "Composite Defect Examples / Примеры composite defect", "01_defect_gallery.png")
    _save_gallery(no_defect_items, "Composite No-Defect Examples / Примеры composite no_defect", "02_no_defect_gallery.png")
    _save_gallery(mixed_items, "Composite Mixed Gallery / Смешанная галерея composite", "03_mixed_gallery.png")


def generate_synthetic_examples() -> None:
    ensure_dir(SYNTH_PLOTS_DIR)
    dataset_pairs = []
    for dataset_name, manifest_path in SYNTHETIC_MANIFESTS.items():
        if not manifest_path.exists():
            continue
        dataset_pairs.append((dataset_name, select_examples(manifest_path, per_label=2)))

    for idx, (dataset_name, pairs) in enumerate(dataset_pairs, start=1):
        safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        build_pair_grid(
            pairs,
            f"{dataset_name}: Real vs Synthetic",
            SYNTH_PLOTS_DIR / f"{idx:02d}_{safe_name}_real_vs_synth.png",
        )

    combined = []
    for _, pairs in dataset_pairs:
        combined.extend(pairs[:4])
    if combined:
        build_pair_grid(
            combined,
            "Combined Gallery / Сводная галерея real vs synthetic",
            SYNTH_PLOTS_DIR / "10_combined_real_vs_synth.png",
            cell_size=(170, 170),
        )


def generate_composite_examples() -> None:
    manifest_path = Path("data/processed/synthetic_magnetic_tile_binary_small_composite_gpu/synthetic_manifest.json")
    if manifest_path.exists():
        build_composite_gallery(manifest_path, COMPOSITE_PLOTS_DIR)


def write_summary_files(rows: list[dict]) -> None:
    ensure_dir(REPORTS_DIR)
    rows_sorted = sorted(rows, key=lambda row: (row["dataset"], row["regime_key"]))

    with (REPORTS_DIR / "current_comparison_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "regime", "accuracy", "precision", "recall", "f1", "test_loss", "run_name"],
        )
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "regime": row["regime"],
                    "accuracy": f"{row['accuracy']:.6f}",
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                    "test_loss": f"{row['test_loss']:.6f}",
                    "run_name": row["run_name"],
                }
            )

    lines = [
        "# Current Comparison Summary",
        "",
        "## Results Table",
        "",
        "| Dataset | Regime | Accuracy | Precision | Recall | F1 | Test loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_sorted:
        lines.append(
            f"| {row['dataset']} | {row['regime']} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['test_loss']:.4f} |"
        )

    lines += ["", "## Short Notes", ""]
    for dataset_name in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        best_row = max(dataset_rows, key=lambda row: row["f1"])
        baseline_small = next(row for row in dataset_rows if row["regime_key"] == "small")
        lines.append(
            f"- `{dataset_name}`: baseline `small` gives `F1 = {baseline_small['f1']:.4f}`, "
            f"best current regime is `{best_row['regime']}` with `F1 = {best_row['f1']:.4f}`."
        )
    (REPORTS_DIR / "current_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_requested_keys(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(PLOT_KEYS.keys())
    if args.plots:
        return args.plots
    return list(PLOT_KEYS.keys())


def main() -> None:
    args = parse_args()
    requested_keys = resolve_requested_keys(args)
    set_style()
    rows = collect_rows()

    actions = {
        "train_size": lambda: plot_train_size_comparison(rows),
        "synthetic_ratio": lambda: plot_synthetic_comparison(rows),
        "histories": lambda: plot_best_histories(rows),
        "best_regimes": lambda: plot_best_regime_bars(rows),
        "synthetic_examples": generate_synthetic_examples,
        "composite_examples": generate_composite_examples,
        "summary": lambda: write_summary_files(rows),
    }

    for key in requested_keys:
        actions[key]()
        print(f"Generated: {key}")

    print(f"Plots directory: {PLOTS_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
