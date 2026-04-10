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
        "synth_half": "neu_resnet18_small_synth_half_segmentation_gpu",
        "synth": "neu_resnet18_small_synth_segmentation_gpu",
        "synth_double": "neu_resnet18_small_synth_double_segmentation_gpu",
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
    "all_results_overview": "Plot unified overview from all_results_table",
    "epoch_comparison": "Plot retained 30 vs 100 epoch comparison",
    "synthetic_examples": "Generate real vs synthetic galleries",
    "composite_examples": "Generate composite galleries",
    "summary": "Write CSV and Markdown summaries",
}

SYNTHETIC_MANIFESTS = {
    "NEU": Path("data/processed/synthetic_neu_small_double_segmentation_gpu/synthetic_manifest.json"),
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


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def collect_rows() -> list[dict]:
    rows = []
    for dataset_name, runs in DATASETS.items():
        for regime_key, run_name in runs.items():
            metrics_path = RUNS_DIR / run_name / "test_metrics.json"
            if not metrics_path.exists():
                continue
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


def shorten_experiment(name: str) -> str:
    value = name.replace("neu_resnet18_", "neu_")
    value = value.replace("py_crackdb_resnet18_binary_", "py_")
    value = value.replace("_balanced_gpu", "")
    value = value.replace("_segmentation_gpu", "_seg")
    value = value.replace("_gpu", "")
    value = value.replace("small_synth_half", "small+0.5x")
    value = value.replace("small_synth_double", "small+2.0x")
    value = value.replace("small_synth", "small+1.0x")
    return value


def plot_train_size_comparison(rows: list[dict]) -> None:
    labels = ["small", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "f1", "Train Size vs F1 / Размер train и F1"),
        (axes[1], "accuracy", "Train Size vs Accuracy / Размер train и accuracy"),
    ]:
        for dataset_name in DATASETS:
            if not all(any(row["dataset"] == dataset_name and row["regime_key"] == label for row in rows) for label in labels):
                continue
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
            if not all(any(row["dataset"] == dataset_name and row["regime_key"] == label for row in rows) for label in labels):
                continue
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
        if not dataset_rows:
            continue
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
        if not dataset_rows:
            continue
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


def plot_all_results_overview() -> None:
    table_path = REPORTS_DIR / "all_results_table.csv"
    if not table_path.exists():
        return

    rows = load_csv_rows(table_path)
    classification_rows = [row for row in rows if row["task_type"] == "classification"]
    segmentation_rows = [row for row in rows if row["task_type"] == "segmentation"]
    generation_rows = [row for row in rows if row["task_type"] == "generation"]

    classification_rows.sort(key=lambda row: (row["dataset_name"], float(row["primary_metric_value"])))
    segmentation_rows.sort(key=lambda row: row["dataset_name"])
    generation_rows.sort(key=lambda row: row["dataset_name"])

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), height_ratios=[2.4, 1.1, 0.8])

    ax = axes[0]
    colors = []
    labels = []
    values = []
    for row in classification_rows:
        labels.append(f"{row['dataset_name']} | {shorten_experiment(row['experiment'])}")
        values.append(float(row["primary_metric_value"]))
        colors.append("#4c72b0" if row["dataset_name"] == "neu_steel" else "#dd8452")

    positions = range(len(classification_rows))
    ax.barh(list(positions), values, color=colors)
    ax.set_yticks(list(positions), labels)
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("F1")
    ax.set_title("Classification Overview / Classification")
    ax.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(value + 0.001, idx, f"{value:.4f}", va="center", fontsize=8)

    ax = axes[1]
    if segmentation_rows:
        seg_positions = range(len(segmentation_rows))
        dice_values = [float(row["primary_metric_value"]) for row in segmentation_rows]
        iou_values = [float(row["secondary_metric_value"]) for row in segmentation_rows]
        width = 0.35
        ax.bar([pos - width / 2 for pos in seg_positions], dice_values, width=width, label="Dice")
        ax.bar([pos + width / 2 for pos in seg_positions], iou_values, width=width, label="IoU")
        ax.set_xticks(list(seg_positions), [row["dataset_name"] for row in segmentation_rows])
        ax.set_ylim(0.0, 0.7)
        ax.set_ylabel("Score")
        ax.set_title("Segmentation Overview / Segmentation")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No segmentation rows", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2]
    if generation_rows:
        gen_positions = range(len(generation_rows))
        l1_values = [float(row["primary_metric_value"]) for row in generation_rows]
        ax.bar(list(gen_positions), l1_values, color="#55a868")
        ax.set_xticks(list(gen_positions), [row["dataset_name"] for row in generation_rows])
        ax.set_ylabel("L1")
        ax.set_title("Generation Overview / Generation")
        ax.grid(axis="y", alpha=0.3)
        for idx, value in enumerate(l1_values):
            ax.text(idx, value + 0.003, f"{value:.4f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No generation rows", ha="center", va="center")
        ax.set_axis_off()

    save_figure(fig, "05_all_results_overview.png")


def collect_epoch_pairs() -> list[dict]:
    table_path = REPORTS_DIR / "all_results_table.csv"
    if not table_path.exists():
        return []
    rows = load_csv_rows(table_path)
    relevant = [row for row in rows if row["task_type"] == "classification" and row["dataset_name"] == "neu_steel"]
    mapping_30 = {
        "0.5x": "neu_resnet18_small_synth_half_gpu",
        "1.0x": "neu_resnet18_small_synth_gpu",
        "2.0x": "neu_resnet18_small_synth_double_gpu",
    }
    mapping_100 = {
        "0.5x": "neu_resnet18_small_synth_half_segmentation_gpu",
        "1.0x": "neu_resnet18_small_synth_segmentation_gpu",
        "2.0x": "neu_resnet18_small_synth_double_segmentation_gpu",
    }
    by_name = {row["experiment"]: row for row in relevant}
    pairs = []
    for label in ["0.5x", "1.0x", "2.0x"]:
        row_30 = by_name.get(mapping_30[label])
        row_100 = by_name.get(mapping_100[label])
        if row_30 and row_100:
            pairs.append(
                {
                    "regime": label,
                    "f1_30": float(row_30["primary_metric_value"]),
                    "f1_100": float(row_100["primary_metric_value"]),
                    "acc_30": float(row_30["secondary_metric_value"]),
                    "acc_100": float(row_100["secondary_metric_value"]),
                }
            )
    return pairs


def plot_epoch_comparison() -> None:
    pairs = collect_epoch_pairs()
    if not pairs:
        return

    labels = [pair["regime"] for pair in pairs]
    x = range(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    f1_30 = [pair["f1_30"] for pair in pairs]
    f1_100 = [pair["f1_100"] for pair in pairs]
    axes[0].bar([idx - width / 2 for idx in x], f1_30, width=width, label="30 epochs")
    axes[0].bar([idx + width / 2 for idx in x], f1_100, width=width, label="100 epochs")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylim(0.94, 0.98)
    axes[0].set_ylabel("F1")
    axes[0].set_title("NEU Synthetic: 30 vs 100 Epochs / F1")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    acc_30 = [pair["acc_30"] for pair in pairs]
    acc_100 = [pair["acc_100"] for pair in pairs]
    axes[1].bar([idx - width / 2 for idx in x], acc_30, width=width, label="30 epochs")
    axes[1].bar([idx + width / 2 for idx in x], acc_100, width=width, label="100 epochs")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylim(0.94, 0.98)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("NEU Synthetic: 30 vs 100 Epochs / Accuracy")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "Retained epoch comparison\nOnly NEU synthetic pairs are directly available in the current workspace; "
        "100-epoch runs use the current mask-guided NEU pipeline.",
        fontsize=11,
    )
    save_figure(fig, "06_epoch_30_vs_100.png")

    lines = [
        "# Epoch 30 vs 100",
        "",
        "Only retained directly comparable local pairs are included here.",
        "For `NEU`, the 100-epoch line uses the current mask-guided synthetic pipeline.",
        "",
        "| Regime | F1 @ 30 | F1 @ 100 | ΔF1 | Acc @ 30 | Acc @ 100 | ΔAcc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in pairs:
        delta_f1 = pair["f1_100"] - pair["f1_30"]
        delta_acc = pair["acc_100"] - pair["acc_30"]
        lines.append(
            f"| {pair['regime']} | {pair['f1_30']:.4f} | {pair['f1_100']:.4f} | {delta_f1:+.4f} | "
            f"{pair['acc_30']:.4f} | {pair['acc_100']:.4f} | {delta_acc:+.4f} |"
        )
    (REPORTS_DIR / "epoch_30_vs_100_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    manifest_path = Path("data/processed/synthetic_py_crackdb_binary_small_balanced_gpu/synthetic_manifest.json")
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
        "all_results_overview": plot_all_results_overview,
        "epoch_comparison": plot_epoch_comparison,
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
