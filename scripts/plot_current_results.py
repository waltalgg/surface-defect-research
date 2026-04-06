from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from defect_lab.utils import ensure_dir


RUNS_DIR = Path("artifacts/runs")
REPORTS_DIR = Path("artifacts/reports")
PLOTS_DIR = Path("artifacts/plots_compare")


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


def plot_train_size_comparison() -> None:
    labels = ["Small\nМалый", "Medium\nСредний", "Full\nПолный"]
    neu_keys = ["small", "medium", "full"]
    mag_keys = ["small", "medium", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title in [
        (axes[0], "f1", "Train Size vs F1 / Размер train и F1"),
        (axes[1], "accuracy", "Train Size vs Accuracy / Размер train и accuracy"),
    ]:
        neu_values = [load_metrics(NEU_RUNS[key])[metric] for key in neu_keys]
        mag_values = [load_metrics(MAG_RUNS[key])[metric] for key in mag_keys]
        ax.plot(labels, neu_values, marker="o", linewidth=2, label="NEU")
        ax.plot(labels, mag_values, marker="s", linewidth=2, label="Magnetic Tile")
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()
        for idx, value in enumerate(neu_values):
            ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
        for idx, value in enumerate(mag_values):
            ax.text(idx, value - 0.04, f"{value:.3f}", ha="center", fontsize=8)

    save_figure(fig, "01_train_size_comparison.png")


def plot_synthetic_comparison() -> None:
    labels = ["Real only\nТолько real", "0.5x synth", "1.0x synth", "2.0x synth"]
    neu_keys = ["small", "synth_half", "synth", "synth_double"]
    mag_keys = ["small", "synth_half", "synth", "synth_double"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title in [
        (axes[0], "f1", "Synthetic Ratio vs F1 / Доля синтетики и F1"),
        (axes[1], "accuracy", "Synthetic Ratio vs Accuracy / Доля синтетики и accuracy"),
    ]:
        neu_values = [load_metrics(NEU_RUNS[key])[metric] for key in neu_keys]
        mag_values = [load_metrics(MAG_RUNS[key])[metric] for key in mag_keys]
        ax.plot(labels, neu_values, marker="o", linewidth=2, label="NEU")
        ax.plot(labels, mag_values, marker="s", linewidth=2, label="Magnetic Tile")
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()

    save_figure(fig, "02_synthetic_ratio_comparison.png")


def plot_best_histories() -> None:
    selected = {
        "NEU full / NEU полный": NEU_RUNS["full"],
        "NEU small + 2x synth / NEU малый + 2x": NEU_RUNS["synth_double"],
        "Mag small + 2x synth / Magnetic малый + 2x": MAG_RUNS["synth_double"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, run_name in selected.items():
        history = load_history(run_name)
        epochs = [item["epoch"] for item in history["epochs"]]
        val_f1 = [item["val_metrics"]["f1"] for item in history["epochs"]]
        val_acc = [item["val_metrics"]["accuracy"] for item in history["epochs"]]
        axes[0].plot(epochs, val_f1, linewidth=2, label=label)
        axes[1].plot(epochs, val_acc, linewidth=2, label=label)

    axes[0].set_title("Validation F1 by Epoch / F1 на валидации по эпохам")
    axes[0].set_xlabel("Epoch / Эпоха")
    axes[0].set_ylabel("F1")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Accuracy by Epoch / Accuracy на валидации по эпохам")
    axes[1].set_xlabel("Epoch / Эпоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    save_figure(fig, "03_validation_histories.png")


def plot_confusion_matrices() -> None:
    selected = {
        "NEU full / NEU полный": NEU_RUNS["full"],
        "Mag small / Magnetic малый": MAG_RUNS["small"],
        "Mag small + 2x / Magnetic малый + 2x": MAG_RUNS["synth_double"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    image = None
    for ax, (title, run_name) in zip(axes, selected.items(), strict=False):
        matrix = load_metrics(run_name)["confusion_matrix"]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted / Предсказано")
        ax.set_ylabel("True / Истина")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    save_figure(fig, "04_confusion_matrices.png")


def write_summary_markdown() -> None:
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

    lines = [
        "# Current Comparison Summary",
        "",
        "## Results Table",
        "",
        "| Dataset | Regime | Accuracy | Precision | Recall | F1 | Test loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, regime, metrics in rows:
        lines.append(
            f"| {dataset} | {regime} | "
            f"{metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['test_loss']:.4f} |"
        )

    lines += [
        "",
        "## Short Notes",
        "",
        f"- `NEU`: лучший режим сейчас `full` с `F1 = {load_metrics(NEU_RUNS['full'])['f1']:.4f}`.",
        f"- `NEU`: лучший synthetic-режим `small + 2.0x synth` с `F1 = {load_metrics(NEU_RUNS['synth_double'])['f1']:.4f}`.",
        f"- `Magnetic Tile`: baseline `small` даёт `F1 = {load_metrics(MAG_RUNS['small'])['f1']:.4f}`.",
        f"- `Magnetic Tile`: лучший synthetic-режим сейчас `small + 1.0x synth` с `F1 = {load_metrics(MAG_RUNS['synth'])['f1']:.4f}`.",
        f"- `Magnetic Tile`: `small + 2.0x synth` близок к нему и даёт `F1 = {load_metrics(MAG_RUNS['synth_double'])['f1']:.4f}`.",
        "",
    ]

    ensure_dir(REPORTS_DIR)
    (REPORTS_DIR / "current_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")

    csv_rows = [
        {
            "dataset": dataset,
            "regime": regime,
            "accuracy": f"{metrics['accuracy']:.6f}",
            "precision": f"{metrics['precision']:.6f}",
            "recall": f"{metrics['recall']:.6f}",
            "f1": f"{metrics['f1']:.6f}",
            "test_loss": f"{metrics['test_loss']:.6f}",
        }
        for dataset, regime, metrics in rows
    ]
    with (REPORTS_DIR / "current_comparison_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "regime", "accuracy", "precision", "recall", "f1", "test_loss"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    set_style()
    ensure_dir(PLOTS_DIR)
    ensure_dir(REPORTS_DIR)
    plot_train_size_comparison()
    plot_synthetic_comparison()
    plot_best_histories()
    plot_confusion_matrices()
    write_summary_markdown()
    print(f"Saved plots to {PLOTS_DIR}")
    print(f"Saved summaries to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
