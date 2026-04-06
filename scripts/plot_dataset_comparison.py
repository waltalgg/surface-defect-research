from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from defect_lab.utils import ensure_dir


RUNS_DIR = Path("artifacts/runs")
PLOTS_DIR = Path("artifacts/plots_compare")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def metric_from_run(run_name: str, metric: str) -> float:
    payload = load_json(RUNS_DIR / run_name / "test_metrics.json")
    if metric == "test_loss":
        return float(payload["test_loss"])
    return float(payload["metrics"][metric])


def plot_train_size_comparison() -> None:
    x = [1, 2, 3]
    labels = ["Small\nМалый", "Medium\nСредний", "Full\nПолный"]
    neu_runs = ["neu_resnet18_small_seed_42", "neu_resnet18_medium", "neu_resnet18"]
    magnetic_runs = [
        "magnetic_tile_resnet18_binary_small_balanced_gpu",
        "magnetic_tile_resnet18_binary_medium_balanced",
        "magnetic_tile_resnet18_binary_full_balanced",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "f1", "Train-Size Scaling by F1 / Рост качества по размеру train"),
        (axes[1], "accuracy", "Train-Size Scaling by Accuracy / Рост accuracy по размеру train"),
    ]:
        neu_values = [metric_from_run(run_name, metric) for run_name in neu_runs]
        magnetic_values = [metric_from_run(run_name, metric) for run_name in magnetic_runs]
        ax.plot(x, neu_values, marker="o", linewidth=2, label="NEU / NEU")
        ax.plot(x, magnetic_values, marker="s", linewidth=2, label="Magnetic Tile / Magnetic Tile")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(f"{metric.upper()} / {metric.upper()}")
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, "01_train_size_comparison.png")


def plot_synthetic_ratio_comparison() -> None:
    ratios = [0.0, 0.5, 1.0, 2.0]
    neu_runs = [
        "neu_resnet18_small_seed_42",
        "neu_resnet18_small_synth_half",
        "neu_resnet18_small_synth",
        "neu_resnet18_small_synth_double_seed_42",
    ]
    magnetic_runs = [
        "magnetic_tile_resnet18_binary_small_balanced_gpu",
        "magnetic_tile_resnet18_binary_small_synth_half_balanced",
        "magnetic_tile_resnet18_binary_small_synth_balanced",
        "magnetic_tile_resnet18_binary_small_synth_double_balanced",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "f1", "Synthetic Ratio vs F1 / Доля синтетики и F1"),
        (axes[1], "accuracy", "Synthetic Ratio vs Accuracy / Доля синтетики и accuracy"),
    ]:
        neu_values = [metric_from_run(run_name, metric) for run_name in neu_runs]
        magnetic_values = [metric_from_run(run_name, metric) for run_name in magnetic_runs]
        ax.plot(ratios, neu_values, marker="o", linewidth=2, label="NEU / NEU")
        ax.plot(ratios, magnetic_values, marker="s", linewidth=2, label="Magnetic Tile / Magnetic Tile")
        ax.set_xticks(ratios)
        ax.set_title(title)
        ax.set_xlabel("Synthetic ratio / Доля синтетики")
        ax.set_ylabel(f"{metric.upper()} / {metric.upper()}")
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, "02_synthetic_ratio_comparison.png")


def plot_relative_improvement() -> None:
    data = {
        "NEU": {
            "small": metric_from_run("neu_resnet18_small_seed_42", "f1"),
            "best_synth": metric_from_run("neu_resnet18_small_synth_double_gpu", "f1"),
            "full": metric_from_run("neu_resnet18", "f1"),
        },
        "Magnetic Tile": {
            "small": metric_from_run("magnetic_tile_resnet18_binary_small_balanced_gpu", "f1"),
            "best_synth": metric_from_run("magnetic_tile_resnet18_binary_small_synth_double_balanced", "f1"),
            "full": metric_from_run("magnetic_tile_resnet18_binary_full_balanced", "f1"),
        },
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(data.keys())
    base = [data[label]["small"] for label in labels]
    synth_gain = [data[label]["best_synth"] - data[label]["small"] for label in labels]
    full_gain = [data[label]["full"] - data[label]["small"] for label in labels]

    x = range(len(labels))
    ax.bar(x, base, label="Small baseline / Малый baseline")
    ax.bar(x, synth_gain, bottom=base, label="Gain from synthetic / Прирост от синтетики")
    ax.bar(x, full_gain, bottom=base, alpha=0.45, label="Gain to full regime / Прирост к full")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 / F1")
    ax.set_title("Relative Gains by Dataset / Относительный прирост по датасетам")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, "03_relative_gains.png")


def plot_best_run_histories() -> None:
    runs = {
        "NEU best synth / NEU лучшая синтетика": "neu_resnet18_small_synth_double_gpu",
        "Magnetic best small / Magnetic лучший малый": "magnetic_tile_resnet18_binary_small_balanced_gpu_100",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, run_name in runs.items():
        history = load_json(RUNS_DIR / run_name / "history.json")
        epochs = [item["epoch"] for item in history["epochs"]]
        train_f1 = [item["train_metrics"]["f1"] for item in history["epochs"]]
        val_f1 = [item["val_metrics"]["f1"] for item in history["epochs"]]
        train_acc = [item["train_metrics"]["accuracy"] for item in history["epochs"]]
        val_acc = [item["val_metrics"]["accuracy"] for item in history["epochs"]]

        axes[0].plot(epochs, train_f1, linestyle="--", linewidth=1.7, label=f"{label} train")
        axes[0].plot(epochs, val_f1, linewidth=2, label=f"{label} val")
        axes[1].plot(epochs, train_acc, linestyle="--", linewidth=1.7, label=f"{label} train")
        axes[1].plot(epochs, val_acc, linewidth=2, label=f"{label} val")

    axes[0].set_title("F1 Dynamics / Динамика F1")
    axes[0].set_xlabel("Epoch / Эпоха")
    axes[0].set_ylabel("F1 / F1")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Accuracy Dynamics / Динамика accuracy")
    axes[1].set_xlabel("Epoch / Эпоха")
    axes[1].set_ylabel("Accuracy / Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    save_figure(fig, "04_best_run_histories.png")


def plot_summary_table_chart() -> None:
    labels = [
        "NEU small",
        "NEU full",
        "NEU small + synth 2x",
        "Mag small",
        "Mag full",
        "Mag small + synth 2x",
    ]
    runs = [
        "neu_resnet18_small_seed_42",
        "neu_resnet18",
        "neu_resnet18_small_synth_double_gpu",
        "magnetic_tile_resnet18_binary_small_balanced_gpu",
        "magnetic_tile_resnet18_binary_full_balanced",
        "magnetic_tile_resnet18_binary_small_synth_double_balanced",
    ]
    f1_values = [metric_from_run(run_name, "f1") for run_name in runs]
    acc_values = [metric_from_run(run_name, "accuracy") for run_name in runs]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(labels)))
    width = 0.38
    ax.bar([item - width / 2 for item in x], f1_values, width=width, label="F1 / F1")
    ax.bar([item + width / 2 for item in x], acc_values, width=width, label="Accuracy / Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_title("Cross-Dataset Summary on ResNet18 / Сводное сравнение двух датасетов на ResNet18")
    ax.set_ylabel("Score / Значение")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, "05_cross_dataset_summary.png")


def main() -> None:
    set_style()
    ensure_dir(PLOTS_DIR)
    plot_train_size_comparison()
    plot_synthetic_ratio_comparison()
    plot_relative_improvement()
    plot_best_run_histories()
    plot_summary_table_chart()
    print(f"Saved comparison plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
