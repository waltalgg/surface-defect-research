import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from defect_lab.utils import ensure_dir


PLOTS_DIR = Path("artifacts/plots")
RUNS_DIR = Path("artifacts/runs")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_figure(fig, name: str) -> None:
    ensure_dir(PLOTS_DIR)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.style.use("tableau-colorblind10")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["axes.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 9


def plot_training_curves() -> None:
    experiments = {
        "Small / Малый": "neu_resnet18_small",
        "Medium / Средний": "neu_resnet18_medium",
        "Full / Полный": "neu_resnet18",
        "Small + Synth 2.0x / Малый + синтетика 2.0x": "neu_resnet18_small_synth_double",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, run_name in experiments.items():
        history = load_json(RUNS_DIR / run_name / "history.json")
        epochs = [item["epoch"] for item in history["epochs"]]
        train_loss = [item["train_loss"] for item in history["epochs"]]
        val_loss = [item["val_loss"] for item in history["epochs"]]
        val_f1 = [item["val_metrics"]["f1"] for item in history["epochs"]]
        axes[0].plot(epochs, train_loss, marker="o", linestyle="--", alpha=0.65, label=f"{label} train")
        axes[0].plot(epochs, val_loss, marker="o", label=f"{label} val")
        axes[1].plot(epochs, val_f1, marker="o", label=label)

    axes[0].set_title("Loss vs Epochs / Потери по эпохам")
    axes[0].set_xlabel("Epoch / Эпоха")
    axes[0].set_ylabel("Loss / Потери")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2)

    axes[1].set_title("Validation F1 vs Epochs / F1 на валидации")
    axes[1].set_xlabel("Epoch / Эпоха")
    axes[1].set_ylabel("F1-score / F1-мера")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    save_figure(fig, "01_training_curves_overview.png")


def plot_small_vs_synth_history() -> None:
    experiments = {
        "Real only / Только реальные": "neu_resnet18_small",
        "Synth 0.5x / Синтетика 0.5x": "neu_resnet18_small_synth_half",
        "Synth 1.0x / Синтетика 1.0x": "neu_resnet18_small_synth",
        "Synth 2.0x / Синтетика 2.0x": "neu_resnet18_small_synth_double",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    for label, run_name in experiments.items():
        history = load_json(RUNS_DIR / run_name / "history.json")
        epochs = [item["epoch"] for item in history["epochs"]]
        val_f1 = [item["val_metrics"]["f1"] for item in history["epochs"]]
        val_acc = [item["val_metrics"]["accuracy"] for item in history["epochs"]]
        axes[0].plot(epochs, val_f1, marker="o", label=label)
        axes[1].plot(epochs, val_acc, marker="o", label=label)

    axes[0].set_title("Small Regime: Validation F1 / Малый режим: F1 на валидации")
    axes[0].set_xlabel("Epoch / Эпоха")
    axes[0].set_ylabel("F1-score / F1-мера")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Small Regime: Validation Accuracy / Малый режим: Accuracy на валидации")
    axes[1].set_xlabel("Epoch / Эпоха")
    axes[1].set_ylabel("Accuracy / Точность")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    save_figure(fig, "02_small_regime_validation_curves.png")


def plot_metric_bars() -> None:
    summary = load_json(RUNS_DIR / "synthetic_ratio_summary.json")
    labels_map = {
        "neu_resnet18_small": "Small\nМалый",
        "neu_resnet18_small_synth_half": "Small+0.5x\nМалый+0.5x",
        "neu_resnet18_small_synth": "Small+1.0x\nМалый+1.0x",
        "neu_resnet18_small_synth_double": "Small+2.0x\nМалый+2.0x",
        "neu_resnet18_medium": "Medium\nСредний",
        "neu_resnet18": "Full\nПолный",
    }
    ordered = list(labels_map)
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics, strict=False):
        xs = range(len(ordered))
        ys = [summary[name][metric] for name in ordered]
        ax.bar(xs, ys)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([labels_map[name] for name in ordered], rotation=20)
        ax.set_ylim(0.8 if metric != "precision" else 0.85, 1.0)
        ax.set_title(f"{metric.upper()} / {metric.upper()}")
        ax.set_ylabel("Score / Значение")
        ax.grid(axis="y", alpha=0.3)
        for x, y in zip(xs, ys, strict=False):
            ax.text(x, y + 0.003, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "03_metric_bars_all_regimes.png")


def plot_synthetic_ratio_lines() -> None:
    summary = load_json(RUNS_DIR / "synthetic_ratio_summary.json")
    ratios = [0.0, 0.5, 1.0, 2.0]
    runs = [
        "neu_resnet18_small",
        "neu_resnet18_small_synth_half",
        "neu_resnet18_small_synth",
        "neu_resnet18_small_synth_double",
    ]
    f1 = [summary[name]["f1"] for name in runs]
    acc = [summary[name]["accuracy"] for name in runs]
    loss = [summary[name]["test_loss"] for name in runs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    series = [
        ("F1 by Synthetic Ratio / F1 по доле синтетики", f1, "F1-score / F1-мера"),
        ("Accuracy by Synthetic Ratio / Accuracy по доле синтетики", acc, "Accuracy / Точность"),
        ("Test Loss by Synthetic Ratio / Test loss по доле синтетики", loss, "Loss / Потери"),
    ]
    for ax, (title, values, ylabel) in zip(axes, series, strict=False):
        ax.plot(ratios, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Synthetic ratio / Доля синтетики")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        for x, y in zip(ratios, values, strict=False):
            ax.text(x, y + (0.004 if "Loss" not in title else 0.01), f"{y:.3f}", ha="center", fontsize=8)
    save_figure(fig, "04_synthetic_ratio_lines.png")


def plot_data_regime_scaling() -> None:
    summary = load_json(RUNS_DIR / "synthetic_ratio_summary.json")
    train_sizes = [240, 600, 1206]
    f1 = [
        summary["neu_resnet18_small"]["f1"],
        summary["neu_resnet18_medium"]["f1"],
        summary["neu_resnet18"]["f1"],
    ]
    acc = [
        summary["neu_resnet18_small"]["accuracy"],
        summary["neu_resnet18_medium"]["accuracy"],
        summary["neu_resnet18"]["accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, f1, marker="o", linewidth=2, label="F1-score / F1-мера")
    ax.plot(train_sizes, acc, marker="s", linewidth=2, label="Accuracy / Точность")
    ax.set_title("Performance vs Train Set Size / Качество vs размер train")
    ax.set_xlabel("Training samples / Обучающие примеры")
    ax.set_ylabel("Score / Значение")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "05_train_size_scaling.png")


def plot_confusion_matrices() -> None:
    selected = {
        "Small / Малый": "neu_resnet18_small",
        "Small + Synth 2.0x / Малый + синтетика 2.0x": "neu_resnet18_small_synth_double",
        "Full / Полный": "neu_resnet18",
    }
    labels = ["Cra", "Inc", "Pat", "Pit", "Rol", "Scr"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (title, run_name) in zip(axes, selected.items(), strict=False):
        metrics = load_json(RUNS_DIR / run_name / "test_metrics.json")
        matrix = metrics["metrics"]["confusion_matrix"]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Confusion Matrix\n{title}")
        ax.set_xlabel("Predicted / Предсказано")
        ax.set_ylabel("True / Истина")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    save_figure(fig, "06_confusion_matrices.png")


def plot_seed_scatter() -> None:
    small = load_json(RUNS_DIR / "repeated" / "neu_resnet_small_summary.json")
    synth = load_json(RUNS_DIR / "repeated" / "neu_resnet_small_synth_double_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["f1", "accuracy"], strict=False):
        seeds_small = [run["seed"] for run in small["runs"]]
        vals_small = [run[metric] for run in small["runs"]]
        seeds_synth = [run["seed"] for run in synth["runs"]]
        vals_synth = [run[metric] for run in synth["runs"]]
        ax.plot(seeds_small, vals_small, marker="o", linewidth=2, label="Small real only / Только реальные")
        ax.plot(seeds_synth, vals_synth, marker="s", linewidth=2, label="Small + Synth 2.0x / Малый + синтетика 2.0x")
        ax.set_title(f"{metric.upper()} by Seed / {metric.upper()} по seed")
        ax.set_xlabel("Seed / Начальное зерно")
        ax.set_ylabel("Score / Значение")
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, "07_seed_comparison_lines.png")


def plot_seed_ranges() -> None:
    small = load_json(RUNS_DIR / "repeated" / "neu_resnet_small_summary.json")
    synth = load_json(RUNS_DIR / "repeated" / "neu_resnet_small_synth_double_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["f1", "accuracy"], strict=False):
        labels = ["Small\nМалый", "Small+Synth2x\nМалый+Синт2x"]
        means = [small["aggregate"][metric]["mean"], synth["aggregate"][metric]["mean"]]
        mins = [small["aggregate"][metric]["min"], synth["aggregate"][metric]["min"]]
        maxs = [small["aggregate"][metric]["max"], synth["aggregate"][metric]["max"]]
        lower = [m - lo for m, lo in zip(means, mins, strict=False)]
        upper = [hi - m for m, hi in zip(means, maxs, strict=False)]
        ax.bar(range(2), means, yerr=[lower, upper], capsize=8)
        ax.set_xticks(range(2))
        ax.set_xticklabels(labels)
        ax.set_title(f"Mean ± Range for {metric.upper()} / Среднее и разброс {metric.upper()}")
        ax.set_ylabel("Score / Значение")
        ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "08_seed_range_bars.png")


def plot_generalization_gap() -> None:
    runs = {
        "Small / Малый": "neu_resnet18_small",
        "Medium / Средний": "neu_resnet18_medium",
        "Full / Полный": "neu_resnet18",
        "Small+Synth2x / Малый+Синт2x": "neu_resnet18_small_synth_double",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(runs))
    train_f1 = []
    val_f1 = []
    for run_name in runs.values():
        history = load_json(RUNS_DIR / run_name / "history.json")
        train_f1.append(history["epochs"][-1]["train_metrics"]["f1"])
        val_f1.append(max(epoch["val_metrics"]["f1"] for epoch in history["epochs"]))
    width = 0.36
    ax.bar([i - width / 2 for i in x], train_f1, width=width, label="Train F1 / Train F1")
    ax.bar([i + width / 2 for i in x], val_f1, width=width, label="Best Val F1 / Лучший Val F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(runs.keys()), rotation=15)
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Generalization Gap / Разрыв обобщения")
    ax.set_ylabel("F1-score / F1-мера")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, "09_generalization_gap.png")


def plot_f1_improvement_waterfall() -> None:
    summary = load_json(RUNS_DIR / "synthetic_ratio_summary.json")
    steps = [
        ("Small\nМалый", summary["neu_resnet18_small"]["f1"]),
        ("+0.5x", summary["neu_resnet18_small_synth_half"]["f1"]),
        ("+1.0x", summary["neu_resnet18_small_synth"]["f1"]),
        ("+2.0x", summary["neu_resnet18_small_synth_double"]["f1"]),
    ]
    base = steps[0][1]
    diffs = [0.0] + [value - base for _, value in steps[1:]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(steps)), diffs, bottom=base)
    ax.axhline(base, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([label for label, _ in steps])
    ax.set_title("F1 Improvement over Small Baseline / Прирост F1 относительно малого baseline")
    ax.set_ylabel("F1-score / F1-мера")
    ax.grid(axis="y", alpha=0.3)
    for idx, (_, value) in enumerate(steps):
        ax.text(idx, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    save_figure(fig, "10_f1_improvement_over_small.png")


def plot_metric_radar_like() -> None:
    summary = load_json(RUNS_DIR / "synthetic_ratio_summary.json")
    selected = {
        "Small / Малый": summary["neu_resnet18_small"],
        "Medium / Средний": summary["neu_resnet18_medium"],
        "Full / Полный": summary["neu_resnet18"],
        "Small+Synth2x / Малый+Синт2x": summary["neu_resnet18_small_synth_double"],
    }
    metrics = ["accuracy", "precision", "recall", "f1"]
    angles = [n / float(len(metrics)) * 2 * math.pi for n in range(len(metrics))]
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for label, values_dict in selected.items():
        values = [values_dict[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Accuracy\nТочность", "Precision\nТочность+", "Recall\nПолнота", "F1\nF1-мера"])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax.set_yticklabels(["0.85", "0.90", "0.95", "1.00"])
    ax.set_title("Metric Profile / Профиль метрик", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.15))
    save_figure(fig, "11_metric_profile_polar.png")


def plot_train_val_accuracy_curve() -> None:
    run_name = "neu_resnet18_small_synth_double"
    history = load_json(RUNS_DIR / run_name / "history.json")
    epochs = [item["epoch"] for item in history["epochs"]]
    train_acc = [item["train_metrics"]["accuracy"] for item in history["epochs"]]
    val_acc = [item["val_metrics"]["accuracy"] for item in history["epochs"]]
    train_f1 = [item["train_metrics"]["f1"] for item in history["epochs"]]
    val_f1 = [item["val_metrics"]["f1"] for item in history["epochs"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc, marker="o", label="Train Accuracy / Train точность")
    ax.plot(epochs, val_acc, marker="o", label="Val Accuracy / Val точность")
    ax.plot(epochs, train_f1, marker="s", linestyle="--", label="Train F1 / Train F1")
    ax.plot(epochs, val_f1, marker="s", linestyle="--", label="Val F1 / Val F1")
    ax.set_title("Best Synthetic Run Dynamics / Динамика лучшего синтетического запуска")
    ax.set_xlabel("Epoch / Эпоха")
    ax.set_ylabel("Score / Значение")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    save_figure(fig, "12_best_synth_train_val_dynamics.png")


def main() -> None:
    set_style()
    ensure_dir(PLOTS_DIR)
    plot_training_curves()
    plot_small_vs_synth_history()
    plot_metric_bars()
    plot_synthetic_ratio_lines()
    plot_data_regime_scaling()
    plot_confusion_matrices()
    plot_seed_scatter()
    plot_seed_ranges()
    plot_generalization_gap()
    plot_f1_improvement_waterfall()
    plot_metric_radar_like()
    plot_train_val_accuracy_curve()
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
