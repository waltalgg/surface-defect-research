from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REAL_ROOT = Path("data/raw/neu_steel")
SYNTH_ROOT = Path("data/processed/synthetic_neu_small_double")
OUTPUT_DIR = Path("artifacts/plots_synthetic")
CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.style.use("tableau-colorblind10")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["axes.labelsize"] = 10


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def save(fig, name: str) -> None:
    ensure_dir(OUTPUT_DIR)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def paired_examples() -> dict[str, list[tuple[Path, Path]]]:
    pairs = {}
    for class_name in CLASS_NAMES:
        real_dir = REAL_ROOT / class_name
        synth_dir = SYNTH_ROOT / class_name
        real_map = {path.stem: path for path in real_dir.glob("*.bmp")}
        class_pairs = []
        for synth_path in sorted(synth_dir.glob("*.bmp")):
            source_stem = synth_path.stem.split("_synthetic_")[0]
            real_path = real_map.get(source_stem)
            if real_path is not None:
                class_pairs.append((real_path, synth_path))
        pairs[class_name] = class_pairs
    return pairs


def plot_real_vs_synth_grid(pairs: dict[str, list[tuple[Path, Path]]]) -> None:
    fig, axes = plt.subplots(len(CLASS_NAMES), 6, figsize=(12, 14))
    for row, class_name in enumerate(CLASS_NAMES):
        selected = pairs[class_name][:3]
        for idx, (real_path, synth_path) in enumerate(selected):
            axes[row, 2 * idx].imshow(load_gray(real_path), cmap="gray")
            axes[row, 2 * idx].set_title("Real\nРеальное", fontsize=9)
            axes[row, 2 * idx + 1].imshow(load_gray(synth_path), cmap="gray")
            axes[row, 2 * idx + 1].set_title("Synthetic\nСинтетическое", fontsize=9)
            axes[row, 2 * idx].axis("off")
            axes[row, 2 * idx + 1].axis("off")
        axes[row, 0].set_ylabel(f"{class_name}\n{class_name}", rotation=0, labelpad=35, va="center")
    fig.suptitle("Real vs Synthetic Samples / Примеры реальных и синтетических изображений", y=1.01)
    save(fig, "01_real_vs_synthetic_grid.png")


def plot_per_class_pair_pages(pairs: dict[str, list[tuple[Path, Path]]]) -> None:
    for class_name in CLASS_NAMES:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()
        selected = pairs[class_name][:8]
        for idx, (real_path, synth_path) in enumerate(selected):
            axes[2 * idx].imshow(load_gray(real_path), cmap="gray")
            axes[2 * idx].set_title(f"Real / Реальное\n{real_path.stem}", fontsize=8)
            axes[2 * idx + 1].imshow(load_gray(synth_path), cmap="gray")
            axes[2 * idx + 1].set_title(f"Synth / Синтетика\n{synth_path.stem}", fontsize=8)
            axes[2 * idx].axis("off")
            axes[2 * idx + 1].axis("off")
        for ax in axes[2 * len(selected):]:
            ax.axis("off")
        fig.suptitle(f"{class_name}: Real vs Synthetic / Реальные и синтетические", y=1.01)
        save(fig, f"02_pairs_{class_name.lower()}.png")


def plot_difference_maps(pairs: dict[str, list[tuple[Path, Path]]]) -> None:
    fig, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(9, 14))
    for row, class_name in enumerate(CLASS_NAMES):
        real_path, synth_path = pairs[class_name][0]
        real = load_gray(real_path)
        synth = load_gray(synth_path)
        diff = np.abs(real - synth)
        for col, (img, title) in enumerate(
            [(real, "Real\nРеальное"), (synth, "Synthetic\nСинтетическое"), (diff, "Abs diff\nМодуль разницы")]
        ):
            axes[row, col].imshow(img, cmap="gray" if col < 2 else "inferno")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"{class_name}\n{class_name}", rotation=0, labelpad=35, va="center")
    fig.suptitle("Pairwise Difference Maps / Карты различий для пар", y=1.01)
    save(fig, "03_difference_maps.png")


def plot_mean_real_vs_synth() -> None:
    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(14, 5))
    for col, class_name in enumerate(CLASS_NAMES):
        real_stack = np.stack([load_gray(path) for path in sorted((REAL_ROOT / class_name).glob("*.bmp"))[:80]], axis=0)
        synth_stack = np.stack([load_gray(path) for path in sorted((SYNTH_ROOT / class_name).glob("*.bmp"))[:80]], axis=0)
        axes[0, col].imshow(real_stack.mean(axis=0), cmap="magma")
        axes[0, col].set_title(f"{class_name}\nReal mean / Реальное среднее", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(synth_stack.mean(axis=0), cmap="magma")
        axes[1, col].set_title("Synthetic mean / Синтетическое среднее", fontsize=9)
        axes[1, col].axis("off")
    fig.suptitle("Mean Appearance: Real vs Synthetic / Средний образ: реальные vs синтетические", y=1.01)
    save(fig, "04_mean_real_vs_synth.png")


def plot_intensity_hist_comparison() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    bins = np.linspace(0, 255, 64)
    for ax, class_name in zip(axes, CLASS_NAMES, strict=False):
        real_pixels = np.concatenate([load_gray(path).reshape(-1) for path in sorted((REAL_ROOT / class_name).glob("*.bmp"))[:80]])
        synth_pixels = np.concatenate([load_gray(path).reshape(-1) for path in sorted((SYNTH_ROOT / class_name).glob("*.bmp"))[:80]])
        ax.hist(real_pixels, bins=bins, density=True, alpha=0.55, label="Real / Реальные")
        ax.hist(synth_pixels, bins=bins, density=True, alpha=0.55, label="Synthetic / Синтетика")
        ax.set_title(class_name)
        ax.set_xlabel("Intensity / Яркость")
        ax.set_ylabel("Density / Плотность")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Pixel Intensity: Real vs Synthetic / Яркость пикселей: реальные vs синтетические", y=1.01)
    save(fig, "05_intensity_hist_comparison.png")


def plot_feature_scatter_real_vs_synth() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for class_name in CLASS_NAMES:
        real_points = []
        synth_points = []
        for path in sorted((REAL_ROOT / class_name).glob("*.bmp"))[:60]:
            gray = load_gray(path)
            real_points.append((float(gray.mean()), float(gray.std())))
        for path in sorted((SYNTH_ROOT / class_name).glob("*.bmp"))[:60]:
            gray = load_gray(path)
            synth_points.append((float(gray.mean()), float(gray.std())))
        real_points = np.asarray(real_points)
        synth_points = np.asarray(synth_points)
        axes[0].scatter(real_points[:, 0], real_points[:, 1], s=18, alpha=0.55, label=class_name)
        axes[1].scatter(synth_points[:, 0], synth_points[:, 1], s=18, alpha=0.55, label=class_name)
    axes[0].set_title("Real Images / Реальные изображения")
    axes[1].set_title("Synthetic Images / Синтетические изображения")
    for ax in axes:
        ax.set_xlabel("Mean brightness / Средняя яркость")
        ax.set_ylabel("Contrast / Контраст")
        ax.grid(alpha=0.3)
    axes[1].legend(ncol=2)
    fig.suptitle("Feature Scatter: Real vs Synthetic / Пространство признаков: реальные vs синтетические", y=1.01)
    save(fig, "06_feature_scatter_real_vs_synth.png")


def plot_real_synth_mosaic() -> None:
    fig, axes = plt.subplots(6, 8, figsize=(13, 10))
    for row, class_name in enumerate(CLASS_NAMES):
        real_paths = sorted((REAL_ROOT / class_name).glob("*.bmp"))[:4]
        synth_paths = sorted((SYNTH_ROOT / class_name).glob("*.bmp"))[:4]
        for col, path in enumerate(real_paths):
            axes[row, col].imshow(load_gray(path), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title("Real\nРеальное", fontsize=9)
        for col, path in enumerate(synth_paths, start=4):
            axes[row, col].imshow(load_gray(path), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title("Synthetic\nСинтетика", fontsize=9)
        axes[row, 0].set_ylabel(f"{class_name}\n{class_name}", rotation=0, labelpad=35, va="center")
    fig.suptitle("Large Mosaic of Real and Synthetic Images / Большая мозаика реальных и синтетических изображений", y=1.01)
    save(fig, "07_real_synth_large_mosaic.png")


def plot_real_synth_statistics_bars() -> None:
    real_means = []
    synth_means = []
    real_std = []
    synth_std = []
    for class_name in CLASS_NAMES:
        real_images = [load_gray(path) for path in sorted((REAL_ROOT / class_name).glob("*.bmp"))[:80]]
        synth_images = [load_gray(path) for path in sorted((SYNTH_ROOT / class_name).glob("*.bmp"))[:80]]
        real_means.append(np.mean([img.mean() for img in real_images]))
        synth_means.append(np.mean([img.mean() for img in synth_images]))
        real_std.append(np.mean([img.std() for img in real_images]))
        synth_std.append(np.mean([img.std() for img in synth_images]))

    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - width / 2, real_means, width, label="Real / Реальные")
    axes[0].bar(x + width / 2, synth_means, width, label="Synthetic / Синтетика")
    axes[0].set_title("Average Brightness / Средняя яркость")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=25)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, real_std, width, label="Real / Реальные")
    axes[1].bar(x + width / 2, synth_std, width, label="Synthetic / Синтетика")
    axes[1].set_title("Average Contrast / Средний контраст")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=25)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    save(fig, "08_real_synth_statistics_bars.png")


def plot_best_case_gallery(pairs: dict[str, list[tuple[Path, Path]]]) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(12, 7))
    selected_classes = CLASS_NAMES[:3]
    for row, class_name in enumerate(selected_classes):
        selected = pairs[class_name][5:8]
        for idx, (real_path, synth_path) in enumerate(selected):
            col = idx * 2
            axes[row, col].imshow(load_gray(real_path), cmap="gray")
            axes[row, col].set_title("Real / Реальное", fontsize=9)
            axes[row, col + 1].imshow(load_gray(synth_path), cmap="gray")
            axes[row, col + 1].set_title("Synthetic / Синтетическое", fontsize=9)
            axes[row, col].axis("off")
            axes[row, col + 1].axis("off")
        axes[row, 0].set_ylabel(f"{class_name}", rotation=0, labelpad=30, va="center")
    fig.suptitle("Additional Real-Synthetic Examples / Дополнительные примеры real-synthetic", y=1.01)
    save(fig, "09_additional_pair_gallery.png")


def plot_difference_histograms(pairs: dict[str, list[tuple[Path, Path]]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for ax, class_name in zip(axes, CLASS_NAMES, strict=False):
        diffs = []
        for real_path, synth_path in pairs[class_name][:20]:
            diff = np.abs(load_gray(real_path) - load_gray(synth_path))
            diffs.append(diff.reshape(-1))
        diffs = np.concatenate(diffs)
        ax.hist(diffs, bins=40, color="tab:red", alpha=0.8)
        ax.set_title(f"{class_name}\nDifference histogram / Гистограмма различий")
        ax.set_xlabel("Absolute difference / Абсолютное отличие")
        ax.set_ylabel("Count / Число")
        ax.grid(alpha=0.25)
    save(fig, "10_difference_histograms.png")


def main() -> None:
    set_style()
    ensure_dir(OUTPUT_DIR)
    pairs = paired_examples()
    plot_real_vs_synth_grid(pairs)
    plot_per_class_pair_pages(pairs)
    plot_difference_maps(pairs)
    plot_mean_real_vs_synth()
    plot_intensity_hist_comparison()
    plot_feature_scatter_real_vs_synth()
    plot_real_synth_mosaic()
    plot_real_synth_statistics_bars()
    plot_best_case_gallery(pairs)
    plot_difference_histograms(pairs)
    print(f"Saved synthetic comparison plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
