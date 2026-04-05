import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASET_ROOT = Path("data/raw/neu_steel")
OUTPUT_DIR = Path("artifacts/plots_dataset")
CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.style.use("tableau-colorblind10")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["axes.labelsize"] = 10


def list_images_by_class() -> dict[str, list[Path]]:
    return {class_name: sorted((DATASET_ROOT / class_name).glob("*.bmp")) for class_name in CLASS_NAMES}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def compute_edge_strength(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return float((gx + gy) / 2.0)


def save(fig, name: str) -> None:
    ensure_dir(OUTPUT_DIR)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(images_by_class: dict[str, list[Path]]) -> None:
    counts = [len(images_by_class[class_name]) for class_name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASS_NAMES, counts)
    ax.set_title("Class Distribution / Распределение классов")
    ax.set_xlabel("Class / Класс")
    ax.set_ylabel("Number of images / Число изображений")
    ax.grid(axis="y", alpha=0.3)
    for bar, count in zip(bars, counts, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 2, str(count), ha="center")
    save(fig, "01_dataset_class_distribution.png")


def plot_sample_grid(images_by_class: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(len(CLASS_NAMES), 5, figsize=(10, 12))
    for row, class_name in enumerate(CLASS_NAMES):
        sample_paths = images_by_class[class_name][:5]
        for col, path in enumerate(sample_paths):
            axes[row, col].imshow(load_gray(path), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"Ex {col + 1}\nПример {col + 1}")
        axes[row, 0].set_ylabel(f"{class_name}\n{class_name}", rotation=0, labelpad=35, va="center")
    fig.suptitle("Sample Images by Class / Примеры изображений по классам", y=1.01)
    save(fig, "02_dataset_sample_grid.png")


def plot_mean_images(images_by_class: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for ax, class_name in zip(axes, CLASS_NAMES, strict=False):
        stack = np.stack([load_gray(path) for path in images_by_class[class_name]], axis=0)
        mean_image = stack.mean(axis=0)
        ax.imshow(mean_image, cmap="magma")
        ax.set_title(f"{class_name}\nMean image / Средний образ")
        ax.axis("off")
    save(fig, "03_dataset_mean_images.png")


def plot_brightness_histograms(images_by_class: dict[str, list[Path]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 255, 64)
    for class_name in CLASS_NAMES:
        pixels = np.concatenate([load_gray(path).reshape(-1) for path in images_by_class[class_name][:80]])
        hist, edges = np.histogram(pixels, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, linewidth=2, label=f"{class_name} / {class_name}")
    ax.set_title("Pixel Intensity Distribution / Распределение яркости пикселей")
    ax.set_xlabel("Pixel intensity / Яркость пикселя")
    ax.set_ylabel("Density / Плотность")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    save(fig, "04_dataset_brightness_histograms.png")


def plot_brightness_boxplots(images_by_class: dict[str, list[Path]]) -> None:
    means = []
    stds = []
    for class_name in CLASS_NAMES:
        class_means = []
        class_stds = []
        for path in images_by_class[class_name]:
            gray = load_gray(path)
            class_means.append(float(gray.mean()))
            class_stds.append(float(gray.std()))
        means.append(class_means)
        stds.append(class_stds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(means, tick_labels=CLASS_NAMES)
    axes[0].set_title("Image Brightness / Средняя яркость изображений")
    axes[0].set_ylabel("Mean intensity / Средняя интенсивность")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(stds, tick_labels=CLASS_NAMES)
    axes[1].set_title("Image Contrast / Контраст изображений")
    axes[1].set_ylabel("Std intensity / Стандартное отклонение")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.3)
    save(fig, "05_dataset_brightness_contrast_boxplots.png")


def plot_edge_strength_boxplot(images_by_class: dict[str, list[Path]]) -> None:
    values = []
    for class_name in CLASS_NAMES:
        values.append([compute_edge_strength(load_gray(path)) for path in images_by_class[class_name]])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, tick_labels=CLASS_NAMES)
    ax.set_title("Edge Strength by Class / Сила границ по классам")
    ax.set_ylabel("Average gradient / Средний градиент")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "06_dataset_edge_strength_boxplot.png")


def plot_file_size_distribution(images_by_class: dict[str, list[Path]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[path.stat().st_size / 1024 for path in images_by_class[class_name]] for class_name in CLASS_NAMES]
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(CLASS_NAMES) + 1))
    ax.set_xticklabels(CLASS_NAMES, rotation=25)
    ax.set_title("File Size Distribution / Распределение размеров файлов")
    ax.set_ylabel("File size, KB / Размер файла, КБ")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "07_dataset_file_size_violin.png")


def plot_feature_scatter(images_by_class: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for class_name in CLASS_NAMES:
        samples = []
        for path in images_by_class[class_name][:120]:
            gray = load_gray(path)
            samples.append((float(gray.mean()), float(gray.std()), compute_edge_strength(gray)))
        samples = np.asarray(samples)
        axes[0].scatter(samples[:, 0], samples[:, 1], s=18, alpha=0.6, label=class_name)
        axes[1].scatter(samples[:, 1], samples[:, 2], s=18, alpha=0.6, label=class_name)

    axes[0].set_title("Brightness vs Contrast / Яркость vs контраст")
    axes[0].set_xlabel("Mean intensity / Средняя яркость")
    axes[0].set_ylabel("Std intensity / Контраст")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Contrast vs Edge Strength / Контраст vs сила границ")
    axes[1].set_xlabel("Std intensity / Контраст")
    axes[1].set_ylabel("Edge strength / Сила границ")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2)
    save(fig, "08_dataset_feature_scatter.png")


def plot_central_profiles(images_by_class: dict[str, list[Path]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for class_name in CLASS_NAMES:
        stack = np.stack([load_gray(path) for path in images_by_class[class_name]], axis=0)
        mean_image = stack.mean(axis=0)
        central_profile = mean_image[mean_image.shape[0] // 2, :]
        ax.plot(central_profile, linewidth=2, label=f"{class_name} / {class_name}")
    ax.set_title("Central Intensity Profiles / Центральные профили яркости")
    ax.set_xlabel("Horizontal position / Горизонтальная координата")
    ax.set_ylabel("Intensity / Яркость")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    save(fig, "09_dataset_central_profiles.png")


def plot_top_variance_maps(images_by_class: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for ax, class_name in zip(axes, CLASS_NAMES, strict=False):
        stack = np.stack([load_gray(path) for path in images_by_class[class_name]], axis=0)
        variance_map = stack.var(axis=0)
        ax.imshow(variance_map, cmap="inferno")
        ax.set_title(f"{class_name}\nVariance map / Карта вариативности")
        ax.axis("off")
    save(fig, "10_dataset_variance_maps.png")


def plot_class_texture_summary(images_by_class: dict[str, list[Path]]) -> None:
    means = []
    contrasts = []
    edges = []
    for class_name in CLASS_NAMES:
        class_means = []
        class_contrasts = []
        class_edges = []
        for path in images_by_class[class_name]:
            gray = load_gray(path)
            class_means.append(float(gray.mean()))
            class_contrasts.append(float(gray.std()))
            class_edges.append(compute_edge_strength(gray))
        means.append(np.mean(class_means))
        contrasts.append(np.mean(class_contrasts))
        edges.append(np.mean(class_edges))

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, means, width=width, label="Mean brightness / Средняя яркость")
    ax.bar(x, contrasts, width=width, label="Mean contrast / Средний контраст")
    ax.bar(x + width, edges, width=width, label="Mean edge strength / Средняя сила границ")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25)
    ax.set_title("Texture Summary by Class / Сводка текстурных признаков по классам")
    ax.set_ylabel("Feature value / Значение признака")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, "11_dataset_texture_summary.png")


def plot_image_montage_with_stats(images_by_class: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    selected_classes = CLASS_NAMES[:4]
    for col, class_name in enumerate(selected_classes):
        sample = load_gray(images_by_class[class_name][0])
        axes[0, col].imshow(sample, cmap="gray")
        axes[0, col].set_title(f"{class_name}\nExample / Пример")
        axes[0, col].axis("off")

        axes[1, col].hist(sample.ravel(), bins=32, color="tab:blue", alpha=0.8)
        axes[1, col].set_title("Histogram / Гистограмма")
        axes[1, col].set_xlabel("Intensity / Яркость")
        axes[1, col].set_ylabel("Count / Число")

        edges = np.abs(np.diff(sample, axis=1))
        axes[2, col].imshow(edges, cmap="magma")
        axes[2, col].set_title("Edge map / Карта границ")
        axes[2, col].axis("off")
    fig.suptitle("Image, Histogram, Edge Map / Изображение, гистограмма, карта границ", y=1.01)
    save(fig, "12_dataset_image_hist_edge_montage.png")


def main() -> None:
    set_style()
    ensure_dir(OUTPUT_DIR)
    images_by_class = list_images_by_class()
    plot_class_distribution(images_by_class)
    plot_sample_grid(images_by_class)
    plot_mean_images(images_by_class)
    plot_brightness_histograms(images_by_class)
    plot_brightness_boxplots(images_by_class)
    plot_edge_strength_boxplot(images_by_class)
    plot_file_size_distribution(images_by_class)
    plot_feature_scatter(images_by_class)
    plot_central_profiles(images_by_class)
    plot_top_variance_maps(images_by_class)
    plot_class_texture_summary(images_by_class)
    plot_image_montage_with_stats(images_by_class)
    print(f"Saved dataset plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
