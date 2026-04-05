[Русский](#русская-версия) | [English](#english-version)

---

# Русская версия

## Surface Defect Research Pipeline

Этот репозиторий содержит воспроизводимый исследовательский пайплайн компьютерного зрения для распознавания поверхностных дефектов на малых выборках. Текущий фокус проекта:

1. как меняется качество модели при уменьшении количества реальных обучающих изображений;
2. помогает ли синтетическое расширение выборки компенсировать дефицит real data.

Сейчас проект уже включает полный цикл подготовки данных, обучения, оценки, синтетической генерации, repeated-run экспериментов и визуальной аналитики.

## Что уже реализовано

- подготовка датасета и создание manifest-файлов;
- сбалансированные `train/val/test` splits;
- low-data режимы `small`, `medium`, `full`;
- baseline- и improved-модели для multiclass classification;
- synthetic generation на основе controlled augmentation;
- сравнение `real only` и `real + synthetic`;
- repeated runs по нескольким `seed`;
- графики по результатам, датасету и сравнению real-vs-synthetic.

## Датасет

Текущие эксперименты выполняются на NEU steel surface defect dataset.

Классы:

- `Crazing`
- `Inclusion`
- `Patches`
- `Pitted`
- `Rolled`
- `Scratches`

Статистика:

- всего изображений: `1728`
- классов: `6`
- изображений на класс: `288`
- размер: `200x200`
- формат: `.bmp`

Сплиты:

- full train: `1206`
- validation: `258`
- test: `264`

Low-data режимы:

- `small`: `40` train-изображений на класс
- `medium`: `100` train-изображений на класс
- `full`: `201` train-изображение на класс

Во всех режимах `val` и `test` фиксированы, меняется только train subset.

## Технологии

- Python `3.13`
- PyTorch `2.11`
- Torchvision `0.26`
- Pillow
- PyYAML
- Matplotlib

Структура проекта:

- YAML-конфиги для экспериментов;
- модульная логика в `src/defect_lab`;
- отдельные скрипты для подготовки, обучения, оценки, генерации и визуализации;
- JSON-артефакты для метрик, историй обучения и repeated-run summary.

## Модель и обучение

Начальный baseline был построен на небольшой custom CNN:

- три сверточных блока;
- max pooling;
- adaptive average pooling;
- dropout перед классификатором.

Текущий основной baseline построен на `ResNet18`:

- backbone: `resnet18`
- classifier head: dropout + linear layer
- loss: cross-entropy с label smoothing
- optimizer: `AdamW`
- scheduler: cosine annealing

Train-time preprocessing:

- random resized crop;
- horizontal flip;
- rotation;
- brightness/contrast jitter;
- ImageNet-style normalization.

## Synthetic pipeline

Текущая синтетическая ветка реализована как controlled augmentation:

- flips;
- небольшие повороты;
- Gaussian blur;
- изменение контраста.

Synthetic pipeline умеет:

- генерировать class-balanced synthetic images из train split;
- сохранять их в `data/processed/...`;
- собирать merged manifest с real + synthetic training samples;
- оставлять validation и test неизменными.

## Результаты

Начальная small CNN:

- accuracy: `0.6212`
- F1-score: `0.6158`

Улучшенный `ResNet18` baseline:

- accuracy: `0.9470`
- precision: `0.9496`
- recall: `0.9470`
- F1-score: `0.9471`

Результаты по train-size режимам:

- `small`: accuracy `0.9015`, F1 `0.9020`
- `medium`: accuracy `0.9280`, F1 `0.9271`
- `full`: accuracy `0.9470`, F1 `0.9471`

Результаты по synthetic ratios в `small`-режиме:

- `small real only`: accuracy `0.9015`, F1 `0.9020`
- `small + 0.5x synthetic`: accuracy `0.9205`, F1 `0.9186`
- `small + 1.0x synthetic`: accuracy `0.9129`, F1 `0.9111`
- `small + 2.0x synthetic`: accuracy `0.9508`, F1 `0.9498`

Repeated runs по `seed = 7, 42, 99`:

- `small real only`: mean accuracy `0.9003`, mean F1 `0.8998`
- `small + 2.0x synthetic`: mean accuracy `0.9508`, mean F1 `0.9505`

## Визуализация

Уже доступны три группы графиков:

- `artifacts/plots` — графики по обучению и метрикам;
- `artifacts/plots_dataset` — графики по самому датасету;
- `artifacts/plots_synthetic` — сравнения real vs synthetic.

## Основные скрипты

- `scripts/prepare_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/generate_synthetic.py`
- `scripts/repeat_experiment.py`
- `scripts/plot_results.py`
- `scripts/plot_dataset_analysis.py`
- `scripts/plot_synthetic_comparison.py`

## Быстрый старт

```bash
pip install -e .
python scripts/prepare_dataset.py --config configs/neu_resnet_full.yaml
python scripts/train.py --config configs/neu_resnet_full.yaml
python scripts/evaluate.py --config configs/neu_resnet_full.yaml
python scripts/generate_synthetic.py --config configs/neu_resnet_small_synth_double.yaml
python scripts/repeat_experiment.py --config configs/neu_resnet_small_synth_double.yaml --seeds 7 42 99
python scripts/plot_results.py
python scripts/plot_dataset_analysis.py
python scripts/plot_synthetic_comparison.py
```

## Дальнейшее развитие

- заменить controlled synthetic generator на более сильную генеративную модель;
- проверить pipeline на другом публичном defect dataset;
- сравнить multiclass classification и binary defect detection;
- добавить CSV-экспорт результатов и publication-ready таблицы;
- расширить repeated-run эксперименты на большее число `seed`.

---

# English version

## Surface Defect Research Pipeline

This repository contains a reproducible computer vision research pipeline for surface defect recognition on small datasets. The current project focuses on:

1. how model quality changes when the amount of real training data is reduced;
2. whether synthetic data can compensate for the lack of real images.

At this stage the repository already includes a full workflow for dataset preparation, training, evaluation, synthetic generation, repeated runs, and visual analytics.

## What is already implemented

- dataset preparation and manifest generation;
- balanced `train/val/test` splits;
- low-data regimes: `small`, `medium`, `full`;
- baseline and improved models for multiclass classification;
- synthetic generation via controlled augmentation;
- `real only` versus `real + synthetic` comparisons;
- repeated runs across multiple random seeds;
- plots for training behavior, dataset analysis, and real-vs-synthetic comparison.

## Dataset

Current experiments use the NEU steel surface defect dataset.

Classes:

- `Crazing`
- `Inclusion`
- `Patches`
- `Pitted`
- `Rolled`
- `Scratches`

Statistics:

- total images: `1728`
- classes: `6`
- images per class: `288`
- image size: `200x200`
- format: `.bmp`

Splits:

- full train: `1206`
- validation: `258`
- test: `264`

Low-data regimes:

- `small`: `40` train images per class
- `medium`: `100` train images per class
- `full`: `201` train images per class

Validation and test remain fixed across regimes; only the training subset changes.

## Technologies

- Python `3.13`
- PyTorch `2.11`
- Torchvision `0.26`
- Pillow
- PyYAML
- Matplotlib

Project style:

- YAML-based experiment configuration;
- modular code in `src/defect_lab`;
- dedicated scripts for data preparation, training, evaluation, generation, and plotting;
- JSON artifacts for metrics, histories, and repeated-run summaries.

## Model and training

The initial baseline used a small custom CNN:

- three convolutional blocks;
- max pooling;
- adaptive average pooling;
- dropout before the classifier.

The current main baseline uses `ResNet18`:

- backbone: `resnet18`
- classifier head: dropout + linear layer
- loss: cross-entropy with label smoothing
- optimizer: `AdamW`
- scheduler: cosine annealing

Train-time preprocessing:

- random resized crop;
- horizontal flip;
- rotation;
- brightness/contrast jitter;
- ImageNet-style normalization.

## Synthetic pipeline

The current synthetic branch is implemented as controlled augmentation:

- flips;
- small rotations;
- Gaussian blur;
- contrast modification.

The synthetic pipeline can:

- generate class-balanced synthetic images from the train split;
- save them into `data/processed/...`;
- create a merged manifest with real + synthetic training samples;
- keep validation and test unchanged.

## Results

Initial small CNN:

- accuracy: `0.6212`
- F1-score: `0.6158`

Improved `ResNet18` baseline:

- accuracy: `0.9470`
- precision: `0.9496`
- recall: `0.9470`
- F1-score: `0.9471`

Train-size regimes:

- `small`: accuracy `0.9015`, F1 `0.9020`
- `medium`: accuracy `0.9280`, F1 `0.9271`
- `full`: accuracy `0.9470`, F1 `0.9471`

Synthetic ratios in the `small` regime:

- `small real only`: accuracy `0.9015`, F1 `0.9020`
- `small + 0.5x synthetic`: accuracy `0.9205`, F1 `0.9186`
- `small + 1.0x synthetic`: accuracy `0.9129`, F1 `0.9111`
- `small + 2.0x synthetic`: accuracy `0.9508`, F1 `0.9498`

Repeated runs for `seed = 7, 42, 99`:

- `small real only`: mean accuracy `0.9003`, mean F1 `0.8998`
- `small + 2.0x synthetic`: mean accuracy `0.9508`, mean F1 `0.9505`

## Visualization

Three groups of plots are already available:

- `artifacts/plots` — training and metric plots;
- `artifacts/plots_dataset` — dataset analysis plots;
- `artifacts/plots_synthetic` — real-vs-synthetic comparison plots.

## Main scripts

- `scripts/prepare_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/generate_synthetic.py`
- `scripts/repeat_experiment.py`
- `scripts/plot_results.py`
- `scripts/plot_dataset_analysis.py`
- `scripts/plot_synthetic_comparison.py`

## Quick start

```bash
pip install -e .
python scripts/prepare_dataset.py --config configs/neu_resnet_full.yaml
python scripts/train.py --config configs/neu_resnet_full.yaml
python scripts/evaluate.py --config configs/neu_resnet_full.yaml
python scripts/generate_synthetic.py --config configs/neu_resnet_small_synth_double.yaml
python scripts/repeat_experiment.py --config configs/neu_resnet_small_synth_double.yaml --seeds 7 42 99
python scripts/plot_results.py
python scripts/plot_dataset_analysis.py
python scripts/plot_synthetic_comparison.py
```

## Next directions

- replace the controlled synthetic generator with a stronger generative model;
- test the pipeline on another public defect dataset;
- compare multiclass classification with binary defect detection;
- add CSV export and publication-ready tables;
- extend repeated-run experiments to more seeds.
