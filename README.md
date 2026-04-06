[Русский](#русская-версия) | [English](#english-version)

---

# Русская версия

## Surface Defect Research Pipeline

Этот репозиторий содержит исследовательский пайплайн компьютерного зрения для анализа поверхностных дефектов на малых выборках. Цель проекта — изучить, как меняется качество модели при ограниченном количестве реальных данных и насколько полезно синтетическое расширение обучающей выборки.

## Что уже сделано

- собран единый пайплайн подготовки данных, обучения, оценки и генерации синтетики;
- поддерживаются режимы `small`, `medium`, `full`;
- реализованы baseline-эксперименты на `ResNet18`;
- поддерживаются binary- и multiclass-сценарии;
- добавлены инструменты для таблиц, графиков и галерей `real vs synthetic`;
- добавлен локальный UI для запуска этапов пайплайна и просмотра логов.

## Датасеты

В проекте сейчас используются:

- `NEU Steel Surface Defect Dataset` — многоклассовый набор дефектов металлической поверхности;
- `Magnetic Tile Surface Defects` — компактный industrial dataset для multiclass и binary-экспериментов;
- `PY-CrackDB` — дорожный датасет трещин, подготовленный для следующего этапа экспериментов.

## Как устроена синтетика

В проекте реализованы несколько режимов генерации:

- `basic` — базовые геометрические и фотометрические преобразования;
- `strong` — более агрессивные crop/resize и искажения;
- `blend` — смешивание изображений одного класса на уровне локальных патчей;
- `composite` — перенос и деформация предполагаемой дефектной области на новый фон.

Текущий `composite`-подход использует оценку маски дефекта, извлечение patch, его деформацию и вставку в другой контекст с адаптацией под локальный фон.

## Быстрый запуск

```bash
pip install -e .
python scripts/prepare_dataset.py --config configs/neu_resnet_small_gpu.yaml
python scripts/train.py --config configs/neu_resnet_small_gpu.yaml
python scripts/evaluate.py --config configs/neu_resnet_small_gpu.yaml
python scripts/generate_synthetic.py --config configs/magnetic_tile_binary_small_synth_composite_gpu.yaml
```

Для запуска UI:

```bash
streamlit run ui/app.py
```

---

# English version

## Surface Defect Research Pipeline

This repository contains a computer vision research pipeline for surface defect analysis in low-data regimes. The main goal is to study how model quality changes when only a limited amount of real data is available and whether synthetic data can improve training under these conditions.

## What is already implemented

- a unified workflow for dataset preparation, training, evaluation, and synthetic generation;
- `small`, `medium`, and `full` training regimes;
- baseline experiments based on `ResNet18`;
- binary and multiclass setups depending on the dataset;
- tools for summary tables, plots, and `real vs synthetic` galleries;
- a lightweight local UI for running pipeline stages and monitoring logs.

## Datasets

The project currently uses:

- `NEU Steel Surface Defect Dataset` for multiclass metal surface defect recognition;
- `Magnetic Tile Surface Defects` as a compact industrial dataset for multiclass and binary experiments;
- `PY-CrackDB` as a road crack dataset prepared for the next experimental stage.

## Synthetic data pipeline

Several synthetic generation modes are available:

- `basic` — simple geometric and photometric transforms;
- `strong` — stronger crop/resize and perturbation strategy;
- `blend` — same-class local patch mixing;
- `composite` — defect-region transfer and deformation on a new background.

The current `composite` approach estimates a defect mask, extracts a patch, deforms it, and pastes it into a different visual context while adapting it to the target background.

## Quick start

```bash
pip install -e .
python scripts/prepare_dataset.py --config configs/neu_resnet_small_gpu.yaml
python scripts/train.py --config configs/neu_resnet_small_gpu.yaml
python scripts/evaluate.py --config configs/neu_resnet_small_gpu.yaml
python scripts/generate_synthetic.py --config configs/magnetic_tile_binary_small_synth_composite_gpu.yaml
```

To launch the UI:

```bash
streamlit run ui/app.py
```
