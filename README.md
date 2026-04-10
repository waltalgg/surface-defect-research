[Русский](#русская-версия) | [English](#english-version)

---

# Русская версия

## Surface Defect Research Pipeline

Этот репозиторий содержит исследовательский пайплайн компьютерного зрения для анализа поверхностных дефектов на малых выборках. Цель проекта — изучить, как меняется качество модели при ограниченном количестве реальных данных и насколько полезно синтетическое расширение обучающей выборки.

## Что уже сделано

- собран единый пайплайн подготовки данных, обучения, оценки и генерации синтетики;
- поддерживаются режимы `small`, `full` и варианты `small + synthetic`;
- реализованы baseline-эксперименты на `ResNet18`;
- поддерживаются `multiclass` и `binary` сценарии;
- добавлены инструменты для таблиц, графиков и галерей `real vs synthetic`;
- добавлен локальный UI для запуска этапов пайплайна и просмотра логов.

## Датасеты

В проекте сейчас используются:

- `NEU Steel Surface Defect Dataset` — многоклассовый набор дефектов металлической поверхности;
- `PY-CrackDB` — дорожный датасет трещин для binary low-data экспериментов.

Для обоих датасетов в проекте используется единая mask-aware логика, но с разными источниками масок и генерации:

- для `NEU` используются автоматически построенные `pseudo-masks`, полученные эвристическим выделением дефектной области;
- для `PY-CrackDB` основная target-ветка строится вокруг `SPADE`, где crack-синтетика генерируется из маски трещины.

Дополнительно для `PY-CrackDB` мы отдельно сравниваем два режима генерации:

- обычный `composite`;
- генеративный `SPADE`.

## Как устроена синтетика

В проекте реализованы несколько режимов генерации:

- `basic` — базовые геометрические и фотометрические преобразования;
- `strong` — более агрессивные crop/resize и искажения;
- `blend` — смешивание изображений одного класса на уровне локальных патчей;
- `composite` — перенос и деформация предполагаемой дефектной области на новый фон.

Текущий `composite`-подход использует оценку маски дефекта, извлечение patch, его деформацию и вставку в другой контекст с адаптацией под локальный фон.

Общая схема генерации синтетики выглядит так:

1. из реального `train`-набора выбирается исходное изображение;
2. определяется область дефекта:
   - для `NEU` через автоматически построенную `pseudo-mask`,
   - для `PY-CrackDB` через mask-based road-crack branch;
3. дефектная структура деформируется:
   - масштабируется,
   - поворачивается,
   - сдвигается,
   - частично искажается;
4. затем она переносится на другой реальный фон того же домена;
5. на финальном этапе применяется локальная подстройка под фон:
   - перенос текстуры,
   - сглаживание краёв,
   - quality filtering для отбора неудачных примеров.

Для `NEU` используется mask-guided synthetic pipeline на основе `pseudo-masks`:

- сначала строится `pseudo-mask` дефекта;
- затем деформируется именно маска дефектной области;
- после этого дефектная структура переносится на новый реальный фон;
- результат сглаживается и проходит quality filtering.

Для road-crack ветки `PY-CrackDB` используются два сравниваемых подхода:

- `composite`:
  - деформация маски,
  - перенос crack-texture,
  - gradient field,
  - `seamless / Poisson-like` blending;
- `SPADE`:
  - генерация трещины по маске как отдельной condition map.

Именно для `PY-CrackDB` мы отдельно сравниваем, что лучше работает в synthetic augmentation:

- обычный `composite`;
- или `SPADE`.

## Быстрый запуск

```bash
pip install -e .
python scripts/prepare_dataset.py --config configs/neu_resnet_small_gpu.yaml
python scripts/train.py --config configs/neu_resnet_small_gpu.yaml
python scripts/evaluate.py --config configs/neu_resnet_small_gpu.yaml
python scripts/generate_synthetic.py --config configs/neu_resnet_small_synth_segmentation_gpu.yaml
```

Для обучения масочной ветки `NEU` и segmentation-базы:

```bash
python scripts/segmentation/build_neu_pseudo_masks.py
python scripts/segmentation/train.py --config configs/neu_segmentation_gpu.yaml
python scripts/segmentation/train.py --config configs/py_crackdb_segmentation_gpu.yaml
```

Для `SPADE`-ветки на `PY-CrackDB`:

```bash
python scripts/spade/train.py --config configs/py_crackdb_spade_gpu.yaml
```

Для запуска UI:

```bash
streamlit run ui/app.py
```

Для генерации графиков и сводок из одной точки входа:

```bash
python scripts/plot_latest_results.py --all
```

Можно генерировать и только нужные части:

```bash
python scripts/plot_latest_results.py --plots train_size synthetic_ratio
python scripts/plot_latest_results.py --plots synthetic_examples composite_examples
python scripts/plot_latest_results.py --plots summary
```

---

# English version

## Surface Defect Research Pipeline

This repository contains a computer vision research pipeline for surface defect analysis in low-data regimes. The main goal is to study how model quality changes when only a limited amount of real data is available and whether synthetic data can improve training under these conditions.

## What is already implemented

- a unified workflow for dataset preparation, training, evaluation, and synthetic generation;
- `small`, `full`, and `small + synthetic` training regimes;
- baseline experiments based on `ResNet18`;
- `multiclass` and `binary` setups depending on the dataset;
- tools for summary tables, plots, and `real vs synthetic` galleries;
- a lightweight local UI for running pipeline stages and monitoring logs.

## Datasets

The project currently uses:

- `NEU Steel Surface Defect Dataset` for multiclass metal surface defect recognition;
- `PY-CrackDB` as a road crack dataset for binary low-data experiments.

The project uses a unified mask-aware idea across both datasets, but with different main synthetic paths:

- `NEU` uses automatically generated `pseudo-masks` built from heuristic defect-region extraction;
- `PY-CrackDB` uses a crack-generation branch centered on `SPADE`.

For `PY-CrackDB`, the project explicitly compares two synthetic strategies:

- standard `composite`;
- `SPADE`.

## Synthetic data pipeline

Several synthetic generation modes are available:

- `basic` — simple geometric and photometric transforms;
- `strong` — stronger crop/resize and perturbation strategy;
- `blend` — same-class local patch mixing;
- `composite` — defect-region transfer and deformation on a new background.

The current `composite` approach estimates a defect mask, extracts a patch, deforms it, and pastes it into a different visual context while adapting it to the target background.

The overall synthetic generation pipeline is:

1. select a real sample from the training split;
2. estimate the defect region:
   - via an automatically generated `pseudo-mask` for `NEU`,
   - or via the road-crack mask branch for `PY-CrackDB`;
3. deform the defect structure:
   - scaling,
   - rotation,
   - translation,
   - local geometric perturbations;
4. transfer it onto another real background from the same domain;
5. apply local adaptation:
   - texture transfer,
   - edge smoothing,
   - quality filtering to reject weak or implausible samples.

For `NEU`, the main synthetic path is a mask-guided pipeline based on `pseudo-masks`:

- build a pseudo-mask of the defect region;
- deform the mask rather than the whole image;
- transfer defect structure onto a new real background;
- apply local smoothing and quality filtering.

For `PY-CrackDB`, the project compares two road-crack synthetic paths:

- `composite`:
  - mask deformation,
  - crack-texture transfer,
  - gradient field transfer,
  - `seamless / Poisson-like` blending;
- `SPADE`:
  - crack generation from a mask-conditioned input.

This comparison is an explicit part of the project: we evaluate whether `SPADE` outperforms the standard `composite` pipeline for road-crack synthesis.

## Quick start

```bash
pip install -e .
python scripts/prepare_dataset.py --config configs/neu_resnet_small_gpu.yaml
python scripts/train.py --config configs/neu_resnet_small_gpu.yaml
python scripts/evaluate.py --config configs/neu_resnet_small_gpu.yaml
python scripts/generate_synthetic.py --config configs/neu_resnet_small_synth_segmentation_gpu.yaml
```

For the `NEU` mask-learning branch and the segmentation backbone:

```bash
python scripts/segmentation/build_neu_pseudo_masks.py
python scripts/segmentation/train.py --config configs/neu_segmentation_gpu.yaml
python scripts/segmentation/train.py --config configs/py_crackdb_segmentation_gpu.yaml
```

For the `PY-CrackDB` `SPADE` branch:

```bash
python scripts/spade/train.py --config configs/py_crackdb_spade_gpu.yaml
```

To launch the UI:

```bash
streamlit run ui/app.py
```

To generate plots and summaries from a single entry point:

```bash
python scripts/plot_latest_results.py --all
```

You can also generate only specific parts:

```bash
python scripts/plot_latest_results.py --plots train_size synthetic_ratio
python scripts/plot_latest_results.py --plots synthetic_examples composite_examples
python scripts/plot_latest_results.py --plots summary
```
