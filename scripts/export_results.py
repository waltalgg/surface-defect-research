from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_lab.utils import ensure_dir, write_json


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Export experiment summaries into CSV, JSON, and Markdown tables.")
    parser.add_argument("--runs-dir", default="artifacts/runs", help="Directory with experiment run folders.")
    parser.add_argument("--output-dir", default="artifacts/reports", help="Directory for exported summaries.")
    return parser.parse_args()


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _infer_group(name: str) -> str:
    lowered = name.lower()
    if "synth" in lowered:
        return "synthetic"
    if "small" in lowered:
        return "small"
    if "medium" in lowered:
        return "medium"
    if "full" in lowered:
        return "full"
    return "other"


def _infer_dataset_name(config: dict) -> str:
    dataset_cfg = config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir")
    if raw_dir:
        raw_path = Path(str(raw_dir))
        parts = {part.lower() for part in raw_path.parts}
        if "py_crackdb" in parts:
            return "py_crackdb"
        if "neu_steel" in parts:
            return "neu_steel"
        return raw_path.name
    image_dir = dataset_cfg.get("image_dir")
    if image_dir:
        image_path = Path(str(image_dir))
        parts = {part.lower() for part in image_path.parts}
        if "py_crackdb" in parts:
            return "py_crackdb"
        if "neu_segmentation" in parts or "neu_steel" in parts:
            return "neu_steel"
        return image_path.parent.name or image_path.name
    return "unknown"


def _load_run_record(run_dir: Path) -> dict | None:
    metrics_payload = _safe_read_json(run_dir / "test_metrics.json")
    if metrics_payload is None:
        return None
    if "metrics" not in metrics_payload:
        return None
    required_metric_keys = {"accuracy", "precision", "recall", "f1"}
    if not required_metric_keys.issubset(set(metrics_payload["metrics"].keys())):
        return None

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    history_payload = _safe_read_json(run_dir / "history.json") or {}
    dataset_cfg = config.get("dataset", {})
    synthetic_cfg = config.get("synthetic", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    metadata = checkpoint.get("manifest_metadata", {})

    return {
        "experiment": run_dir.name,
        "task_type": "classification",
        "group": _infer_group(run_dir.name),
        "dataset_name": _infer_dataset_name(config),
        "manifest_path": dataset_cfg.get("manifest_path", ""),
        "num_classes": len(checkpoint.get("classes", [])),
        "model": model_cfg.get("name", ""),
        "epochs": training_cfg.get("epochs", ""),
        "train_images_per_class": dataset_cfg.get("train_images_per_class", ""),
        "label_map": json.dumps(dataset_cfg.get("label_map") or {}, ensure_ascii=False),
        "synthetic_enabled": synthetic_cfg.get("enabled", False),
        "synthetic_method": synthetic_cfg.get("method", "basic"),
        "synthetic_multiplier": synthetic_cfg.get("multiplier", 0.0),
        "accuracy": metrics_payload["metrics"]["accuracy"],
        "precision": metrics_payload["metrics"]["precision"],
        "recall": metrics_payload["metrics"]["recall"],
        "f1": metrics_payload["metrics"]["f1"],
        "test_loss": metrics_payload["test_loss"],
        "best_val_f1": history_payload.get("best_f1", ""),
        "manifest_synthetic_method": metadata.get("synthetic_method", ""),
    }


def _load_special_run_record(run_dir: Path) -> dict | None:
    metrics_payload = _safe_read_json(run_dir / "test_metrics.json")
    if metrics_payload is None:
        return None

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    dataset_name = _infer_dataset_name(config)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    if {"test_dice", "test_iou"}.issubset(metrics_payload.keys()):
        return {
            "experiment": run_dir.name,
            "task_type": "segmentation",
            "group": _infer_group(run_dir.name),
            "dataset_name": dataset_name,
            "model": model_cfg.get("name", ""),
            "epochs": training_cfg.get("epochs", ""),
            "primary_metric_name": "dice",
            "primary_metric_value": metrics_payload["test_dice"],
            "secondary_metric_name": "iou",
            "secondary_metric_value": metrics_payload["test_iou"],
            "test_loss": metrics_payload.get("test_loss", ""),
            "notes": "",
        }

    if "test_l1" in metrics_payload:
        return {
            "experiment": run_dir.name,
            "task_type": "generation",
            "group": _infer_group(run_dir.name),
            "dataset_name": dataset_name,
            "model": model_cfg.get("name", ""),
            "epochs": training_cfg.get("epochs", ""),
            "primary_metric_name": "l1",
            "primary_metric_value": metrics_payload["test_l1"],
            "secondary_metric_name": "",
            "secondary_metric_value": "",
            "test_loss": "",
            "notes": "lower is better",
        }

    return None


def _collect_run_records(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        record = _load_run_record(run_dir)
        if record is not None:
            rows.append(record)
    return rows


def _collect_all_run_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        record = _load_run_record(run_dir)
        if record is not None:
            rows.append(
                {
                    "experiment": record["experiment"],
                    "task_type": "classification",
                    "dataset_name": record["dataset_name"],
                    "model": record["model"],
                    "epochs": record["epochs"],
                    "group": record["group"],
                    "synthetic_enabled": record["synthetic_enabled"],
                    "synthetic_method": record["synthetic_method"],
                    "synthetic_multiplier": record["synthetic_multiplier"],
                    "primary_metric_name": "f1",
                    "primary_metric_value": record["f1"],
                    "secondary_metric_name": "accuracy",
                    "secondary_metric_value": record["accuracy"],
                    "test_loss": record["test_loss"],
                    "notes": "",
                }
            )
            continue

        special_record = _load_special_run_record(run_dir)
        if special_record is not None:
            rows.append(special_record)
    return rows


def _aggregate_by_experiment(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        base_name = row["experiment"]
        if "_seed_" in base_name:
            base_name = base_name.rsplit("_seed_", 1)[0]
        grouped.setdefault(base_name, []).append(row)

    summary_rows = []
    for experiment, items in sorted(grouped.items()):
        numeric_keys = ["accuracy", "precision", "recall", "f1", "test_loss"]
        first = items[0]
        summary = {
            "experiment": experiment,
            "runs": len(items),
            "dataset_name": first["dataset_name"],
            "model": first["model"],
            "num_classes": first["num_classes"],
            "synthetic_enabled": first["synthetic_enabled"],
            "synthetic_method": first["synthetic_method"],
            "synthetic_multiplier": first["synthetic_multiplier"],
        }
        for key in numeric_keys:
            values = [float(item[key]) for item in items]
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
        summary_rows.append(summary)
    return summary_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    lines = [
        "# Experiment Summary",
        "",
        "## Mean Metrics by Experiment",
        "",
        "| Experiment | Dataset | Runs | Synthetic | Method | Multiplier | Accuracy | F1 |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['dataset_name']} | {row['runs']} | "
            f"{row['synthetic_enabled']} | {row['synthetic_method']} | {row['synthetic_multiplier']} | "
            f"{row['accuracy_mean']:.4f} | {row['f1_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_all_results_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_all_results_markdown(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    lines = [
        "# All Results Table",
        "",
        "## Unified Overview",
        "",
        "| Experiment | Task | Dataset | Model | Epochs | Primary metric | Value | Secondary metric | Value | Test loss | Notes |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        test_loss = row["test_loss"]
        test_loss_text = f"{float(test_loss):.4f}" if test_loss not in ("", None) else ""
        secondary_value = row["secondary_metric_value"]
        secondary_text = f"{float(secondary_value):.4f}" if secondary_value not in ("", None, "") else ""
        lines.append(
            f"| {row['experiment']} | {row['task_type']} | {row['dataset_name']} | {row['model']} | "
            f"{row['epochs']} | {row['primary_metric_name']} | {float(row['primary_metric_value']):.4f} | "
            f"{row['secondary_metric_name']} | {secondary_text} | {test_loss_text} | {row['notes']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = ensure_dir(args.output_dir)
    run_rows = _collect_run_records(runs_dir)
    summary_rows = _aggregate_by_experiment(run_rows)
    all_rows = _collect_all_run_rows(runs_dir)

    _write_csv(output_dir / "experiment_runs.csv", run_rows)
    _write_csv(output_dir / "experiment_summary.csv", summary_rows)
    _write_markdown_table(output_dir / "experiment_summary.md", summary_rows)
    _write_all_results_csv(output_dir / "all_results_table.csv", all_rows)
    _write_all_results_markdown(output_dir / "all_results_table.md", all_rows)
    write_json(output_dir / "experiment_runs.json", {"runs": run_rows})
    write_json(output_dir / "experiment_summary.json", {"summary": summary_rows})
    write_json(output_dir / "all_results_table.json", {"rows": all_rows})
    print(
        f"Exported {len(run_rows)} classification rows, {len(summary_rows)} experiment summaries, "
        f"and {len(all_rows)} total rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
