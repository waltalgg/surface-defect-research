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
        "group": _infer_group(run_dir.name),
        "dataset_name": Path(str(dataset_cfg.get("raw_dir", "unknown"))).name,
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


def _collect_run_records(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        record = _load_run_record(run_dir)
        if record is not None:
            rows.append(record)
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


def main() -> None:
    args = _parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = ensure_dir(args.output_dir)
    run_rows = _collect_run_records(runs_dir)
    summary_rows = _aggregate_by_experiment(run_rows)

    _write_csv(output_dir / "experiment_runs.csv", run_rows)
    _write_csv(output_dir / "experiment_summary.csv", summary_rows)
    _write_markdown_table(output_dir / "experiment_summary.md", summary_rows)
    write_json(output_dir / "experiment_runs.json", {"runs": run_rows})
    write_json(output_dir / "experiment_summary.json", {"summary": summary_rows})
    print(f"Exported {len(run_rows)} run rows and {len(summary_rows)} experiment summaries to {output_dir}")


if __name__ == "__main__":
    main()
