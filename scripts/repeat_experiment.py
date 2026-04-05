import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from defect_lab.config import Config, load_config
from defect_lab.dataset_prep import prepare_dataset
from defect_lab.evaluate import run_evaluation
from defect_lab.synthetic import generate_synthetic_dataset
from defect_lab.train import run_training
from defect_lab.utils import ensure_dir, write_json


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Repeat one experiment config across multiple seeds.")
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument("--seeds", required=True, nargs="+", type=int, help="Seed values to run.")
    return parser.parse_args()


def _experiment_name(base_name: str, seed: int) -> str:
    return f"{base_name}_seed_{seed}"


def _prepare_seed_config(base_config: Config, seed: int) -> Config:
    data = deepcopy(base_config.data)
    experiment_name = _experiment_name(data["experiment"]["name"], seed)
    data["experiment"]["seed"] = seed
    data["experiment"]["name"] = experiment_name
    data["experiment"]["output_dir"] = f"artifacts/runs/{experiment_name}"

    manifest_path = Path(data["dataset"]["manifest_path"])
    data["dataset"]["manifest_path"] = str(
        manifest_path.with_name(f"{manifest_path.stem}_seed_{seed}{manifest_path.suffix}").as_posix()
    )

    if data.get("synthetic", {}).get("enabled", False):
        synthetic_dir = Path(data["synthetic"]["output_dir"])
        data["synthetic"]["output_dir"] = str((synthetic_dir.parent / f"{synthetic_dir.name}_seed_{seed}").as_posix())

        merged_manifest = data["synthetic"].get("merged_manifest_path")
        if merged_manifest:
            merged_manifest_path = Path(merged_manifest)
            data["synthetic"]["merged_manifest_path"] = str(
                merged_manifest_path.with_name(
                    f"{merged_manifest_path.stem}_seed_{seed}{merged_manifest_path.suffix}"
                ).as_posix()
            )
            data["dataset"]["manifest_path"] = data["synthetic"]["merged_manifest_path"]

    data["evaluation"]["checkpoint_path"] = f"artifacts/runs/{experiment_name}/best.pt"
    return Config(data=data)


def _load_metrics(checkpoint_path: str) -> dict:
    metrics_path = Path(checkpoint_path).parent / "test_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _summarize(results: list[dict]) -> dict:
    metrics = ["accuracy", "precision", "recall", "f1", "test_loss"]
    summary = {"runs": results, "aggregate": {}}
    for metric in metrics:
        values = [run[metric] for run in results]
        summary["aggregate"][metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    summary_rows: list[dict] = []

    for seed in args.seeds:
        config = _prepare_seed_config(base_config, seed)
        print(f"Running seed {seed}: {config['experiment']['name']}")
        prepare_dataset(config)
        if config["synthetic"]["enabled"]:
            generate_synthetic_dataset(config)
        run_training(config)
        run_evaluation(config)
        metrics_payload = _load_metrics(config["evaluation"]["checkpoint_path"])
        summary_rows.append(
            {
                "seed": seed,
                "experiment": config["experiment"]["name"],
                "accuracy": metrics_payload["metrics"]["accuracy"],
                "precision": metrics_payload["metrics"]["precision"],
                "recall": metrics_payload["metrics"]["recall"],
                "f1": metrics_payload["metrics"]["f1"],
                "test_loss": metrics_payload["test_loss"],
            }
        )

    base_name = Path(args.config).stem
    summary_dir = ensure_dir("artifacts/runs/repeated")
    summary_path = summary_dir / f"{base_name}_summary.json"
    write_json(summary_path, _summarize(summary_rows))
    yaml_path = summary_dir / f"{base_name}_summary.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_summarize(summary_rows), fh, sort_keys=False)
    print(f"Saved repeated-run summary to {summary_path}")


if __name__ == "__main__":
    main()
