from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .data import create_dataloaders
from .engine import run_epoch
from .metrics import classification_metrics
from .model import build_model
from .utils import write_json


def run_evaluation(config: Config) -> None:
    checkpoint_path = Path(config["evaluation"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaders, classes = create_dataloaders(config)
    device = torch.device(config["training"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        name=config["model"]["name"],
        num_classes=len(classes),
        dropout=float(config["model"]["dropout"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(config["training"].get("label_smoothing", 0.0)))
    test_loss, test_preds, test_targets = run_epoch(model, loaders["test"], criterion, device)
    metrics = classification_metrics(test_preds, test_targets, len(classes))

    report = {
        "checkpoint": str(checkpoint_path),
        "classes": classes,
        "test_loss": test_loss,
        "metrics": metrics,
        "config": checkpoint.get("config", {}),
        "manifest_metadata": checkpoint.get("manifest_metadata", {}),
    }
    report_path = checkpoint_path.parent / "test_metrics.json"
    write_json(report_path, report)
    print(f"Saved evaluation metrics to {report_path}")
    print(report)
