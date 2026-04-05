from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .data import create_dataloaders
from .engine import run_epoch
from .metrics import classification_metrics
from .model import build_model
from .utils import ensure_dir, set_seed, write_json


def run_training(config: Config) -> None:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    set_seed(int(experiment_cfg["seed"]))
    output_dir = ensure_dir(experiment_cfg["output_dir"])
    loaders, classes = create_dataloaders(config)

    device = torch.device(training_cfg["device"])
    model = build_model(
        name=model_cfg["name"],
        num_classes=len(classes),
        dropout=float(model_cfg["dropout"]),
        pretrained=bool(model_cfg.get("pretrained", False)),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_cfg["epochs"]),
        )
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    history: list[dict] = []
    best_f1 = -1.0
    best_path = Path(output_dir) / "best.pt"

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss, train_preds, train_targets = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_preds, val_targets = run_epoch(model, loaders["val"], criterion, device)

        train_metrics = classification_metrics(train_preds, train_targets, len(classes))
        val_metrics = classification_metrics(val_preds, val_targets, len(classes))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "config": config.data,
                },
                best_path,
            )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    write_json(
        Path(output_dir) / "history.json",
        {
            "experiment": experiment_cfg["name"],
            "best_f1": best_f1,
            "epochs": history,
        },
    )
    print(f"Training complete. Best checkpoint saved to {best_path}")
