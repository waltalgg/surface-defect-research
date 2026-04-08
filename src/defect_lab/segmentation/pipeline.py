from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from ..utils import ensure_dir, set_seed, write_json
from .data import create_segmentation_dataloaders, estimate_pos_weight
from .losses import BCEDiceLoss
from .metrics import dice_coefficient, iou_score
from .model import SmallUNet


def run_segmentation_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    sample_count = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += float(dice_coefficient(logits, masks).item()) * batch_size
        total_iou += float(iou_score(logits, masks).item()) * batch_size
        sample_count += batch_size

    denom = max(1, sample_count)
    return total_loss / denom, total_dice / denom, total_iou / denom


def run_segmentation_training(config: Config) -> None:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    set_seed(int(experiment_cfg["seed"]))
    output_dir = ensure_dir(experiment_cfg["output_dir"])
    loaders = create_segmentation_dataloaders(config)
    device = torch.device(training_cfg["device"])
    model = SmallUNet(
        base_channels=int(model_cfg.get("base_channels", 48)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    pos_weight = torch.tensor([estimate_pos_weight(config)], dtype=torch.float32, device=device)
    criterion = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=float(training_cfg.get("bce_weight", 0.45)),
        dice_weight=float(training_cfg.get("dice_weight", 0.55)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_cfg["epochs"]))

    best_dice = -1.0
    history: list[dict] = []
    best_path = Path(output_dir) / "best.pt"

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss, train_dice, train_iou = run_segmentation_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_dice, val_iou = run_segmentation_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_dice": train_dice,
                "val_dice": val_dice,
                "train_iou": train_iou,
                "val_iou": val_iou,
            }
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model_state": model.state_dict(), "config": config.data}, best_path)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_dice={train_dice:.4f} val_dice={val_dice:.4f} "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    write_json(Path(output_dir) / "history.json", {"best_dice": best_dice, "epochs": history})
    print(f"Segmentation training complete. Best checkpoint saved to {best_path}")


def run_segmentation_evaluation(config: Config) -> None:
    checkpoint_path = Path(config["evaluation"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaders = create_segmentation_dataloaders(config)
    device = torch.device(config["training"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SmallUNet(
        base_channels=int(config["model"].get("base_channels", 48)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    pos_weight = torch.tensor([estimate_pos_weight(config)], dtype=torch.float32, device=device)
    criterion = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=float(config["training"].get("bce_weight", 0.45)),
        dice_weight=float(config["training"].get("dice_weight", 0.55)),
    )
    test_loss, test_dice, test_iou = run_segmentation_epoch(model, loaders["test"], criterion, device)
    report = {
        "checkpoint": str(checkpoint_path),
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "config": checkpoint.get("config", {}),
    }
    report_path = checkpoint_path.parent / "test_metrics.json"
    write_json(report_path, report)
    print(f"Saved segmentation evaluation metrics to {report_path}")
    print(report)
