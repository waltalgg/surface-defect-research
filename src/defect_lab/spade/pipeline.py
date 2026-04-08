from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from ..config import Config
from ..utils import ensure_dir, set_seed, write_json
from .data import create_spade_dataloaders
from .model import PatchDiscriminator, SpadeGenerator


def _denorm(image: torch.Tensor) -> torch.Tensor:
    return ((image.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def _gan_loss(prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return F.binary_cross_entropy_with_logits(prediction, target)


def _save_spade_preview(generator: SpadeGenerator, loader, device: torch.device, path: Path) -> None:
    generator.eval()
    with torch.inference_mode():
        masks, real_images = next(iter(loader))
        masks = masks[:6].to(device)
        real_images = real_images[:6].to(device)
        fake_images = generator(masks)
        preview = torch.cat([masks.repeat(1, 3, 1, 1), _denorm(real_images), _denorm(fake_images)], dim=0)
        grid = make_grid(preview, nrow=6)
        save_image(grid, path)


def run_spade_training(config: Config) -> None:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    set_seed(int(experiment_cfg["seed"]))
    output_dir = ensure_dir(experiment_cfg["output_dir"])
    preview_dir = ensure_dir(Path(output_dir) / "previews")
    loaders = create_spade_dataloaders(config)
    device = torch.device(training_cfg["device"])

    generator = SpadeGenerator(base_channels=int(model_cfg.get("base_channels", 64))).to(device)
    discriminator = PatchDiscriminator(base_channels=int(model_cfg.get("disc_channels", 64))).to(device)

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=(0.5, 0.999),
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(training_cfg.get("disc_learning_rate", training_cfg["learning_rate"])),
        betas=(0.5, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=int(training_cfg["epochs"]))

    adv_weight = float(training_cfg.get("adv_weight", 1.0))
    l1_weight = float(training_cfg.get("l1_weight", 50.0))
    best_val = float("inf")
    history: list[dict] = []
    best_path = Path(output_dir) / "best.pt"

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        generator.train()
        discriminator.train()
        train_g_loss = 0.0
        train_d_loss = 0.0
        train_l1 = 0.0
        sample_count = 0

        for masks, real_images in loaders["train"]:
            masks = masks.to(device)
            real_images = real_images.to(device)
            batch_size = masks.size(0)

            fake_images = generator(masks)

            d_optimizer.zero_grad()
            pred_real = discriminator(real_images, masks)
            pred_fake = discriminator(fake_images.detach(), masks)
            d_loss = 0.5 * (_gan_loss(pred_real, True) + _gan_loss(pred_fake, False))
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            pred_fake = discriminator(fake_images, masks)
            g_adv = _gan_loss(pred_fake, True)
            g_l1 = F.l1_loss(fake_images, real_images)
            g_loss = adv_weight * g_adv + l1_weight * g_l1
            g_loss.backward()
            g_optimizer.step()

            train_g_loss += float(g_loss.item()) * batch_size
            train_d_loss += float(d_loss.item()) * batch_size
            train_l1 += float(g_l1.item()) * batch_size
            sample_count += batch_size

        scheduler.step()

        generator.eval()
        val_l1 = 0.0
        val_count = 0
        with torch.inference_mode():
            for masks, real_images in loaders["val"]:
                masks = masks.to(device)
                real_images = real_images.to(device)
                fake_images = generator(masks)
                batch = masks.size(0)
                val_l1 += float(F.l1_loss(fake_images, real_images).item()) * batch
                val_count += batch

        train_g_loss /= max(1, sample_count)
        train_d_loss /= max(1, sample_count)
        train_l1 /= max(1, sample_count)
        val_l1 /= max(1, val_count)

        history.append(
            {
                "epoch": epoch,
                "train_g_loss": train_g_loss,
                "train_d_loss": train_d_loss,
                "train_l1": train_l1,
                "val_l1": val_l1,
            }
        )

        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"generator_state": generator.state_dict(), "config": config.data}, best_path)
            _save_spade_preview(generator, loaders["val"], device, preview_dir / "best_preview.png")

        if epoch == 1 or epoch % int(training_cfg.get("preview_every", 5)) == 0:
            _save_spade_preview(generator, loaders["val"], device, preview_dir / f"epoch_{epoch:03d}.png")

        print(
            f"Epoch {epoch:03d} | "
            f"train_g={train_g_loss:.4f} train_d={train_d_loss:.4f} "
            f"train_l1={train_l1:.4f} val_l1={val_l1:.4f} "
            f"lr={g_optimizer.param_groups[0]['lr']:.6f}"
        )

    write_json(Path(output_dir) / "history.json", {"best_val_l1": best_val, "epochs": history})
    print(f"SPADE training complete. Best checkpoint saved to {best_path}")


def run_spade_evaluation(config: Config) -> None:
    checkpoint_path = Path(config["evaluation"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaders = create_spade_dataloaders(config)
    device = torch.device(config["training"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = SpadeGenerator(base_channels=int(config["model"].get("base_channels", 64))).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()

    test_l1 = 0.0
    sample_count = 0
    with torch.inference_mode():
        for masks, real_images in loaders["test"]:
            masks = masks.to(device)
            real_images = real_images.to(device)
            fake_images = generator(masks)
            batch_size = masks.size(0)
            test_l1 += float(F.l1_loss(fake_images, real_images).item()) * batch_size
            sample_count += batch_size

    report = {
        "checkpoint": str(checkpoint_path),
        "test_l1": test_l1 / max(1, sample_count),
        "config": checkpoint.get("config", {}),
    }
    report_path = checkpoint_path.parent / "test_metrics.json"
    write_json(report_path, report)
    print(f"Saved SPADE evaluation metrics to {report_path}")
    print(report)


def generate_spade_samples(config: Config, split: str = "test", limit: int = 8) -> None:
    checkpoint_path = Path(config["evaluation"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaders = create_spade_dataloaders(config)
    device = torch.device(config["training"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = SpadeGenerator(base_channels=int(config["model"].get("base_channels", 64))).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()

    sample_dir = ensure_dir(Path(config["experiment"]["output_dir"]) / "samples")
    with torch.inference_mode():
        masks, real_images = next(iter(loaders[split]))
        masks = masks[:limit].to(device)
        real_images = real_images[:limit].to(device)
        fake_images = generator(masks)
        preview = torch.cat([masks.repeat(1, 3, 1, 1), _denorm(real_images), _denorm(fake_images)], dim=0)
        save_image(make_grid(preview, nrow=limit), sample_dir / f"{split}_samples.png")
        save_image(make_grid(_denorm(fake_images), nrow=limit), sample_dir / f"{split}_fake_only.png")
    print(f"Saved SPADE samples to {sample_dir}")
