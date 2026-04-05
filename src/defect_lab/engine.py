from __future__ import annotations

import torch


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) else 0.0
    return avg_loss, predictions, targets
