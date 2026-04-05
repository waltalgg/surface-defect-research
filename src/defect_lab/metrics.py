from __future__ import annotations


def classification_metrics(predictions: list[int], targets: list[int], num_classes: int) -> dict[str, float | list[list[int]]]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, pred in zip(targets, predictions, strict=False):
        confusion[target][pred] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[idx][idx] for idx in range(num_classes))
    accuracy = correct / total if total else 0.0

    precisions = []
    recalls = []
    f1_scores = []

    for class_idx in range(num_classes):
        tp = confusion[class_idx][class_idx]
        fp = sum(confusion[row][class_idx] for row in range(num_classes) if row != class_idx)
        fn = sum(confusion[class_idx][col] for col in range(num_classes) if col != class_idx)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = sum(precisions) / num_classes if num_classes else 0.0
    macro_recall = sum(recalls) / num_classes if num_classes else 0.0
    macro_f1 = sum(f1_scores) / num_classes if num_classes else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "confusion_matrix": confusion,
    }
