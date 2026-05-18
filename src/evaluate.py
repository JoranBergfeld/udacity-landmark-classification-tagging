"""
Evaluation helpers for the experiment-matrix pipeline.

Kept conceptually identical to the original `landmark/` package: top-k and
per-class accuracy plus confusion-matrix / per-class reporting. These build on
top of the rubric `src` API and are imported by `experiment.py`, they are not
part of the rubric-pinned files.
"""
import numpy as np
import torch


def evaluate_model(model, test_loader, classes, device=None, top_k=(1, 3, 5)):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    top_k_correct = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_device = labels.to(device)
            outputs = model(images)
            _, top_k_preds = outputs.topk(max(top_k), dim=1)
            for k in top_k:
                correct = (top_k_preds[:, :k] == labels_device.unsqueeze(1)).any(dim=1).sum().item()
                top_k_correct[k] += correct
            _, predictions = outputs.max(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            total += labels.size(0)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    overall_accuracy = (all_predictions == all_labels).mean() * 100
    top_k_accuracy = {f"top_{k}": 100.0 * top_k_correct[k] / max(1, total) for k in top_k}

    per_class_accuracy = {}
    for index, class_name in enumerate(classes):
        mask = all_labels == index
        if mask.sum() > 0:
            per_class_accuracy[class_name] = float((all_predictions[mask] == index).mean() * 100)

    return {
        "overall_accuracy": float(overall_accuracy),
        "top_k_accuracy": top_k_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "predictions": all_predictions,
        "ground_truth": all_labels,
    }


def confusion_matrix(predictions, ground_truth, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(ground_truth, predictions):
        matrix[int(true_label)][int(predicted_label)] += 1
    return matrix


def per_class_report(predictions, ground_truth, classes):
    report = {}
    for index, class_name in enumerate(classes):
        true_positives = ((predictions == index) & (ground_truth == index)).sum()
        false_positives = ((predictions == index) & (ground_truth != index)).sum()
        false_negatives = ((predictions != index) & (ground_truth == index)).sum()
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return report
