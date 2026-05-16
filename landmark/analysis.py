import numpy as np
import torch


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


def misclassified_samples(model, test_loader, device=None, n=25):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    misclassified = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predictions = outputs.max(1)
            predictions = predictions.cpu()
            for i in range(len(labels)):
                if predictions[i] != labels[i] and len(misclassified) < n:
                    misclassified.append({
                        "image": images[i],
                        "predicted": int(predictions[i]),
                        "true_label": int(labels[i]),
                    })
            if len(misclassified) >= n:
                break
    return misclassified
