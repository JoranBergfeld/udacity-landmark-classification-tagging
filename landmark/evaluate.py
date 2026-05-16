import torch
import numpy as np


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
