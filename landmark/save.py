import json
import os
import torch


def save_model(model, name, save_dir="./models"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path


def load_model_state(model, path, device=None):
    state = torch.load(path, weights_only=True, map_location=device or "cpu")
    model.load_state_dict(state)
    return model


def save_run_metrics(run_name, metrics, eval_results, config, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}.json")
    data = {
        "config": config,
        "training": {
            "train_loss": metrics["train_loss"],
            "val_loss": metrics.get("val_loss", []),
            "val_accuracy": metrics.get("val_accuracy", []),
            "learning_rates": metrics["learning_rates"],
            "epoch_times": metrics["epoch_times"],
            "total_time": metrics["total_time"],
            "best_val_loss": metrics.get("best_val_loss"),
        },
        "evaluation": {
            "overall_accuracy": eval_results["overall_accuracy"],
            "top_k_accuracy": eval_results.get("top_k_accuracy", {}),
            "per_class_accuracy": eval_results["per_class_accuracy"],
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Metrics saved to {path}")
    return path


def load_run_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def load_all_results(results_dir="./results"):
    results = {}
    if not os.path.exists(results_dir):
        return results
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".json"):
            results[filename[:-5]] = load_run_metrics(os.path.join(results_dir, filename))
    return results
