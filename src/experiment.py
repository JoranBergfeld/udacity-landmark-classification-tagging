"""
Experiment matrix for landmark classification.

This module preserves the conceptual pipeline of the original `landmark/`
package: pick any subset of models / augmentations / optimizers / schedulers
and the cartesian product is trained, evaluated and persisted. It is built
entirely on the rubric `src` API (`data`, `model`, `transfer`, `optimization`)
plus the `evaluate`/`save` helpers, and is importable from the notebooks so
the experiment-tracking table can be reproduced without rerunning anything.

Nothing here is on the rubric-pinned path: the notebook's required `optimize`
(src/train.py) still uses ReduceLROnPlateau exactly as the rubric demands.
This runner has its own metrics-collecting loop so it can record the per-epoch
history the JSON schema in `save.py` expects.
"""
import argparse
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.data import AUGMENTATION_REGISTRY, get_data_loaders
from src.model import MODEL_REGISTRY, get_scratch_model
from src.transfer import TRANSFER_BACKBONES, get_model_transfer_learning
from src.optimization import get_loss, get_optimizer, create_scheduler, PLATEAU_SCHEDULERS
from src.evaluate import evaluate_model
from src.save import save_run_metrics
from src.mrl import MRL_MODES, MRLLoss, parse_mrl_granularities_argument


# Map experiment run-name keys to builders, keeping the same naming the
# existing results/*.json files use (scratch_cnn, tl_resnet50, ...).
def _model_specs():
    specs = {}
    for name in MODEL_REGISTRY:
        specs[name] = ("scratch", name)
    for backbone in TRANSFER_BACKBONES:
        specs[f"tl_{backbone.split('_')[0]}"] = ("transfer", backbone)
    return specs


MODEL_SPECS = _model_specs()
OPTIMIZER_NAMES = ["adam", "adamw", "sgd"]


def make_run_name(model_key, augmentation, optimizer_name, scheduler_name, epochs):
    return f"{model_key}__{augmentation}__{optimizer_name}__{scheduler_name}__{epochs}ep"


def default_learning_rate(optimizer_name):
    """Adam-family optimizers want a smaller step than SGD."""
    return 1e-3 if optimizer_name in ("adam", "adamw") else 1e-2


def build_model(model_key, num_classes, mrl_granularities=None, mrl_mode="mrl-e"):
    """Build the requested model, threading MRL kwargs to scratch backbones only.

    Transfer-learning backbones intentionally do not support MRL yet -- the
    head-replacement helper in ``src/transfer.py`` is family-aware and
    adapting it is a separate change. If MRL is requested for a transfer
    model we print a warning and fall back to the vanilla head so the run
    still completes.
    """
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_SPECS)}")
    kind, name = MODEL_SPECS[model_key]
    if kind == "scratch":
        return get_scratch_model(
            name,
            num_classes=num_classes,
            dropout=0.5,
            mrl_granularities=mrl_granularities,
            mrl_mode=mrl_mode,
        )
    if mrl_granularities is not None:
        print(
            f"Warning: MRL is not yet wired up for transfer models ({model_key}); "
            "falling back to the vanilla classifier head."
        )
    return get_model_transfer_learning(name, n_classes=num_classes)


def _validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predictions = outputs.max(1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), 100.0 * correct / max(1, total)


def train_with_metrics(model, train_loader, val_loader, epochs, optimizer, scheduler, scheduler_name, device, checkpoint_path=None):
    """Compact training loop that records the per-epoch history save.py expects.

    Auto-detects an MRL head on the model: if ``model.mrl_head`` is set,
    training uses :class:`MRLLoss` (summed CE across all nested prefixes).
    Validation loss is always computed with plain ``nn.CrossEntropyLoss`` on
    the full-prefix logits so the reported ``val_loss`` stays directly
    comparable between MRL and non-MRL runs.
    """
    model = model.to(device)
    eval_criterion = nn.CrossEntropyLoss()
    mrl_head = getattr(model, "mrl_head", None)
    if mrl_head is not None:
        criterion = MRLLoss(mrl_head)
        print(
            f"MRL training enabled (mode={mrl_head.mode}, granularities={mrl_head.granularities})"
        )
    else:
        criterion = eval_criterion

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    is_plateau = scheduler_name in PLATEAU_SCHEDULERS

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_accuracy = _validate(model, val_loader, eval_criterion, device)

        if scheduler is not None:
            if is_plateau:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["learning_rates"].append(current_learning_rate)
        metrics["epoch_times"].append(epoch_time)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)

        marker = " *" if improved else ""
        print(f"Epoch {epoch + 1}/{epochs} - train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, val_acc {val_accuracy:.2f}%, lr {current_learning_rate:.2e}, {epoch_time:.1f}s{marker}")

    metrics["total_time"] = sum(metrics["epoch_times"])
    metrics["best_val_loss"] = best_val_loss

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return metrics


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training transfer models on CPU will be very slow).")
    return device


def _evaluate_per_granularity(model, test_loader, device):
    """Compute top-1 accuracy at every MRL granularity for the test set.

    Re-iterates the test loader to read the side-channel ``last_per_prefix_logits``
    populated by the model's forward at each batch. Returns a dict keyed by
    the granularity stringified for JSON compatibility, with float percentages.
    """
    mrl_head = model.mrl_head
    correct = {m: 0 for m in mrl_head.granularities}
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_device = labels.to(device)
            _ = model(images)
            for m, logits in mrl_head.last_per_prefix_logits.items():
                preds = logits.argmax(dim=1)
                correct[m] += (preds == labels_device).sum().item()
            total += labels.size(0)
    return {str(m): 100.0 * correct[m] / max(1, total) for m in mrl_head.granularities}


def run_single(
    model_key,
    augmentation,
    optimizer_name,
    scheduler_name,
    epochs,
    batch_size,
    save_dir,
    results_dir,
    device,
    num_workers=2,
    learning_rate=None,
    mrl_granularities=None,
    mrl_mode="mrl-e",
):

    run_name = make_run_name(model_key, augmentation, optimizer_name, scheduler_name, epochs)
    if mrl_granularities is not None:
        run_name = f"{run_name}__mrl-{mrl_mode}"
    print(f"\nRun: {run_name}")

    if learning_rate is None:
        learning_rate = default_learning_rate(optimizer_name)

    data_loaders = get_data_loaders(batch_size=batch_size, num_workers=num_workers, augmentation=augmentation)
    classes = data_loaders["train"].dataset.classes
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    model = build_model(model_key, num_classes, mrl_granularities=mrl_granularities, mrl_mode=mrl_mode)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = get_optimizer(model, optimizer=optimizer_name, learning_rate=learning_rate, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, scheduler_name, epochs=epochs)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{run_name}.pt")
    metrics = train_with_metrics(model, data_loaders["train"], data_loaders["valid"], epochs, optimizer, scheduler, scheduler_name, device, checkpoint_path=checkpoint_path)

    eval_results = evaluate_model(model, data_loaders["test"], classes=classes, device=device)
    accuracy = eval_results["overall_accuracy"]
    print(f"Test accuracy: {accuracy:.2f}%")

    mrl_head = getattr(model, "mrl_head", None)
    if mrl_head is not None:
        per_g = _evaluate_per_granularity(model, data_loaders["test"], device)
        print("Per-granularity test accuracy:")
        for m_str, acc in per_g.items():
            print(f"  m={m_str}: {acc:.2f}%")
        eval_results["per_granularity_accuracy"] = per_g

    config = {
        "model": model_key,
        "augmentation": augmentation,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    if mrl_head is not None:
        config["mrl"] = {
            "mode": mrl_head.mode,
            "granularities": list(mrl_head.granularities),
        }
    save_run_metrics(run_name, metrics, eval_results, config, save_dir=results_dir)

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_name, accuracy


def run_matrix(
    models,
    augmentations,
    optimizers,
    schedulers,
    epochs=20,
    batch_size=32,
    save_dir="./models",
    results_dir="./results",
    num_workers=2,
    learning_rate=None,
    mrl_granularities=None,
    mrl_mode="mrl-e",
):
    """Train + evaluate the cartesian product. Importable from the notebooks.

    ``mrl_granularities`` is shared across the matrix: pass it in to opt the
    whole sweep into the same MRL configuration, or leave it ``None`` for
    the standard vanilla-head behavior.
    """
    device = get_device()
    combinations = list(itertools.product(models, augmentations, optimizers, schedulers))

    print(f"\nPlanned runs: {len(combinations)}")

    summary = []
    for i, (model_key, augmentation, optimizer_name, scheduler_name) in enumerate(combinations, 1):
        print(f"\nRun {i}/{len(combinations)}")
        try:
            run_name, accuracy = run_single(
                model_key,
                augmentation,
                optimizer_name,
                scheduler_name,
                epochs,
                batch_size,
                save_dir,
                results_dir,
                device,
                num_workers=num_workers,
                learning_rate=learning_rate,
                mrl_granularities=mrl_granularities,
                mrl_mode=mrl_mode,
            )
            summary.append((run_name, accuracy))
        except Exception as exc:
            print(f"Run failed: {exc!r}")
            failed_name = make_run_name(model_key, augmentation, optimizer_name, scheduler_name, epochs)
            if mrl_granularities is not None:
                failed_name = f"{failed_name}__mrl-{mrl_mode}"
            summary.append((failed_name, float("nan")))

    summary.sort(key=lambda pair: pair[1] if pair[1] == pair[1] else -1, reverse=True)

    print("\nResults")
    for run_name, accuracy in summary:
        print(f"{accuracy:.2f}% {run_name}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate landmark classifiers across an experiment matrix")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--augmentations", nargs="+", choices=list(AUGMENTATION_REGISTRY), default=list(AUGMENTATION_REGISTRY))
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_NAMES, default=["adamw"])
    parser.add_argument("--schedulers", nargs="+", default=["cosine", "warmup-cosine"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument(
        "--mrl-granularities",
        type=str,
        default=None,
        help=(
            "Opt into Matryoshka Representation Learning. "
            "'none' (or omitted) disables it. 'auto' uses the default log-spaced "
            "schedule (8,16,32,...,feature_dim). Or pass an explicit list like "
            "'8,32,128,512'. Applies to scratch backbones only."
        ),
    )
    parser.add_argument(
        "--mrl-mode",
        type=str,
        default="mrl-e",
        choices=list(MRL_MODES),
        help="MRL head variant: 'mrl-e' (shared, default) or 'mrl' (independent heads).",
    )
    arguments = parser.parse_args()

    mrl_granularities = parse_mrl_granularities_argument(arguments.mrl_granularities)

    run_matrix(
        arguments.models,
        arguments.augmentations,
        arguments.optimizers,
        arguments.schedulers,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        save_dir=arguments.save_dir,
        results_dir=arguments.results_dir,
        num_workers=arguments.num_workers,
        learning_rate=arguments.learning_rate,
        mrl_granularities=mrl_granularities,
        mrl_mode=arguments.mrl_mode,
    )


if __name__ == "__main__":
    main()
