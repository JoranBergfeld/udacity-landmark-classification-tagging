"""
Runtime selector + experiment matrix for landmark classification.

Mirrors the cifar10 CLI: choose any subset of models / augmentations / optimizers / schedulers and the cartesian product is run. 
Defaults run the full sweep.
"""
import argparse
import itertools
import os
import torch

from landmark.augmentation import AUGMENTATION_REGISTRY, get_augmentation
from landmark.data import IMAGENET_MEAN, IMAGENET_STANDARD_DEVIATION, DEFAULT_IMAGE_SIZE, compute_mean_standard_deviation, get_eval_transforms, get_train_val_loaders, get_test_loader
from landmark.evaluate import evaluate_model
from landmark.models import MODEL_REGISTRY, TRANSFER_MODELS, get_model
from landmark.optim import OPTIMIZER_NAMES, SCHEDULER_NAMES, create_optimizer, create_scheduler
from landmark.save import save_run_metrics
from landmark.train import get_device, train_model


def make_run_name(model_name, augmentation, optimizer_name, scheduler_name, epochs):
    return f"{model_name}__{augmentation}__{optimizer_name}__{scheduler_name}__{epochs}ep"


def default_learning_rate(optimizer_name):
    """Adam-family optimizers want a smaller step than SGD."""
    return 1e-3 if optimizer_name in ("adam", "adamw") else 1e-2


def run_single(model_name, augmentation_name, optimizer_name, scheduler_name,
               epochs, batch_size, data_dir, save_dir, results_dir, device,
               mean, standard_deviation, image_size, num_workers, learning_rate=None,
               freeze_backbone=True):
    
    run_name = make_run_name(model_name, augmentation_name, optimizer_name, scheduler_name, epochs)
    print(f"\nRun: {run_name}")

    if learning_rate is None:
        learning_rate = default_learning_rate(optimizer_name)

    # Data
    train_transform = get_augmentation(augmentation_name, image_size=image_size, mean=mean, standard_deviation=standard_deviation)
    eval_transform = get_eval_transforms(image_size=image_size, mean=mean, standard_deviation=standard_deviation)
    train_loader, val_loader, classes = get_train_val_loaders(train_transform, eval_transform, batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)
    test_loader = get_test_loader(eval_transform, batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)
    
    print(f"Classes: {len(classes)} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    model = get_model(model_name, num_classes=len(classes), freeze_backbone=freeze_backbone) if model_name in TRANSFER_MODELS else get_model(model_name, num_classes=len(classes))
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Optim + scheduler
    optimizer = create_optimizer(model, optimizer_name, learning_rate)
    scheduler = create_scheduler(optimizer, scheduler_name, epochs, steps_per_epoch=len(train_loader), learning_rate=learning_rate)

    # Train
    use_mixup = augmentation_name == "mixup"
    checkpoint_path = os.path.join(save_dir, f"{run_name}.pt")
    os.makedirs(save_dir, exist_ok=True)
    metrics = train_model(model, train_loader, val_loader, epochs=epochs, optimizer=optimizer, scheduler=scheduler, scheduler_name=scheduler_name, device=device, use_mixup=use_mixup, checkpoint_path=checkpoint_path)

    # Evaluate
    eval_results = evaluate_model(model, test_loader, classes=classes, device=device)
    accuracy = eval_results["overall_accuracy"]
    top_k = eval_results["top_k_accuracy"]
    print(f"Test accuracy: {accuracy:.2f}% (top-3 {top_k['top_3']:.2f}%, top-5 {top_k['top_5']:.2f}%)")

    # Persist
    config = {
        "model": model_name, "augmentation": augmentation_name,
        "optimizer": optimizer_name, "scheduler": scheduler_name,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "image_size": image_size, "freeze_backbone": freeze_backbone,
        "trainable_parameters": trainable, "total_parameters": total,
    }
    save_run_metrics(run_name, metrics, eval_results, config, save_dir=results_dir)

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_name, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate landmark classifiers across an experiment matrix")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    parser.add_argument("--augmentations", nargs="+", choices=list(AUGMENTATION_REGISTRY), default=list(AUGMENTATION_REGISTRY))
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_NAMES, default=OPTIMIZER_NAMES)
    parser.add_argument("--schedulers", nargs="+", choices=SCHEDULER_NAMES, default=SCHEDULER_NAMES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override default LR (per-model defaults used otherwise)")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--compute-stats", action="store_true", help="Compute dataset mean/std from train images (defaults to ImageNet stats)")
    parser.add_argument("--no-freeze", action="store_true", help="Fine-tune transfer backbones instead of freezing")
    parser.add_argument("--preset", choices=["quick", "transfer", "full"], default=None, help="Override selections: quick = small smoke matrix; transfer = transfer models only; full = everything")
    arguments = parser.parse_args()

    if arguments.preset == "quick":
        arguments.models = ["scratch_cnn", "tl_resnet50"]
        arguments.augmentations = ["flip-crop"]
        arguments.optimizers = ["adam"]
        arguments.schedulers = ["cosine"]
        arguments.epochs = min(arguments.epochs, 3)
    elif arguments.preset == "transfer":
        arguments.models = sorted(TRANSFER_MODELS)
    # "full" keeps argparse defaults (all combinations).

    device = get_device()

    mean, standard_deviation = IMAGENET_MEAN, IMAGENET_STANDARD_DEVIATION
    if arguments.compute_stats:
        mean, standard_deviation = compute_mean_standard_deviation(arguments.data_dir, image_size=arguments.image_size)

    combinations = list(itertools.product(arguments.models, arguments.augmentations, arguments.optimizers, arguments.schedulers))

    print(f"\nPlanned runs: {len(combinations)}")
    print(f"Models: {arguments.models}")
    print(f"Augmentations: {arguments.augmentations}")
    print(f"Optimizers: {arguments.optimizers}")
    print(f"Schedulers: {arguments.schedulers}")
    print(f"Epochs {arguments.epochs}, batch_size {arguments.batch_size}, image_size {arguments.image_size}")

    summary = []
    for i, (model_name, augmentation, optimizer_name, scheduler_name) in enumerate(combinations, 1):
        print(f"\nRun {i}/{len(combinations)}")
        try:
            run_name, accuracy = run_single(model_name, augmentation, optimizer_name, scheduler_name, arguments.epochs, arguments.batch_size, arguments.data_dir, arguments.save_dir, arguments.results_dir, device, mean, standard_deviation, arguments.image_size, arguments.num_workers, learning_rate=arguments.learning_rate, freeze_backbone=not arguments.no_freeze)
            summary.append((run_name, accuracy))
        except Exception as exc:
            print(f"Run failed: {exc!r}")
            failed_name = make_run_name(model_name, augmentation, optimizer_name, scheduler_name, arguments.epochs)
            summary.append((failed_name, float("nan")))

    # nan sorts last so failed runs land at the bottom
    summary.sort(key=lambda pair: pair[1] if pair[1] == pair[1] else -1, reverse=True)

    print("\nResults")
    for run_name, accuracy in summary:
        print(f"{accuracy:.2f}% {run_name}")

    print(f"\nResults written to {arguments.results_dir}/, models to {arguments.save_dir}/")


if __name__ == "__main__":
    main()
