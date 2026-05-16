import time
import torch
import torch.nn as nn

from landmark.augmentation import mixup_batch
from landmark.optim import PER_STEP_SCHEDULERS, PLATEAU_SCHEDULERS


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
        print(f"GPU memory: {properties.total_memory / 1024 ** 3:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"Using CPU ({torch.get_num_threads()} threads)")
        print("WARNING: training transfer models on CPU will be very slow.")

    return device


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


def train_model(model, train_loader, val_loader, epochs, optimizer, scheduler=None, scheduler_name="none", device=None, use_mixup=False, mixup_alpha=0.2, checkpoint_path=None):
    if epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {epochs}")
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    is_per_step = scheduler_name in PER_STEP_SCHEDULERS
    is_plateau = scheduler_name in PLATEAU_SCHEDULERS

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if use_mixup:
                images, labels_a, labels_b, lambda_ = mixup_batch(images, labels, alpha=mixup_alpha)
                outputs = model(images)
                loss = lambda_ * criterion(outputs, labels_a) + (1 - lambda_) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None and is_per_step:
                scheduler.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_accuracy = _validate(model, val_loader, criterion, device)

        if scheduler is not None and not is_per_step:
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
        print(f"Epoch {epoch + 1}/{epochs} — train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, val_acc {val_accuracy:.2f}%, lr {current_learning_rate:.2e}, {epoch_time:.1f}s{marker}")

    metrics["total_time"] = sum(metrics["epoch_times"])
    metrics["best_val_loss"] = best_val_loss

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))

    return metrics
