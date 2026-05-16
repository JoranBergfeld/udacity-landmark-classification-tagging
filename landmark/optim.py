import torch.optim as optim


def create_optimizer(model, name, learning_rate, weight_decay=1e-4):
    if learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {learning_rate}")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters — did you freeze everything?")
    if name == "adam":
        return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}. Choose from {OPTIMIZER_NAMES}")


def create_scheduler(optimizer, name, epochs, steps_per_epoch=None, learning_rate=None):
    if name == "none":
        return None
    if name == "step-decay":
        milestones = [max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "warmup-cosine":
        warmup = max(1, epochs // 10)
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        return optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
    if name == "onecycle":
        if steps_per_epoch is None or learning_rate is None:
            raise ValueError("onecycle scheduler needs steps_per_epoch and learning_rate")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    raise ValueError(f"Unknown scheduler: {name}. Choose from {SCHEDULER_NAMES}")


OPTIMIZER_NAMES = ["adam", "adamw", "sgd"]
SCHEDULER_NAMES = ["none", "step-decay", "cosine", "warmup-cosine", "onecycle", "plateau"]
PER_STEP_SCHEDULERS = {"onecycle"}
PLATEAU_SCHEDULERS = {"plateau"}
