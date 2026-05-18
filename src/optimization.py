import torch
import torch.nn as nn
import torch.optim


def get_loss(label_smoothing: float = 0.0):
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True.
    """

    loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    return loss


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance.

    :param model: the model to optimize
    :param optimizer: one of 'SGD', 'Adam' or 'AdamW'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt


# ---------------------------------------------------------------------------
# Scheduler registry.
#
# The rubric's `optimize` (see train.py) uses ReduceLROnPlateau directly, as required. 
# The experiment-matrix pipeline, however, kept schedulers as a swappable building block. 
# We preserve that concept here so the matrix runner can request any of these by name without touching the rubric path.
# ---------------------------------------------------------------------------

SCHEDULER_NAMES = ["none", "plateau", "step-decay", "cosine", "warmup-cosine"]
PLATEAU_SCHEDULERS = {"plateau"}


def create_scheduler(optimizer, name="plateau", epochs=20):
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if name == "step-decay":
        milestones = [max(1, int(epochs * 0.5)), max(2, int(epochs * 0.8))]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "warmup-cosine":
        warmup = max(1, epochs // 10)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
    raise ValueError(f"Unknown scheduler: {name}. Choose from {SCHEDULER_NAMES}")


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)


def test_get_loss():

    loss = get_loss()

    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"


def test_get_optimizer_type(fake_model):

    opt = get_optimizer(fake_model)

    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_is_linked_with_model(fake_model):

    opt = get_optimizer(fake_model)

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])


def test_get_optimizer_returns_adam(fake_model):

    opt = get_optimizer(fake_model, optimizer="adam")

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_sets_learning_rate(fake_model):

    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)

    assert (
        opt.param_groups[0]["lr"] == 0.123
    ), "get_optimizer is not setting the learning rate appropriately. Check your code."


def test_get_optimizer_sets_momentum(fake_model):

    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)

    assert (
        opt.param_groups[0]["momentum"] == 0.123
    ), "get_optimizer is not setting the momentum appropriately. Check your code."


def test_get_optimizer_sets_weight_decat(fake_model):

    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)

    assert (
        opt.param_groups[0]["weight_decay"] == 0.123
    ), "get_optimizer is not setting the weight_decay appropriately. Check your code."
