import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


# Backbones the experiment-matrix pipeline knows how to transfer-learn from.
# `resnet18` stays the rubric default, the rest preserve the multi-backbone
# sweep concept from the original pipeline.
TRANSFER_BACKBONES = [
    "resnet18",
    "resnet50",
    "convnext_tiny",
    "efficientnet_b0",
    "vit_b_16",
]


def _replace_classifier(model, n_classes):
    """
    Freeze-friendly head replacement that works across torchvision families.

    Different backbones expose their final classifier under a different
    attribute (`fc` for ResNet, `classifier` for ConvNeXt / EfficientNet /
    VGG, `heads` for ViT). We locate it, read its input feature count, and
    swap in a fresh Linear with `n_classes` outputs.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        return model

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, n_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, n_classes)
            return model
        # Sequential classifier: replace the last Linear in place.
        for index in range(len(classifier) - 1, -1, -1):
            if isinstance(classifier[index], nn.Linear):
                classifier[index] = nn.Linear(classifier[index].in_features, n_classes)
                return model

    raise ValueError("Could not locate a Linear classifier head to replace on this backbone")


def get_model_transfer_learning(model_name="resnet18", n_classes=50):

    # Get the requested architecture
    if hasattr(models, model_name):

        model_transfer = getattr(models, model_name)(pretrained=True)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model so only the new head trains
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes.
    # The input feature count is read from the backbone, the output count is
    # the requested n_classes.
    model_transfer = _replace_classifier(model_transfer, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
