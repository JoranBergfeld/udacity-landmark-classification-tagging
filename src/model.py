from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mrl import MRLClassifier, resolve_granularities


# ---------------------------------------------------------------------------
# Residual building block.
#
# A skip connection adds the block's input back onto its output. 
# This lets gradients flow straight through a deep stack, which is what makes a from-scratch network this deep actually trainable.
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # When the block changes the channel count or spatial size, the
        # shortcut needs a 1x1 conv so the dimensions line up for the add.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.3,
        mrl_granularities: Optional[Union[str, Sequence[int]]] = None,
        mrl_mode: str = "mrl-e",
    ) -> None:
        """The rubric from-scratch model, optionally with a Matryoshka head.

        When ``mrl_granularities`` is None the architecture is byte-identical
        to the original: a single ``nn.Linear(512, num_classes)`` after the
        global pool. When MRL is opted into, that final Linear is swapped for
        an :class:`MRLClassifier`, which still returns a regular
        ``[batch, num_classes]`` logits tensor at the full feature width but
        also exposes per-prefix logits via its ``last_per_prefix_logits``
        side channel so the training loop can compute the MRL loss.
        """
        super().__init__()

        feature_dim = 512

        # Stem: 224 -> 112 (stride-2 conv) -> 56 (max pool).
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Four residual stages. The first block of each later stage uses
        # stride 2 to halve the spatial size: 56 -> 56 -> 28 -> 14 -> 7.
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

        if mrl_granularities is None:
            self.mrl_head: Optional[MRLClassifier] = None
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            resolved = resolve_granularities(mrl_granularities, feature_dim)
            self.mrl_head = MRLClassifier(feature_dim, num_classes, resolved, mode=mrl_mode)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                self.mrl_head,
            )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Model registry.
#
# The original pipeline kept a registry of from-scratch architectures so the
# experiment matrix could sweep over them. `MyModel` is the canonical
# from-scratch model the notebook uses, `ScratchResNet` is a deeper variant
# (three blocks per stage) kept for the matrix.
# ---------------------------------------------------------------------------

class ScratchResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.3,
        mrl_granularities: Optional[Union[str, Sequence[int]]] = None,
        mrl_mode: str = "mrl-e",
    ) -> None:
        """Deeper from-scratch ResNet variant, optionally with a Matryoshka head.

        Same MRL contract as :class:`MyModel`: leave ``mrl_granularities``
        unset to preserve the original behavior, or pass a schedule (or
        ``"auto"``) to swap the final ``nn.Linear`` for an
        :class:`MRLClassifier` and have the model expose ``.mrl_head``.
        """
        super().__init__()

        feature_dim = 512

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        if mrl_granularities is None:
            self.mrl_head: Optional[MRLClassifier] = None
            self.fc: nn.Module = nn.Linear(feature_dim, num_classes)
        else:
            resolved = resolve_granularities(mrl_granularities, feature_dim)
            self.mrl_head = MRLClassifier(feature_dim, num_classes, resolved, mode=mrl_mode)
            # The MRLClassifier returns the full-prefix logits as a Tensor,
            # so it is API-compatible with the original ``self.fc`` Linear.
            self.fc = self.mrl_head

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


MODEL_REGISTRY = {
    "scratch_cnn": MyModel,
    "scratch_resnet": ScratchResNet,
}


def get_scratch_model(
    name: str = "scratch_cnn",
    num_classes: int = 1000,
    dropout: float = 0.3,
    mrl_granularities: Optional[Union[str, Sequence[int]]] = None,
    mrl_mode: str = "mrl-e",
):
    """Build a from-scratch model, optionally with an MRL head.

    The MRL kwargs flow through unchanged so the experiment runner can opt
    a single run into MRL without disturbing other runs in the same matrix.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        num_classes=num_classes,
        dropout=dropout,
        mrl_granularities=mrl_granularities,
        mrl_mode=mrl_mode,
    )


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"


def test_model_construction_with_mrl(data_loaders):
    """An MRL-enabled MyModel still returns full-prefix logits and exposes per-prefix logits.

    The point is the side-channel contract: forward returns the same shape
    as the vanilla head, but the model now also surfaces ``mrl_head`` with
    one entry per requested granularity stored in its side channel.
    """
    granularities = [8, 32, 128, 512]
    model = MyModel(
        num_classes=23,
        dropout=0.3,
        mrl_granularities=granularities,
        mrl_mode="mrl-e",
    )

    assert model.mrl_head is not None
    assert model.mrl_head.granularities == granularities

    dataiter = iter(data_loaders["train"])
    images, _ = next(dataiter)

    out = model(images)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([2, 23])

    per_prefix = model.mrl_head.last_per_prefix_logits
    assert set(per_prefix.keys()) == set(granularities)
    for m in granularities:
        assert per_prefix[m].shape == torch.Size([2, 23])
