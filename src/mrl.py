"""
Matryoshka Representation Learning (MRL) head and loss.

Reference: Kusupati et al., "Matryoshka Representation Learning",
NeurIPS 2022. https://arxiv.org/abs/2205.13147

The idea: train one classifier such that the first m dimensions of the
penultimate feature vector are also a usable classifier, for several nested
m. After training you get an "elastic" embedding -- you can truncate it to a
smaller width and still classify reasonably -- without paying any inference
overhead, since the smaller heads are slices of the bigger one (in the
"mrl-e" / efficient variant) or independent heads run on the truncated
prefix (in the "mrl" / original-paper variant).

Two design choices in this file are worth calling out, because they keep
the rest of the training pipeline blissfully unaware of MRL:

1. ``MRLClassifier.forward`` returns a plain Tensor (the full-prefix
   logits), shape ``[batch, num_classes]``. That is what an existing model
   would have returned from its final ``nn.Linear``. So every call site
   that does ``output = model(x); output.argmax(1)`` keeps working without
   any branching for "is this MRL or not".

2. The per-prefix logits used for the MRL loss are stashed as a side-channel
   on the classifier (``self.last_per_prefix_logits``) during the forward,
   and ``MRLLoss`` reads them from there. That way ``MRLLoss(output,
   target)`` matches the standard ``nn.CrossEntropyLoss(output, target)``
   API even though it actually needs more than just ``output``. Training
   loops that already write ``loss(output, target)`` do not need to change.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


MRL_MODES = ("mrl-e", "mrl")


def default_mrl_granularities(feature_dim: int, min_dim: int = 8) -> List[int]:
    """Return a log-spaced doubling of widths from ``min_dim`` up to ``feature_dim``.

    For ``feature_dim=512`` this returns ``[8, 16, 32, 64, 128, 256, 512]``,
    the schedule used in the paper for d=512. For non-powers-of-two we still
    double, then terminate at ``feature_dim`` exactly so the largest
    granularity equals the full embedding width. If ``feature_dim`` is below
    ``min_dim`` we just return ``[feature_dim]``: there is nothing smaller
    to nest inside.
    """
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")
    if min_dim <= 0:
        raise ValueError(f"min_dim must be positive, got {min_dim}")
    if feature_dim < min_dim:
        return [feature_dim]
    out: List[int] = []
    m = min_dim
    while m < feature_dim:
        out.append(m)
        m *= 2
    out.append(feature_dim)
    return out


def validate_mrl_granularities(granularities: Sequence[int], feature_dim: int) -> List[int]:
    """Validate a user-provided list of nested widths and return it as a list of ints.

    Requirements (enforced):
      * non-empty
      * every entry is a positive int in ``(0, feature_dim]``
      * strictly ascending
      * the largest entry equals ``feature_dim`` -- the "outermost matryoshka"
        must be the full feature vector, otherwise the head would never see
        the full embedding during training and the convenient property that
        the standard ``model(x)`` output equals a vanilla classifier's output
        would be lost.
    """
    if len(granularities) == 0:
        raise ValueError("granularities must be non-empty")
    values = [int(m) for m in granularities]
    for m in values:
        if m <= 0 or m > feature_dim:
            raise ValueError(
                f"granularity {m} is outside (0, feature_dim={feature_dim}]"
            )
    for previous, current in zip(values, values[1:]):
        if current <= previous:
            raise ValueError(
                f"granularities must be strictly ascending, got {values}"
            )
    if values[-1] != feature_dim:
        raise ValueError(
            f"largest granularity ({values[-1]}) must equal feature_dim ({feature_dim})"
        )
    return values


class MRLClassifier(nn.Module):
    """A classification head that emits logits at several nested prefix widths.

    Two variants are supported:

    * ``"mrl-e"`` (efficient, default). A single ``nn.Linear(feature_dim,
      num_classes)`` is created. Logits at prefix width m are computed via
      ``F.linear(features[:, :m], shared.weight[:, :m], shared.bias)``,
      i.e. by slicing the columns of the shared weight matrix. Parameter
      count is identical to a vanilla classifier head, and at inference time
      with the full feature you literally just call the shared Linear.

    * ``"mrl"`` (original paper variant). An independent ``nn.Linear(m,
      num_classes)`` is allocated for every granularity m. Strictly more
      parameters than the vanilla head, but each prefix head is free to
      learn its own decision boundary unconstrained by the shared matrix.

    Forward returns the full-prefix logits as a plain Tensor (shape
    ``[batch, num_classes]``), so this module is a drop-in replacement for
    the final ``nn.Linear`` of a backbone. The per-prefix logits dict is
    written to ``self.last_per_prefix_logits`` as a side-channel so
    ``MRLLoss`` can pick it up without changing the call-site API.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        granularities: Sequence[int],
        mode: str = "mrl-e",
    ) -> None:
        super().__init__()
        if mode not in MRL_MODES:
            raise ValueError(f"mode must be one of {MRL_MODES}, got {mode!r}")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.mode = mode
        self.granularities: List[int] = validate_mrl_granularities(granularities, self.feature_dim)

        if mode == "mrl-e":
            self.shared = nn.Linear(self.feature_dim, self.num_classes)
        else:  # mode == "mrl"
            self.heads = nn.ModuleDict(
                {str(m): nn.Linear(m, self.num_classes) for m in self.granularities}
            )

        # Populated on every forward(). Kept as a regular Python dict (not a
        # buffer / not registered) because it holds activation tensors with
        # gradients, not persistent parameters.
        self.last_per_prefix_logits: Dict[int, torch.Tensor] = {}

    def _logits_at(self, features: torch.Tensor, m: int) -> torch.Tensor:
        if self.mode == "mrl-e":
            return F.linear(features[:, :m], self.shared.weight[:, :m], self.shared.bias)
        return self.heads[str(m)](features[:, :m])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(
                f"MRLClassifier expects a 2-D feature tensor [batch, feature_dim], got shape {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: got {features.shape[1]}, expected {self.feature_dim}"
            )

        per_prefix: Dict[int, torch.Tensor] = {}
        for m in self.granularities:
            per_prefix[m] = self._logits_at(features, m)

        self.last_per_prefix_logits = per_prefix
        return per_prefix[self.feature_dim]


class MRLLoss(nn.Module):
    """Weighted sum of cross-entropy losses over an MRLClassifier's nested heads.

    Bound to a specific ``MRLClassifier`` at construction time. ``forward``
    takes the same ``(output, target)`` arguments as ``nn.CrossEntropyLoss``
    so it is interchangeable with it at call sites, but ``output`` is
    deliberately ignored -- the per-prefix logits are read from
    ``self.classifier.last_per_prefix_logits``, which the classifier just
    populated during its forward pass.

    Default weights are 1.0 per granularity, matching the paper.
    """

    def __init__(
        self,
        classifier: MRLClassifier,
        weights: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(classifier, MRLClassifier):
            raise TypeError(
                f"MRLLoss expects an MRLClassifier, got {type(classifier).__name__}"
            )
        self.classifier = classifier
        if weights is None:
            self.weights: Dict[int, float] = {m: 1.0 for m in classifier.granularities}
        else:
            missing = [m for m in classifier.granularities if m not in weights]
            if missing:
                raise ValueError(
                    f"weights missing entries for granularities {missing}"
                )
            self.weights = {int(m): float(w) for m, w in weights.items()}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: ARG002 -- API compat with CrossEntropyLoss
        per_prefix = self.classifier.last_per_prefix_logits
        if not per_prefix:
            raise RuntimeError(
                "MRLLoss called but classifier.last_per_prefix_logits is empty -- "
                "did the model forward pass run, and is the loss bound to the right classifier?"
            )

        total: Optional[torch.Tensor] = None
        for m, logits in per_prefix.items():
            term = self.weights.get(m, 1.0) * F.cross_entropy(logits, target)
            total = term if total is None else total + term
        # ``per_prefix`` is non-empty by the guard above, so ``total`` is set.
        assert total is not None
        return total


def parse_mrl_granularities_argument(value: Optional[str]) -> Optional[object]:
    """Parse the ``--mrl-granularities`` CLI flag into a value the model can consume.

    Returns:
      * ``None`` if MRL is disabled (``None``/empty/"none"/"off"),
      * the string ``"auto"`` to defer resolution until the model knows its
        feature_dim,
      * a list of ints for an explicit schedule like ``"8,32,128,512"``.
    """
    if value is None:
        return None
    stripped = value.strip().lower()
    if stripped in ("", "none", "off"):
        return None
    if stripped == "auto":
        return "auto"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def resolve_granularities(
    requested: object,
    feature_dim: int,
    min_dim: int = 8,
) -> List[int]:
    """Turn whatever the user supplied into a validated list of granularities.

    Accepts:
      * ``"auto"`` -- use :func:`default_mrl_granularities`
      * an iterable of ints -- validate and return
    """
    if requested == "auto":
        return default_mrl_granularities(feature_dim, min_dim=min_dim)
    if isinstance(requested, Iterable) and not isinstance(requested, (str, bytes)):
        return validate_mrl_granularities(list(requested), feature_dim)
    raise ValueError(
        f"Cannot resolve mrl_granularities={requested!r} for feature_dim={feature_dim}"
    )


######################################################################################
#                                     TESTS
######################################################################################
import pytest


def test_default_mrl_granularities_power_of_two():
    assert default_mrl_granularities(512) == [8, 16, 32, 64, 128, 256, 512]


def test_default_mrl_granularities_non_power_of_two():
    # Doubling from 8 walks past 256; the schedule must still terminate at
    # feature_dim exactly, otherwise the largest matryoshka would not be the
    # full embedding.
    assert default_mrl_granularities(300) == [8, 16, 32, 64, 128, 256, 300]


def test_default_mrl_granularities_below_min():
    assert default_mrl_granularities(4, min_dim=8) == [4]


def test_validate_rejects_max_not_equal_feature_dim():
    with pytest.raises(ValueError):
        validate_mrl_granularities([8, 16, 32], feature_dim=512)


def test_validate_rejects_unsorted():
    with pytest.raises(ValueError):
        validate_mrl_granularities([8, 32, 16, 512], feature_dim=512)


def test_validate_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_mrl_granularities([8, 16, 1024], feature_dim=512)
    with pytest.raises(ValueError):
        validate_mrl_granularities([0, 8, 512], feature_dim=512)


def test_mrl_e_param_count_matches_vanilla_linear():
    classes = 23
    classifier = MRLClassifier(512, classes, [8, 16, 32, 64, 128, 256, 512], mode="mrl-e")
    vanilla = nn.Linear(512, classes)
    mrl_params = sum(p.numel() for p in classifier.parameters())
    vanilla_params = sum(p.numel() for p in vanilla.parameters())
    assert mrl_params == vanilla_params, (mrl_params, vanilla_params)


def test_mrl_independent_param_count_strictly_larger_than_vanilla():
    classes = 23
    granularities = [8, 16, 32, 64, 128, 256, 512]
    classifier = MRLClassifier(512, classes, granularities, mode="mrl")
    vanilla_params = sum(p.numel() for p in nn.Linear(512, classes).parameters())
    mrl_params = sum(p.numel() for p in classifier.parameters())
    assert mrl_params > vanilla_params


def test_forward_returns_full_prefix_and_populates_side_channel():
    batch, dim, classes = 4, 512, 23
    granularities = [8, 32, 128, 512]
    classifier = MRLClassifier(dim, classes, granularities, mode="mrl-e")
    features = torch.randn(batch, dim)

    output = classifier(features)
    assert output.shape == (batch, classes)
    # Full-prefix logits returned by forward must match the entry stored at
    # the largest granularity in the side-channel.
    assert torch.allclose(output, classifier.last_per_prefix_logits[dim])
    for m in granularities:
        assert m in classifier.last_per_prefix_logits
        assert classifier.last_per_prefix_logits[m].shape == (batch, classes)


def test_mrl_e_slice_equivalence():
    """MRL-E logits at prefix m must equal F.linear on the sliced weight matrix."""
    batch, dim, classes = 3, 64, 7
    granularities = [8, 16, 32, 64]
    classifier = MRLClassifier(dim, classes, granularities, mode="mrl-e")
    features = torch.randn(batch, dim)

    _ = classifier(features)
    for m in granularities:
        expected = F.linear(
            features[:, :m], classifier.shared.weight[:, :m], classifier.shared.bias
        )
        assert torch.allclose(classifier.last_per_prefix_logits[m], expected)


def test_mrl_loss_equals_sum_of_cross_entropies():
    batch, dim, classes = 5, 32, 4
    granularities = [8, 16, 32]
    classifier = MRLClassifier(dim, classes, granularities, mode="mrl-e")
    loss_fn = MRLLoss(classifier)

    features = torch.randn(batch, dim)
    target = torch.randint(0, classes, (batch,))

    output = classifier(features)
    actual = loss_fn(output, target)

    expected = sum(
        F.cross_entropy(classifier.last_per_prefix_logits[m], target)
        for m in granularities
    )
    assert torch.allclose(actual, expected)


def test_mrl_loss_backward_produces_gradients():
    batch, dim, classes = 5, 32, 4
    classifier = MRLClassifier(dim, classes, [8, 16, 32], mode="mrl-e")
    loss_fn = MRLLoss(classifier)

    features = torch.randn(batch, dim, requires_grad=True)
    target = torch.randint(0, classes, (batch,))

    output = classifier(features)
    loss = loss_fn(output, target)
    loss.backward()

    assert classifier.shared.weight.grad is not None
    assert classifier.shared.weight.grad.abs().sum().item() > 0
    assert features.grad is not None
    assert features.grad.abs().sum().item() > 0


def test_mrl_loss_raises_if_forward_never_ran():
    classifier = MRLClassifier(16, 3, [8, 16], mode="mrl-e")
    loss_fn = MRLLoss(classifier)
    # No model forward yet -> side channel is empty -> must raise.
    with pytest.raises(RuntimeError):
        loss_fn(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))


def test_mrl_loss_rejects_non_classifier():
    with pytest.raises(TypeError):
        MRLLoss(nn.Linear(4, 4))  # type: ignore[arg-type]


def test_parse_argument_variants():
    assert parse_mrl_granularities_argument(None) is None
    assert parse_mrl_granularities_argument("none") is None
    assert parse_mrl_granularities_argument("auto") == "auto"
    assert parse_mrl_granularities_argument("8, 32 ,128,512") == [8, 32, 128, 512]


def test_resolve_granularities_auto_and_list():
    assert resolve_granularities("auto", 512) == [8, 16, 32, 64, 128, 256, 512]
    assert resolve_granularities([8, 16, 32, 64], 64) == [8, 16, 32, 64]
