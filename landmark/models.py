"""
Model registry — scratch CNNs and transfer-learning backbones.
Transfer backbones use ImageNet-supervised weights (ResNet50, ConvNeXt, EfficientNet, ViT) plus a self-supervised I-JEPA ViT for comparison.
I-JEPA loading requires timm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


NUM_CLASSES_DEFAULT = 50


# ---------------------------------------------------------------------------
# Scratch: classic CNN with batch norm and dropout, sized for 224x224
# ---------------------------------------------------------------------------

class ScratchCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES_DEFAULT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),  # 112

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),  # 56

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),  # 28

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),  # 14

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Scratch ResNet — small, trained from random init
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
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


class ScratchResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES_DEFAULT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

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
        return self.fc(x)


# ---------------------------------------------------------------------------
# Transfer-learning factories — freeze backbone, train new head
# ---------------------------------------------------------------------------

def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def build_tl_resnet50(num_classes=NUM_CLASSES_DEFAULT, freeze_backbone=True):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        _freeze(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_tl_convnext(num_classes=NUM_CLASSES_DEFAULT, freeze_backbone=True):
    model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze(model)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def build_tl_efficientnet(num_classes=NUM_CLASSES_DEFAULT, freeze_backbone=True):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze(model)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_tl_vit(num_classes=NUM_CLASSES_DEFAULT, freeze_backbone=True):
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze(model)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# I-JEPA — self-supervised ViT (Meta). Loaded via timm.
# ---------------------------------------------------------------------------

# Candidate timm identifiers, tried in order. Different timm versions ship different tags.
_IJEPA_CANDIDATES = ("vit_huge_patch14_gap_224.in22k_ijepa", "vit_huge_patch16_gap_448.in22k_ijepa", "vit_huge_patch14_224.in22k_ijepa")


class _TimmBackboneHead(nn.Module):
    """Wrap a timm feature extractor with a linear head for our class count."""

    def __init__(self, backbone, in_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim == 3:  # (B, tokens, dim) — global-average pool
            features = features.mean(dim=1)
        return self.head(features)


def build_tl_ijepa(num_classes=NUM_CLASSES_DEFAULT, freeze_backbone=True):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for the I-JEPA backbone. Add it with: uv add timm") from exc

    last_error = None
    for name in _IJEPA_CANDIDATES:
        try:
            backbone = timm.create_model(name, pretrained=True, num_classes=0)
            break
        except Exception as exc:
            last_error = exc
            backbone = None
    if backbone is None:
        raise RuntimeError(f"Could not load any I-JEPA model from timm. Tried: {_IJEPA_CANDIDATES}. Last error: {last_error}")

    if freeze_backbone:
        _freeze(backbone)

    in_features = backbone.num_features
    return _TimmBackboneHead(backbone, in_features, num_classes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "scratch_cnn": ScratchCNN,
    "scratch_resnet": ScratchResNet,
    "tl_resnet50": build_tl_resnet50,
    "tl_convnext": build_tl_convnext,
    "tl_efficientnet": build_tl_efficientnet,
    "tl_vit": build_tl_vit,
    "tl_ijepa": build_tl_ijepa,
}

TRANSFER_MODELS = {"tl_resnet50", "tl_convnext", "tl_efficientnet", "tl_vit", "tl_ijepa"}


def get_model(name, num_classes=NUM_CLASSES_DEFAULT, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    factory = MODEL_REGISTRY[name]
    return factory(num_classes=num_classes, **kwargs)
