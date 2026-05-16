import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset


# ImageNet statistics — defaults for transfer-learning backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_DEVIATION = (0.229, 0.224, 0.225)

DEFAULT_DATA_DIR = "./data"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 42


def discover_classes(data_dir=DEFAULT_DATA_DIR):
    """Read the class folder names from data/train, sorted (ImageFolder order)."""
    train_root = os.path.join(data_dir, "train")
    classes = sorted(d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d)))
    return classes


def pretty_class_name(folder_name):
    """Convert "09.Golden_Gate_Bridge" -> "Golden Gate Bridge"."""
    name = folder_name.split(".", 1)[-1] if "." in folder_name else folder_name
    return name.replace("_", " ")


def compute_mean_standard_deviation(data_dir=DEFAULT_DATA_DIR, image_size=DEFAULT_IMAGE_SIZE, max_samples=2000):
    """Compute per-channel mean/std from the train set (sampled for speed)."""
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    if len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(DEFAULT_SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    for images, _ in loader:
        b, _, h, w = images.shape
        n_pixels += b * h * w
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))
    mean = (channel_sum / n_pixels).tolist()
    var = (channel_sum_sq / n_pixels) - (channel_sum / n_pixels) ** 2
    std = var.clamp(min=0).sqrt().tolist()

    print(f"Computed mean: {tuple(round(m, 4) for m in mean)}")
    print(f"Computed standard deviation: {tuple(round(s, 4) for s in std)}")

    return tuple(mean), tuple(std)


def get_eval_transforms(image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ])


def get_train_val_loaders(train_transform, eval_transform, batch_size=32, num_workers=2, data_dir=DEFAULT_DATA_DIR, val_fraction=DEFAULT_VAL_FRACTION, seed=DEFAULT_SEED):
    """
    Build train and validation loaders by splitting data/train/.
    The same on-disk dataset is loaded twice with different transforms so train augmentation does not leak into validation.
    """
    train_root = os.path.join(data_dir, "train")
    train_set_aug = torchvision.datasets.ImageFolder(train_root, transform=train_transform)
    val_set = torchvision.datasets.ImageFolder(train_root, transform=eval_transform)

    n = len(train_set_aug)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(n), [n_train, n_val], generator=generator)

    train_subset = Subset(train_set_aug, list(train_indices))
    val_subset = Subset(val_set, list(val_indices))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_set_aug.classes


def get_test_loader(eval_transform, batch_size=32, num_workers=2, data_dir=DEFAULT_DATA_DIR):
    test_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def get_sample_batch(loader, n=5):
    images, labels = next(iter(loader))
    return images[:n], labels[:n]
