import torch
import numpy as np
import torchvision.transforms as transforms

from landmark.data import IMAGENET_MEAN, IMAGENET_STANDARD_DEVIATION, DEFAULT_IMAGE_SIZE


class CutoutTransform:
    def __init__(self, size=32):
        self.size = size

    def __call__(self, img):
        height, width = img.shape[1], img.shape[2]
        y_axis = np.random.randint(height)
        x_axis = np.random.randint(width)
        y1 = max(0, y_axis - self.size // 2)
        y2 = min(height, y_axis + self.size // 2)
        x1 = max(0, x_axis - self.size // 2)
        x2 = min(width, x_axis + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


def _base_resize(image_size):
    return [
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
    ]


def get_none_transforms(image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    return transforms.Compose([
        *_base_resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ])


def get_flip_crop_transforms(image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ])


def get_color_jitter_transforms(image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ])


def get_randaug_transforms(image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
        CutoutTransform(size=image_size // 6),
    ])


def get_mixup_transforms(image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    """Base transforms — mixing happens in the training loop."""
    return get_flip_crop_transforms(image_size, mean, standard_deviation)


def mixup_batch(images, labels, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lambda_ * images + (1 - lambda_) * images[index]
    return mixed_images, labels, labels[index], lambda_


AUGMENTATION_REGISTRY = {
    "none": get_none_transforms,
    "flip-crop": get_flip_crop_transforms,
    "color-jitter": get_color_jitter_transforms,
    "randaug": get_randaug_transforms,
    "mixup": get_mixup_transforms,
}


def get_augmentation(name, image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}. Choose from {list(AUGMENTATION_REGISTRY)}")
    return AUGMENTATION_REGISTRY[name](image_size, mean, standard_deviation)
