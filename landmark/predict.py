import os
import torch
from PIL import Image

from landmark.data import IMAGENET_MEAN, IMAGENET_STANDARD_DEVIATION, DEFAULT_IMAGE_SIZE, get_eval_transforms, pretty_class_name


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def predict_landmarks(model, image_path, classes, k=3, device=None, image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    """Return the top-k predicted landmark names for a single image."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_eval_transforms(image_size=image_size, mean=mean, standard_deviation=standard_deviation)
    tensor = load_image(image_path, transform).unsqueeze(0).to(device)

    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = probabilities.topk(k)

    predictions = []
    for p, i in zip(top_probs, top_indices):
        folder = classes[int(i)]
        predictions.append({"class": folder, "name": pretty_class_name(folder), "probability": float(p)})

    return predictions


def suggest_locations(model, image_path, classes, k=3, device=None, show=True, image_size=DEFAULT_IMAGE_SIZE, mean=IMAGENET_MEAN, standard_deviation=IMAGENET_STANDARD_DEVIATION):
    """
    Display an image and the top-k landmark suggestions.
    Returns the predictions.
    """
    predictions = predict_landmarks(model, image_path, classes, k=k, device=device, image_size=image_size, mean=mean, standard_deviation=standard_deviation)

    if show:
        import matplotlib.pyplot as plt

        image = Image.open(image_path).convert("RGB")
        lines = [f"{p['name']} ({p['probability'] * 100:.1f}%)" for p in predictions]

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Is this a picture of...\n" + "\n".join(lines))
        plt.show()
    else:
        print(f"Image: {os.path.basename(image_path)}")
        for p in predictions:
            print(f"{p['name']} — {p['probability'] * 100:.1f}%")

    return predictions
