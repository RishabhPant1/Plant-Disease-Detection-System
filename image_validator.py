"""
Image Validation / Input Gating Module

Validates uploaded images before plant disease prediction:
1. Checks that the image is a valid, uncorrupted colored image (rejects B&W / grayscale).
2. Uses zero-shot CLIP classification to verify the image depicts a plant leaf or foliage,
   accepting leaves from any plant/species, while rejecting non-leaf objects
   (people, animals, buildings, vehicles, screenshots, food, everyday objects, etc.).
"""

import os

# Enforce offline / local loading from cached weights
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# Exact required rejection message for any invalid image
REJECTION_MESSAGE = "Please upload a valid colored leaf photo or image"

# Candidate labels for zero-shot gating
CANDIDATE_LABELS = [
    "a photo of a plant leaf or foliage",
    "a photo of a person or human face",
    "a photo of an animal, bird, insect, or pet",
    "a photo of a car, airplane, vehicle, or machinery",
    "a photo of a building, house, or room interior",
    "a photo of food, meal, fruit, or dish",
    "a photo of a flower blossom without leaves",
    "a computer screenshot, text document, diagram, or graphic",
    "a scenic landscape, mountain, sky, or outdoor scenery",
    "a photo of everyday objects, furniture, tools, or clothing",
]


def load_validator(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Loads CLIP model and processor from local cache for zero-shot leaf classification.
    """
    # Enforce offline / local loading from cached weights
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = CLIPModel.from_pretrained(model_name, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return model, processor


def is_colored_image(image: Image.Image) -> bool:
    """
    Checks if an image has sufficient chromatic variation to be considered colored.
    Rejects grayscale, black-and-white, and desaturated images.
    Handles alpha channels / transparency by compositing against a white background.
    """
    if image.mode in ("L", "1", "LA"):
        return False

    # Handle transparency
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        rgb = bg
    else:
        rgb = image.convert("RGB")

    arr = np.array(rgb, dtype=np.float32)
    # Channel divergence: |R - G| + |G - B| + |B - R|
    diff = (
        np.abs(arr[:, :, 0] - arr[:, :, 1])
        + np.abs(arr[:, :, 1] - arr[:, :, 2])
        + np.abs(arr[:, :, 2] - arr[:, :, 0])
    )
    mean_diff = float(np.mean(diff))
    colored_ratio = float(np.mean(diff > 10.0))

    # Real colored leaves (green, yellow, brown blight) have mean_diff >= 5.0 and colored_ratio >= 0.02.
    # Grayscale / monochrome images have mean_diff near 0.
    return (mean_diff >= 5.0) and (colored_ratio >= 0.02)


def is_leaf_image(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    min_confidence: float = 0.40,
) -> tuple[bool, float, str]:
    """
    Classifies whether the image depicts a plant leaf or foliage using zero-shot CLIP.
    Accepts any plant leaf (black pepper, tomato, potato, rose, oak, etc.).
    Rejects non-leaf categories (people, animals, vehicles, buildings, food, objects, etc.).

    Returns:
        (is_leaf, leaf_probability, predicted_label)
    """
    # Ensure RGB
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        rgb_img = bg
    else:
        rgb_img = image.convert("RGB")

    inputs = processor(
        text=CANDIDATE_LABELS,
        images=rgb_img,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    leaf_prob = float(probs[0])
    top_idx = int(probs.argmax())
    top_label = CANDIDATE_LABELS[top_idx]

    # Must be classified as leaf (top label) and meet minimum leaf probability
    is_leaf = (top_idx == 0) and (leaf_prob >= min_confidence)
    return is_leaf, leaf_prob, top_label


def validate_colored_leaf(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> tuple[bool, str]:
    """
    High-level validation function.
    Validates that:
    1. Image is not corrupted or unreadable.
    2. Image is colored (not grayscale or black-and-white).
    3. Image is a plant leaf or foliage (any species).

    Returns:
        (True, "") if valid colored leaf image.
        (False, REJECTION_MESSAGE) if invalid for any reason.
    """
    try:
        if image is None:
            return False, REJECTION_MESSAGE

        # Step 1: Color check
        if not is_colored_image(image):
            return False, REJECTION_MESSAGE

        # Step 2: Leaf vs non-leaf check
        is_leaf, _, _ = is_leaf_image(image, model, processor)
        if not is_leaf:
            return False, REJECTION_MESSAGE

        return True, ""

    except Exception:
        # Failsafe: on any unexpected error or corrupted image, reject safely
        return False, REJECTION_MESSAGE
