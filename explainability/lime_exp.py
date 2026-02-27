import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image


def run_lime(model, image_np, transform):
    """
    LIME explanation for brain MRI — 4-class classifier.
    Green outline = supports prediction | Orange = contradicts.
    """
    model.eval()
    device = next(model.parameters()).device

    # Sanitize input
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    def predict_fn(images):
        batch = torch.stack([
            transform(Image.fromarray(img.astype(np.uint8)).convert("RGB"))
            for img in images
        ]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        return probs  # shape (N, 4)

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=500,
        num_superpixels=60,
        random_seed=42,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=10,
        hide_rest=False,
        min_weight=0.01,
    )

    temp_norm = temp.astype(np.float64) / 255.0
    lime_img  = mark_boundaries(temp_norm, mask,
                                color=(0, 1, 0), outline_color=(1, 0.5, 0))
    lime_img  = (lime_img * 255).clip(0, 255).astype(np.uint8)

    return lime_img, float(np.abs(mask).mean())
