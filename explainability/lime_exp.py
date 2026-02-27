import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image


def run_lime(model, image_np, transform):
    """
    LIME explanation. Returns (numpy uint8 image with boundaries, float score).
    """
    model.eval()
    device = next(model.parameters()).device

    # Ensure input is uint8 RGB (H, W, 3)
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    def predict_fn(images):
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            tensor = transform(pil)
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=300,
        random_seed=42,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )

    # temp is float [0,255], mask is binary
    temp_norm = temp.astype(np.float64) / 255.0
    lime_img = mark_boundaries(temp_norm, mask, color=(1, 0, 0), outline_color=(1, 1, 0))
    lime_img = (lime_img * 255).clip(0, 255).astype(np.uint8)

    score = float(mask.mean())
    return lime_img, score
