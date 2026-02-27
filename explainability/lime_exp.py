import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image


def run_lime(model, image_np, transform):
    model.eval()
    device = next(model.parameters()).device

    def predict(images):
        batch = torch.stack([
            transform(Image.fromarray(img.astype(np.uint8)).convert("RGB"))
            for img in images
        ]).to(device)
        with torch.no_grad():
            preds = model(batch)
            return torch.softmax(preds, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np.astype(np.uint8),
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=300,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
    )

    lime_img = mark_boundaries(temp / 255.0, mask)
    return (lime_img * 255).astype(np.uint8), float(mask.mean())
