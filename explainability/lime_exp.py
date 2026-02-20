import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image


def run_lime(model, image_np, transform):

    def predict(images):
        batch = torch.stack([
            transform(Image.fromarray(img))
            for img in images
        ])
        with torch.no_grad():
            preds = model(batch)
            return torch.softmax(preds, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=500
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp / 255.0, mask)
    lime_score = float(mask.mean())

    return (lime_img * 255).astype(np.uint8), lime_score