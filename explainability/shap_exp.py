import shap
import torch
import numpy as np


def run_shap(model, image_tensor):
    model.eval()
    device = next(model.parameters()).device

    # Clone to avoid in-place modification issues
    inp = image_tensor.clone().detach().to(device)
    background = torch.zeros_like(inp).to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(inp)

    # shap_values is a list of arrays, one per class
    if isinstance(shap_values, list):
        sv = np.array(shap_values)  # (n_classes, 1, C, H, W)
    else:
        sv = np.array(shap_values)

    score = float(np.mean(np.abs(sv)))
    return shap_values, score
