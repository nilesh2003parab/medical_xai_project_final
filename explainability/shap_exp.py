import shap
import torch
import numpy as np


def run_shap(model, image_tensor):
    """
    Lightweight SHAP using GradientExplainer with a small background.
    Returns (shap_values list, score float).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().to(device)

    # Single zero background — minimal memory usage
    background = torch.zeros(1, *inp.shape[1:]).to(device)

    explainer = shap.GradientExplainer(model, background)

    # Compute shap values — no image_plot, no plt calls
    with torch.no_grad():
        pass  # warm up

    shap_values = explainer.shap_values(inp)

    score = float(np.mean(np.abs(np.array(shap_values))))
    return shap_values, score
