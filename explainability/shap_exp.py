import torch
import numpy as np
import shap


def run_shap(model, image_tensor):
    """
    SHAP GradientExplainer on ResNet18.
    Returns (channel_importance numpy array of shape (3,), float score).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().float().to(device)

    # Single black background image
    background = torch.zeros_like(inp).to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(inp)

    # shap_values: list of (1, C, H, W) arrays, one per class
    # Stack to (n_classes, 1, C, H, W)
    sv_array = np.array(shap_values)  # (n_classes, 1, C, H, W)

    # Get predicted class
    with torch.no_grad():
        pred_class = model(inp).argmax(dim=1).item()

    # Get shap for predicted class: (1, C, H, W) -> (C, H, W)
    sv_pred = sv_array[pred_class, 0]  # (C, H, W)  e.g. (3, 224, 224)

    # Per-channel mean absolute shap value
    channel_importance = np.mean(np.abs(sv_pred), axis=(1, 2))  # (3,)

    # Also compute a spatial heatmap for display (mean across channels)
    spatial_map = np.mean(np.abs(sv_pred), axis=0)  # (H, W)

    score = float(np.mean(np.abs(sv_pred)))
    return channel_importance, spatial_map, score
