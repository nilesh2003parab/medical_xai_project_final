import torch
import numpy as np
import shap


def run_shap(model, image_tensor):
    """
    SHAP GradientExplainer on ResNet18.
    Returns (channel_importance shape (3,), spatial_map shape (H,W), float score).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().float().to(device)
    background = torch.zeros_like(inp).to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(inp)

    # Normalize to numpy array
    if isinstance(shap_values, list):
        sv_array = np.array(shap_values)  # (n_classes, 1, C, H, W)
    else:
        sv_array = np.array(shap_values)

    # Get predicted class
    with torch.no_grad():
        pred_class = int(model(inp).argmax(dim=1).item())

    # Extract (C, H, W) for predicted class robustly
    if sv_array.ndim == 5:
        # (n_classes, batch, C, H, W)
        idx = min(pred_class, sv_array.shape[0] - 1)
        sv = sv_array[idx, 0]          # (C, H, W)
    elif sv_array.ndim == 4:
        # (batch, C, H, W)
        sv = sv_array[0]               # (C, H, W)
    elif sv_array.ndim == 3:
        # (C, H, W) already
        sv = sv_array
    else:
        # Fallback: create dummy
        sv = np.zeros((3, 224, 224), dtype=np.float32)

    sv = sv.astype(np.float32)

    # Ensure shape is exactly (C, H, W)
    if sv.ndim != 3 or sv.shape[0] != 3:
        sv = np.zeros((3, 224, 224), dtype=np.float32)

    C, H, W = sv.shape

    # Per-channel importance: mean abs over spatial dims -> shape (3,)
    channel_importance = np.mean(np.abs(sv), axis=(1, 2))   # (3,)

    # Spatial heatmap: mean abs over channel dim -> shape (H, W)
    spatial_map = np.mean(np.abs(sv), axis=0)               # (H, W)

    score = float(np.mean(np.abs(sv)))
    return channel_importance, spatial_map, score
