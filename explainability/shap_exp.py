import torch
import numpy as np
import shap


def run_shap(model, image_tensor):
    """
    SHAP using GradientExplainer with multiple background samples
    to avoid near-zero gradients from a single black background.
    Returns (channel_importance (3,), spatial_map (H,W), float score).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().float().to(device)  # (1, 3, H, W)

    # Use 5 random noise backgrounds — much better than single zero background
    torch.manual_seed(42)
    background = torch.randn(5, *inp.shape[1:]).to(device) * 0.1

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(inp)

    # shap_values: list[n_classes] of (1, C, H, W)  OR  ndarray
    if isinstance(shap_values, list):
        sv_array = np.array(shap_values)   # (n_classes, 1, C, H, W)
    else:
        sv_array = np.array(shap_values)

    # Get predicted class
    with torch.no_grad():
        pred_class = int(model(inp).argmax(dim=1).item())

    # Extract (C, H, W) safely
    if sv_array.ndim == 5:
        idx = min(pred_class, sv_array.shape[0] - 1)
        sv = sv_array[idx, 0]       # (C, H, W)
    elif sv_array.ndim == 4:
        sv = sv_array[0]            # (C, H, W)
    elif sv_array.ndim == 3:
        sv = sv_array
    else:
        sv = np.zeros((3, 224, 224), dtype=np.float32)

    sv = sv.astype(np.float32)
    if sv.ndim != 3 or sv.shape[0] != 3:
        sv = np.zeros((3, 224, 224), dtype=np.float32)

    # Per-channel mean |SHAP| -> bar chart
    channel_importance = np.mean(np.abs(sv), axis=(1, 2))   # (3,)

    # Spatial mean |SHAP| across channels -> heatmap overlay
    spatial_map = np.mean(np.abs(sv), axis=0)               # (H, W)

    score = float(np.mean(np.abs(sv)))
    return channel_importance, spatial_map, score
