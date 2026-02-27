import torch
import numpy as np
import shap


def run_shap(model, image_tensor):
    """
    SHAP GradientExplainer on ResNet18.
    Returns (channel_importance (3,), spatial_map (H,W), float score).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().float().to(device)
    background = torch.zeros_like(inp).to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(inp)

    # shap_values can be:
    #   list of arrays  -> one per class, each shape (1, C, H, W)
    #   single array    -> shape (1, C, H, W) or (n_classes, 1, C, H, W)
    if isinstance(shap_values, list):
        sv_array = np.array(shap_values)   # (n_classes, 1, C, H, W)
    else:
        sv_array = np.array(shap_values)

    # Get predicted class index
    with torch.no_grad():
        pred_class = model(inp).argmax(dim=1).item()

    # Safely pick the right class dimension
    if sv_array.ndim == 5:
        # (n_classes, 1, C, H, W)
        n_classes = sv_array.shape[0]
        idx = min(pred_class, n_classes - 1)
        sv = sv_array[idx, 0]          # (C, H, W)
    elif sv_array.ndim == 4:
        # (1, C, H, W) — only one output
        sv = sv_array[0]               # (C, H, W)
    else:
        # fallback: flatten whatever we have
        sv = sv_array.reshape(3, -1).mean(axis=1, keepdims=True)
        sv = np.broadcast_to(sv[:, :, np.newaxis], (3, 224, 224))

    # Per-channel mean absolute shap  -> bar chart
    channel_importance = np.mean(np.abs(sv), axis=(1, 2))   # (3,)

    # Spatial heatmap -> overlay on image
    spatial_map = np.mean(np.abs(sv), axis=0)               # (H, W)

    score = float(np.mean(np.abs(sv)))
    return channel_importance, spatial_map, score
