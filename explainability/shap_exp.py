import torch
import numpy as np


def run_shap(model, image_tensor):
    """
    Occlusion-based saliency map (SHAP-style) for 4-class brain tumor model.
    Slides a black patch, measures prediction drop → always non-zero values.
    Returns (channel_importance (3,), spatial_map (H,W), float score).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().float().to(device)  # (1, 3, H, W)
    _, C, H, W = inp.shape

    # Baseline
    with torch.no_grad():
        baseline_probs = torch.softmax(model(inp), dim=1)
        pred_class     = baseline_probs.argmax(dim=1).item()
        baseline_score = baseline_probs[0, pred_class].item()

    patch_size = 28
    stride     = 14
    saliency   = np.zeros((H, W), dtype=np.float32)
    counts     = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = inp.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0

            with torch.no_grad():
                score = torch.softmax(model(occluded), dim=1)[0, pred_class].item()

            drop = baseline_score - score
            saliency[y:y+patch_size, x:x+patch_size] += drop
            counts[y:y+patch_size, x:x+patch_size]   += 1

    counts   = np.where(counts == 0, 1, counts)
    saliency = saliency / counts
    saliency = np.clip(saliency, 0, None)

    # Per-channel importance
    inp_np = inp[0].detach().cpu().numpy()  # (3, H, W)
    channel_importance = np.array([
        float(np.mean(np.abs(inp_np[c]) * saliency))
        for c in range(C)
    ])

    score = float(saliency.mean())
    return channel_importance, saliency, score
