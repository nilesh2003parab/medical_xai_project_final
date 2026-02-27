import torch
import numpy as np
import cv2


def run_shap(model, image_tensor):
    """
    Occlusion-based saliency map (SHAP-style).
    Slides a patch across the image and measures prediction drop.
    Always produces real non-zero values — no GradientExplainer issues.
    Returns (channel_importance (3,), spatial_map (H,W), float score).
    """
    model.eval()
    device = next(model.parameters()).device

    inp = image_tensor.clone().detach().float().to(device)  # (1, 3, H, W)
    _, C, H, W = inp.shape

    # Baseline prediction
    with torch.no_grad():
        baseline_output = torch.softmax(model(inp), dim=1)
        pred_class = baseline_output.argmax(dim=1).item()
        baseline_score = baseline_output[0, pred_class].item()

    # Occlusion patch size and stride
    patch_size = 32
    stride = 16

    saliency = np.zeros((H, W), dtype=np.float32)
    counts   = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = inp.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0  # black patch

            with torch.no_grad():
                score = torch.softmax(model(occluded), dim=1)[0, pred_class].item()

            drop = baseline_score - score  # positive = this patch was important
            saliency[y:y+patch_size, x:x+patch_size] += drop
            counts[y:y+patch_size, x:x+patch_size]   += 1

    # Average overlapping patches
    counts = np.where(counts == 0, 1, counts)
    saliency = saliency / counts

    # Clip negatives (regions that helped when occluded = not important)
    saliency = np.clip(saliency, 0, None)

    # Per-channel importance: apply saliency to each channel of the image
    inp_np = inp[0].detach().cpu().numpy()  # (3, H, W)
    channel_importance = np.array([
        float(np.mean(np.abs(inp_np[c]) * saliency))
        for c in range(3)
    ])

    score = float(saliency.mean())
    return channel_importance, saliency, score
