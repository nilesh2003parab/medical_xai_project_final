import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def generate_gradcam(model, image_tensor, original_image):
    """
    Grad-CAM focused overlay for brain MRI.
    Only highlights top 40% activation regions — rest shows original MRI clearly.
    """
    model.eval()
    gradients   = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out.clone())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].clone())

    target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    tensor = image_tensor.clone().detach().float().requires_grad_(True)
    output = model(tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    fh.remove()
    bh.remove()

    if not gradients or not activations:
        return original_image.convert("RGB"), 0.0

    grads   = gradients[0]    # (1, C, H, W)
    acts    = activations[0]  # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = (weights * acts).sum(dim=1).squeeze(0)
    cam     = F.relu(cam).detach().cpu().numpy().astype(np.float32)

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min < 1e-8:
        return original_image.convert("RGB"), 0.0
    cam = (cam - cam_min) / (cam_max - cam_min)

    orig_rgb = original_image.convert("RGB")
    w, h     = orig_rgb.size
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Threshold: only colour top 40% activations
    threshold   = np.percentile(cam_resized, 60)
    cam_focused = np.where(cam_resized >= threshold, cam_resized, 0.0)
    f_max       = cam_focused.max()
    if f_max > 1e-8:
        cam_focused = cam_focused / f_max

    cam_uint8   = np.uint8(255 * cam_focused)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    orig_np = np.array(orig_rgb, dtype=np.float32)
    alpha   = cam_focused[:, :, np.newaxis] * 0.65
    overlay = orig_np * (1.0 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay), float(cam_resized.mean())
