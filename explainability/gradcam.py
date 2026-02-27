import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def generate_gradcam(model, image_tensor, original_image):
    model.eval()
    gradients = []
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

    grads = gradients[0]   # (1, C, H, W)
    acts  = activations[0] # (1, C, H, W)

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze(0)
    cam = F.relu(cam).detach().cpu().numpy().astype(np.float32)

    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Resize to original image size
    orig_rgb = original_image.convert("RGB")
    w, h = orig_rgb.size
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # ✅ Explicit cast to uint8 CV_8UC1 before applyColorMap
    cam_uint8 = np.uint8(255 * cam_resized)          # CV_8UC1
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    orig_np = np.array(orig_rgb, dtype=np.float32)
    overlay = 0.55 * orig_np + 0.45 * heatmap_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay), float(cam_resized.mean())
