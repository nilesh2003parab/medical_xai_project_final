import torch
import numpy as np
import cv2
from PIL import Image


def generate_gradcam(model, image_tensor, original_image):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # Need gradients - use fresh tensor
    tensor = image_tensor.clone().detach().requires_grad_(True)
    output = model(tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    grads = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (grads * activations[0]).sum(dim=1).squeeze()
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    w, h = original_image.size
    heatmap = cv2.resize(cam, (w, h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    image_np = np.array(original_image.convert("RGB"))
    # heatmap_color is BGR, convert to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_rgb, 0.4, 0)

    return Image.fromarray(overlay), float(heatmap.mean())
