import torch
import numpy as np
import cv2


def generate_gradcam(model, image_tensor, original_image):

    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # ✅ FIX 1: Remove .backbone
    target_layer = model.layer4[-1]

    # ✅ FIX 2: Use full backward hook (safer)
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = output.argmax(dim=1)
    score = output[0, class_idx]

    model.zero_grad()
    score.backward()

    grads = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (grads * activations[0]).sum(dim=1).squeeze()
    cam = cam.detach().cpu().numpy()

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Convert PIL size properly
    heatmap = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_np = np.array(original_image)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    handle_f.remove()
    handle_b.remove()

    return overlay, float(cam.mean())
