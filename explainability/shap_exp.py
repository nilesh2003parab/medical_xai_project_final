import shap
import torch
import matplotlib.pyplot as plt


def run_shap(model, image_tensor):

    model.eval()

    # ✅ FIX 1: clone tensor (prevents view modification error)
    image_tensor = image_tensor.clone()

    # ✅ FIX 2: safer background (no repeat view issues)
    background = image_tensor.clone().detach()

    # ✅ FIX 3: use GradientExplainer instead of DeepExplainer
    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(image_tensor)

    fig = plt.figure()
    shap.image_plot(shap_values, image_tensor.detach().cpu().numpy(), show=False)

    shap_score = float(abs(shap_values[0]).mean())

    return shap_values, shap_score
