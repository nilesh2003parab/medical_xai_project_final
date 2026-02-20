import shap
import torch
import matplotlib.pyplot as plt


def run_shap(model, image_tensor):

    model.eval()
    background = image_tensor.repeat(5, 1, 1, 1)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)

    fig = plt.figure()
    shap.image_plot(shap_values, image_tensor.cpu().numpy(), show=False)

    shap_score = float(abs(shap_values[0]).mean())

    return fig, shap_score