def e_score(gradcam_score, lime_score, shap_score):
    return round(
        0.4 * gradcam_score +
        0.3 * lime_score +
        0.3 * shap_score,
        3
    )
