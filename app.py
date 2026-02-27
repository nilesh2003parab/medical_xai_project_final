import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt

from utils.preprocessing import get_transform
from explainability.gradcam import generate_gradcam
from explainability.lime_exp import run_lime
from explainability.shap_exp import run_shap
from evaluation.escore import e_score
from utils.feedback_dataset import doctor_feedback

st.set_page_config(page_title="Medical XAI", layout="wide")
st.title("🧠 Real-Time Explainable Medical Image Classification")
st.caption("Dataset: Kaggle Chest X-Ray Images (Pneumonia) — NORMAL vs PNEUMONIA")

device = torch.device("cpu")

# Kaggle dataset: ImageFolder sorts alphabetically
# NORMAL = index 0, PNEUMONIA = index 1
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def disable_inplace(m):
    for module in m.modules():
        if hasattr(module, "inplace"):
            module.inplace = False


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    disable_inplace(m)
    m = m.to(device)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "weights", "resnet18_pneumonia_classifier.pth")
    if not os.path.exists(path):
        st.error(f"❌ Weights not found at: {path}\nRun train_model.py first.")
        st.stop()
    state_dict = torch.load(path, map_location=device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


model = load_model()
st.success("✅ Model loaded!")
transform = get_transform()


def preprocess_image(image):
    return transform(image).unsqueeze(0).to(device)


def run_prediction(img_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    label = CLASS_NAMES[pred_class]
    normal_prob    = probs[0, 0].item()
    pneumonia_prob = probs[0, 1].item()
    return pred_class, confidence, label, normal_prob, pneumonia_prob


def make_shap_overlay(spatial_map, original_image):
    orig_rgb = original_image.convert("RGB")
    w, h = orig_rgb.size
    orig_np = np.array(orig_rgb, dtype=np.float32)

    s_min, s_max = spatial_map.min(), spatial_map.max()
    if s_max - s_min < 1e-8:
        return orig_rgb

    spatial_norm = (spatial_map - s_min) / (s_max - s_min)
    heat = cv2.resize(spatial_norm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat_uint8 = np.uint8(255 * heat)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_HOT)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    alpha = heat[:, :, np.newaxis]
    overlay = orig_np * (1 - 0.7 * alpha) + heat_rgb * (0.7 * alpha)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("🧾 Patient Information")
patient_id      = st.sidebar.text_input("Patient ID")
patient_name    = st.sidebar.text_input("Patient Name")
patient_age     = st.sidebar.number_input("Age", min_value=0, max_value=120)
disease_present = st.sidebar.radio("Any Disease Present?", ("Yes", "No"))
major_surgeries = st.sidebar.text_area("Major Surgeries")

st.sidebar.subheader("Common Diseases")
diabetes, bp, thyroid, cholesterol, asthma = [
    st.sidebar.checkbox(d) for d in
    ["Diabetes", "Blood Pressure", "Thyroid", "Cholesterol", "Asthma"]
]

# ── IMAGE UPLOAD ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    img_col, info_col = st.columns([1, 1])
    with img_col:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img_tensor = preprocess_image(image)

    with st.spinner("🩺 Running prediction..."):
        pred_class, confidence, label, normal_prob, pneumonia_prob = run_prediction(img_tensor)

    with info_col:
        st.markdown("### 🩺 Prediction Result")
        if label == "PNEUMONIA":
            st.error(f"🔴 **PNEUMONIA DETECTED**")
        else:
            st.success(f"🟢 **NORMAL**")

        st.metric("Confidence", f"{confidence:.2%}")

        st.markdown("**Class Probabilities**")
        fig_prob, ax_prob = plt.subplots(figsize=(4, 2))
        bars = ax_prob.barh(["NORMAL", "PNEUMONIA"],
                            [normal_prob, pneumonia_prob],
                            color=["#2ecc71", "#e74c3c"])
        ax_prob.set_xlim(0, 1)
        ax_prob.set_xlabel("Probability")
        for bar, val in zip(bars, [normal_prob, pneumonia_prob]):
            ax_prob.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                         f"{val:.2%}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

    st.divider()
    st.subheader("🔍 Explainability Results")

    gradcam_score = lime_score = shap_score = None

    # ── Grad-CAM + LIME ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Grad-CAM")
        st.caption("Warm/red areas = regions model focused on for this prediction")
        try:
            with st.spinner("Running Grad-CAM..."):
                gradcam_img, gradcam_score = generate_gradcam(model, img_tensor, image)
            st.image(gradcam_img, use_container_width=True)
            st.caption(f"Score: **{gradcam_score:.3f}**")
        except Exception as e:
            st.error(f"Grad-CAM error: {e}")

    with col2:
        st.markdown("### 🧩 LIME")
        st.caption("Green = supports prediction | Orange = contradicts prediction")
        try:
            with st.spinner("Running LIME (~15s)..."):
                lime_img, lime_score = run_lime(model, np.array(image), transform)
            st.image(lime_img, use_container_width=True)
            st.caption(f"Score: **{lime_score:.3f}**")
        except Exception as e:
            st.error(f"LIME error: {e}")

    st.divider()

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.markdown("### 📊 SHAP — Occlusion Saliency")
    st.caption("Which regions matter most? (black patch sliding test)")
    try:
        with st.spinner("Running SHAP occlusion (~25s)..."):
            channel_importance, spatial_map, shap_score = run_shap(model, img_tensor)

        # Sanitize
        channel_importance = np.array(channel_importance).flatten()
        if len(channel_importance) < 3:
            channel_importance = np.pad(channel_importance, (0, 3 - len(channel_importance)))
        channel_importance = channel_importance[:3]
        spatial_map = np.array(spatial_map)
        if spatial_map.ndim != 2:
            spatial_map = np.zeros((224, 224), dtype=np.float32)

        shap_col1, shap_col2 = st.columns(2)

        with shap_col1:
            st.markdown("**Per-Channel Feature Impact**")
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ["#e74c3c", "#2ecc71", "#3498db"]
            bars = ax.bar(["Red", "Green", "Blue"], channel_importance,
                          color=colors, edgecolor="black", width=0.5)
            ax.set_ylabel("Weighted Saliency")
            ax.set_title("Channel Importance")
            max_val = max(channel_importance.max(), 1e-6)
            for bar, val in zip(bars, channel_importance):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_val * 0.03,
                        f"{val:.5f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, max_val * 1.35)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with shap_col2:
            st.markdown("**Spatial Saliency Heatmap**")
            st.caption("Bright = most influential region for prediction")
            shap_overlay = make_shap_overlay(spatial_map, image)
            st.image(shap_overlay, use_container_width=True)

        st.caption(f"SHAP Occlusion Score: **{shap_score:.5f}**")

    except Exception as e:
        st.error(f"SHAP error: {e}")

    st.divider()

    # ── E-Score ───────────────────────────────────────────────────────────────
    if all(s is not None for s in [gradcam_score, lime_score, shap_score]):
        escore_val = e_score(float(gradcam_score), float(lime_score), float(shap_score))
        st.info(f"🧮 **E-Score** (Weighted Explainability Index): `{escore_val:.3f}`")
    else:
        st.warning("E-Score not computed — one or more explanations failed.")

    st.divider()

    # ── Doctor Feedback ───────────────────────────────────────────────────────
    try:
        feedback_img, feedback_text = doctor_feedback(label)
        st.image(feedback_img, width=500)
        st.caption(feedback_text)
    except Exception:
        st.warning("No doctor feedback image available.")

    st.divider()

    # ── Save ──────────────────────────────────────────────────────────────────
    if st.button("💾 Save & Submit Analysis"):
        row = [patient_id, patient_name, patient_age, disease_present,
               major_surgeries, diabetes, bp, thyroid, cholesterol,
               asthma, label, f"{confidence:.4f}",
               f"{normal_prob:.4f}", f"{pneumonia_prob:.4f}"]
        with open("patient_records.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        st.success("✅ Patient data saved successfully!")
