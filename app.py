import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------- PROJECT IMPORTS ----------
from utils.preprocessing import get_transform
from explainability.gradcam import generate_gradcam
from explainability.lime_exp import run_lime
from explainability.shap_exp import run_shap
from evaluation.escore import e_score
from utils.feedback_dataset import doctor_feedback

# ---------- APP CONFIG ----------
st.set_page_config(page_title="Medical XAI", layout="wide")
st.title("🧠 Real-Time Explainable Medical Image Classification")

device = torch.device("cpu")


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
        st.error(f"❌ Weights not found: {path}  —  Run train_model.py first.")
        st.stop()
    state_dict = torch.load(path, map_location=device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


model = load_model()
st.success("✅ Model loaded successfully!")
transform = get_transform()


def preprocess_image(image):
    return transform(image).unsqueeze(0).to(device)


def run_prediction(img_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    label = "Pneumonia" if pred_class == 1 else "Normal"
    return pred_class, confidence, label


# ---------- SIDEBAR ----------
st.sidebar.header("🧾 Patient Information")
patient_id    = st.sidebar.text_input("Patient ID")
patient_name  = st.sidebar.text_input("Patient Name")
patient_age   = st.sidebar.number_input("Age", min_value=0, max_value=120)
disease_present = st.sidebar.radio("Any Disease Present?", ("Yes", "No"))
major_surgeries = st.sidebar.text_area("Major Surgeries")

st.sidebar.subheader("Common Diseases")
diabetes, bp, thyroid, cholesterol, asthma = [
    st.sidebar.checkbox(d) for d in ["Diabetes", "Blood Pressure", "Thyroid", "Cholesterol", "Asthma"]
]

# ---------- IMAGE UPLOAD ----------
uploaded = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Medical Image", width=400)

    img_tensor = preprocess_image(image)

    with st.spinner("🩺 Running model prediction..."):
        pred_class, confidence, label = run_prediction(img_tensor)

    color = "🔴" if label == "Pneumonia" else "🟢"
    st.success(f"{color} Prediction: **{label}**  |  Confidence: **{confidence:.2%}**")

    st.divider()
    st.subheader("🔍 Explainability Results")

    col1, col2 = st.columns(2)
    gradcam_score = lime_score = shap_score = None

    # ---------- Grad-CAM ----------
    with col1:
        st.markdown("### 🔥 Grad-CAM")
        try:
            with st.spinner("Running Grad-CAM..."):
                gradcam_img, gradcam_score = generate_gradcam(model, img_tensor, image)
            st.image(gradcam_img, use_container_width=True)
            st.caption(f"Score: {gradcam_score:.3f}  |  Highlights regions activating the prediction")
        except Exception as e:
            st.error(f"Grad-CAM error: {e}")

    # ---------- LIME ----------
    with col2:
        st.markdown("### 🧩 LIME")
        try:
            with st.spinner("Running LIME (may take ~15s)..."):
                lime_img, lime_score = run_lime(model, np.array(image), transform)
            st.image(lime_img, use_container_width=True)
            st.caption(f"Score: {lime_score:.3f}  |  Red/yellow outlines = important regions")
        except Exception as e:
            st.error(f"LIME error: {e}")

    st.divider()

    # ---------- SHAP ----------
    st.markdown("### 📊 SHAP Explanation")
    try:
        with st.spinner("Running SHAP..."):
            channel_importance, spatial_map, shap_score = run_shap(model, img_tensor)

        # Ensure correct shapes before any display
        channel_importance = np.array(channel_importance).flatten()[:3]
        if len(channel_importance) < 3:
            channel_importance = np.pad(channel_importance, (0, 3 - len(channel_importance)))

        spatial_map = np.array(spatial_map)
        if spatial_map.ndim != 2:
            spatial_map = np.zeros((224, 224), dtype=np.float32)

        shap_col1, shap_col2 = st.columns(2)

        with shap_col1:
            st.markdown("**Channel Importance (R/G/B)**")
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(["Red", "Green", "Blue"], channel_importance,
                          color=["#e74c3c", "#2ecc71", "#3498db"], edgecolor="black")
            ax.set_ylabel("Mean |SHAP|")
            ax.set_title("Per-Channel Feature Impact")
            for bar, val in zip(bars, channel_importance):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with shap_col2:
            st.markdown("**Spatial SHAP Heatmap**")
            # Normalize spatial map
            s_min, s_max = spatial_map.min(), spatial_map.max()
            if s_max - s_min > 1e-8:
                spatial_norm = (spatial_map - s_min) / (s_max - s_min)
            else:
                spatial_norm = np.zeros_like(spatial_map)

            # Overlay on original image
            orig_resized = image.resize((224, 224))
            orig_np = np.array(orig_resized).astype(np.float32)

            heatmap_colored = cm.hot(spatial_norm)[:, :, :3]  # (H, W, 3) float
            heatmap_colored = (heatmap_colored * 255).astype(np.float32)

            shap_overlay = 0.5 * orig_np + 0.5 * heatmap_colored
            shap_overlay = np.clip(shap_overlay, 0, 255).astype(np.uint8)

            st.image(shap_overlay, caption="Bright = high SHAP impact area", use_container_width=True)

        st.caption(f"SHAP Score: {shap_score:.4f}")

    except Exception as e:
        st.error(f"SHAP error: {e}")

    st.divider()

    # ---------- E-Score ----------
    if gradcam_score is not None and lime_score is not None and shap_score is not None:
        escore_val = e_score(float(gradcam_score), float(lime_score), float(shap_score))
        st.info(f"🧮 **E-Score** (Weighted Explainability Index): `{escore_val:.3f}`")
    else:
        st.warning("E-Score not computed — one or more explanations failed.")

    st.divider()

    # ---------- Doctor Feedback ----------
    try:
        feedback_img, feedback_text = doctor_feedback(label)
        st.image(feedback_img, width=500)
        st.caption(feedback_text)
    except Exception:
        st.warning("No doctor feedback image available.")

    st.divider()

    # ---------- SAVE ----------
    if st.button("💾 Save & Submit Analysis"):
        row = [patient_id, patient_name, patient_age, disease_present,
               major_surgeries, diabetes, bp, thyroid, cholesterol,
               asthma, label, f"{confidence:.4f}"]
        with open("patient_records.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        st.success("✅ Patient data saved successfully!")
