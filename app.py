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

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor XAI", layout="wide", page_icon="🧠")
st.title("🧠 Brain Tumor MRI Classification with Explainable AI")
st.caption("Dataset: Kaggle Brain Tumor MRI Dataset — Glioma | Meningioma | No Tumor | Pituitary")

device = torch.device("cpu")

# 4 classes — ImageFolder sorts alphabetically
CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",
}

CLASS_DISPLAY = {
    "glioma":      ("🔴 GLIOMA",          "#e74c3c"),
    "meningioma":  ("🟠 MENINGIOMA",       "#e67e22"),
    "notumor":     ("🟢 NO TUMOR",         "#2ecc71"),
    "pituitary":   ("🔵 PITUITARY TUMOR",  "#3498db"),
}

NUM_CLASSES = 4


def disable_inplace(m):
    for module in m.modules():
        if hasattr(module, "inplace"):
            module.inplace = False


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    disable_inplace(m)
    m = m.to(device)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "weights", "resnet18_brain_tumor.pth")
    if not os.path.exists(path):
        st.error(f"❌ Weights not found: {path}\nRun train_model.py first.")
        st.stop()
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


model     = load_model()
transform = get_transform()
st.success("✅ Model loaded!")


def preprocess_image(image):
    return transform(image).unsqueeze(0).to(device)


def run_prediction(img_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(np.argmax(probs))
    label      = CLASS_NAMES[pred_class]
    confidence = float(probs[pred_class])
    return pred_class, confidence, label, probs


def make_shap_overlay(spatial_map, original_image):
    orig_rgb = original_image.convert("RGB")
    w, h     = orig_rgb.size
    orig_np  = np.array(orig_rgb, dtype=np.float32)
    s_min, s_max = spatial_map.min(), spatial_map.max()
    if s_max - s_min < 1e-8:
        return orig_rgb
    sn   = (spatial_map - s_min) / (s_max - s_min)
    heat = cv2.resize(sn.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    h8   = np.uint8(255 * heat)
    hbgr = cv2.applyColorMap(h8, cv2.COLORMAP_HOT)
    hrgb = cv2.cvtColor(hbgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha   = heat[:, :, np.newaxis] * 0.7
    overlay = orig_np * (1 - alpha) + hrgb * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("🧾 Patient Information")
patient_id      = st.sidebar.text_input("Patient ID")
patient_name    = st.sidebar.text_input("Patient Name")
patient_age     = st.sidebar.number_input("Age", min_value=0, max_value=120)
patient_gender  = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
symptoms        = st.sidebar.text_area("Symptoms / Clinical Notes")
prev_surgery    = st.sidebar.text_input("Previous Brain Surgeries")

st.sidebar.subheader("Comorbidities")
hypertension, diabetes, epilepsy, headache, vision = [
    st.sidebar.checkbox(d)
    for d in ["Hypertension", "Diabetes", "Epilepsy", "Chronic Headache", "Vision Problems"]
]

# ── UPLOAD ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload Brain MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    img_col, pred_col = st.columns([1, 1])

    with img_col:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    img_tensor = preprocess_image(image)

    with st.spinner("🧠 Analyzing MRI..."):
        pred_class, confidence, label, probs = run_prediction(img_tensor)

    disp_name, disp_color = CLASS_DISPLAY[label]

    with pred_col:
        st.markdown("### 🔬 Classification Result")
        st.markdown(
            f"<div style='background:{disp_color};padding:16px;border-radius:8px;"
            f"color:white;font-size:22px;font-weight:bold;text-align:center'>"
            f"{disp_name}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.2%}")
        st.markdown("**All Class Probabilities**")

        fig_prob, ax_prob = plt.subplots(figsize=(5, 2.5))
        cls_labels = [CLASS_NAMES[i].upper() for i in range(NUM_CLASSES)]
        bar_colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
        bars = ax_prob.barh(cls_labels, probs, color=bar_colors, edgecolor="black")
        ax_prob.set_xlim(0, 1)
        ax_prob.set_xlabel("Probability")
        for bar, val in zip(bars, probs):
            ax_prob.text(min(val + 0.02, 0.95),
                         bar.get_y() + bar.get_height() / 2,
                         f"{val:.2%}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

    st.divider()

    # ── EXPLAINABILITY ─────────────────────────────────────────────────────────
    st.subheader("🔍 Explainability Results")

    gradcam_score = lime_score = shap_score = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Grad-CAM")
        st.caption("Red/warm = regions the model focused on for this prediction")
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

    # ── SHAP ───────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Occlusion Saliency (SHAP-style)")
    st.caption("Which brain regions most influenced the model's decision?")

    try:
        with st.spinner("Running occlusion saliency (~25s)..."):
            channel_importance, spatial_map, shap_score = run_shap(model, img_tensor)

        # Sanitize
        channel_importance = np.array(channel_importance).flatten()[:3]
        if len(channel_importance) < 3:
            channel_importance = np.pad(channel_importance, (0, 3 - len(channel_importance)))
        spatial_map = np.array(spatial_map)
        if spatial_map.ndim != 2:
            spatial_map = np.zeros((224, 224), dtype=np.float32)

        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("**Per-Channel Importance**")
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(["Red", "Green", "Blue"], channel_importance,
                          color=["#e74c3c", "#2ecc71", "#3498db"],
                          edgecolor="black", width=0.5)
            ax.set_ylabel("Weighted Saliency")
            ax.set_title("Channel Importance")
            max_val = max(channel_importance.max(), 1e-8)
            for bar, val in zip(bars, channel_importance):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_val * 0.03,
                        f"{val:.5f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, max_val * 1.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with sc2:
            st.markdown("**Spatial Saliency Heatmap**")
            st.caption("Bright = high impact region for the classification")
            st.image(make_shap_overlay(spatial_map, image), use_container_width=True)

        st.caption(f"Saliency Score: **{shap_score:.5f}**")

    except Exception as e:
        st.error(f"SHAP error: {e}")

    st.divider()

    # ── E-SCORE ────────────────────────────────────────────────────────────────
    if all(s is not None for s in [gradcam_score, lime_score, shap_score]):
        ev = e_score(float(gradcam_score), float(lime_score), float(shap_score))
        st.info(f"🧮 **E-Score** (Weighted Explainability Index): `{ev:.3f}`")
    else:
        st.warning("E-Score not computed — one or more explanations failed.")

    st.divider()

    # ── DOCTOR FEEDBACK ────────────────────────────────────────────────────────
    st.markdown("### 👨‍⚕️ Clinical Recommendations")
    try:
        feedback_img, feedback_text = doctor_feedback(label)
        st.image(feedback_img, width=580)
        with st.expander("📋 Full Clinical Notes"):
            st.text(feedback_text)
    except Exception:
        st.warning("No doctor feedback available.")

    st.divider()

    # ── SAVE ───────────────────────────────────────────────────────────────────
    if st.button("💾 Save & Submit Analysis"):
        row = [
            patient_id, patient_name, patient_age, patient_gender,
            symptoms, prev_surgery,
            hypertension, diabetes, epilepsy, headache, vision,
            label, f"{confidence:.4f}",
            f"{probs[0]:.4f}", f"{probs[1]:.4f}",
            f"{probs[2]:.4f}", f"{probs[3]:.4f}",
        ]
        with open("patient_records.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        st.success("✅ Patient data saved to patient_records.csv!")
