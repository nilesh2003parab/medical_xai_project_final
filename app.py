import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

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

# ---------- DEVICE ----------
device = torch.device("cpu")


# ---------- DISABLE INPLACE (fixes SHAP error) ----------
def disable_inplace(m):
    for module in m.modules():
        if hasattr(module, "inplace"):
            module.inplace = False


# ---------- LOAD MODEL — use cache_resource (not cache_data) ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    disable_inplace(m)
    m = m.to(device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "weights", "resnet18_pneumonia_classifier.pth")

    if not os.path.exists(path):
        st.error(f"❌ Weights not found: {path}\nRun train_model.py first.")
        st.stop()

    state_dict = torch.load(path, map_location=device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


model = load_model()
st.success("✅ Model loaded successfully!")

transform = get_transform()


# ---------- HELPERS (no cache_data on torch objects) ----------
def preprocess_image(image):
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


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
patient_id = st.sidebar.text_input("Patient ID")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120)
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
        st.success(f"🩺 Prediction: **{label}**  (Confidence: {confidence:.2f})")

    st.divider()
    st.subheader("🔍 Explainability Results")

    gradcam_score = lime_score = shap_score = None

    # ---------- LIME ----------
    try:
        st.markdown("### 🧩 LIME Explanation")
        with st.spinner("Running LIME..."):
            lime_img, lime_score = run_lime(model, np.array(image), transform)
        st.image(lime_img, use_container_width=True)
        st.caption(f"LIME Score: {float(lime_score):.3f}")
    except Exception as e:
        st.error(f"LIME error: {e}")

    st.divider()

    # ---------- Grad-CAM ----------
    try:
        st.markdown("### 🔥 Grad-CAM Heatmap")
        with st.spinner("Running Grad-CAM..."):
            gradcam_img, gradcam_score = generate_gradcam(model, img_tensor, image)
        st.image(gradcam_img, use_container_width=True)
        st.caption(f"Grad-CAM Score: {float(gradcam_score):.3f}")
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")

    st.divider()

    # ---------- SHAP ----------
    try:
        st.markdown("### 📊 SHAP Explanation")
        with st.spinner("Running SHAP..."):
            shap_values, shap_score = run_shap(model, img_tensor)

        if isinstance(shap_values, list):
            sv = np.array(shap_values[pred_class])
        else:
            sv = np.array(shap_values)

        # Flatten to 3 channel values for bar chart
        sv = np.squeeze(sv)
        if sv.ndim == 3:
            channel_means = np.mean(np.abs(sv), axis=(1, 2))
        elif sv.ndim == 1:
            channel_means = sv[:3]
        else:
            channel_means = np.mean(np.abs(sv), axis=tuple(range(1, sv.ndim)))[:3]

        channel_means = np.array(channel_means).flatten()[:3]

        labels_rgb = ["Red", "Green", "Blue"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels_rgb, channel_means)
        ax.set_title("SHAP Channel Importance")
        ax.set_ylabel("Mean |SHAP|")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"SHAP Score: {float(shap_score):.3f}")

    except Exception as e:
        st.error(f"SHAP error: {e}")

    st.divider()

    # ---------- E-Score ----------
    try:
        if gradcam_score is not None and lime_score is not None and shap_score is not None:
            escore_value = e_score(float(gradcam_score), float(lime_score), float(shap_score))
            st.info(f"🧮 E-Score (Weighted Explainability): **{float(escore_value):.3f}**")
        else:
            st.warning("E-Score not computed — one or more explanations failed.")
    except Exception as e:
        st.error(f"E-Score error: {e}")

    st.divider()

    # ---------- Doctor Feedback ----------
    try:
        feedback_img, feedback_text = doctor_feedback(label)
        st.image(feedback_img, width=400)
        st.caption(feedback_text)
    except Exception:
        st.warning("No doctor feedback image available.")

    st.divider()

    # ---------- SAVE ----------
    if st.button("💾 Save & Submit Analysis"):
        row = [
            patient_id, patient_name, patient_age, disease_present,
            major_surgeries, diabetes, bp, thyroid, cholesterol,
            asthma, label, confidence
        ]
        with open("patient_records.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        st.success("✅ Patient data saved successfully!")
