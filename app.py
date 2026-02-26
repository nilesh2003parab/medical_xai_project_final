import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import os

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# ✅ FIX FOR SHAP INPLACE ERROR (ONLY THIS ADDED)
def disable_inplace(model):
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

disable_inplace(model)

model = model.to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "weights", "resnet18_pneumonia_classifier.pth")

st.write("Model path:", model_path)
st.write("File exists?", os.path.exists(model_path))

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

# ---------- CACHE LIME ----------
@st.cache_data(show_spinner=False)
def cached_lime(_model, image_np, _transform):
    return run_lime(_model, image_np, _transform)

# ---------- CACHE SHAP ----------
@st.cache_data(show_spinner=False)
def cached_shap(_model, _img_tensor):
    return run_shap(_model, _img_tensor)

# ---------- CACHE IMAGE PREPROCESSING ----------
@st.cache_data(show_spinner=False)
def preprocess_image(_image, _transform):
    tensor = _transform(_image).unsqueeze(0).to(device)
    tensor.requires_grad_()
    return tensor

# ---------- CACHE PREDICTION ----------
@st.cache_data(show_spinner=False)
def cached_prediction(_model, _img_tensor):
    _model.eval()
    with torch.no_grad():
        logits = _model(_img_tensor)
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

    transform = get_transform()
    img_tensor = preprocess_image(image, transform)

    with st.spinner("🩺 Running model prediction..."):
        pred_class, confidence, label = cached_prediction(model, img_tensor)
        st.success(f"🩺 Prediction: **{label}**  (Confidence: {confidence:.2f})")

    st.divider()
    st.subheader("🔍 Explainability Results")

    # ---------- LIME ----------
    try:
        st.markdown("### 🧩 LIME Explanation")
        progress = st.progress(0)
        for i in range(0, 60, 20):
            progress.progress(i)
            time.sleep(0.1)

        lime_img, lime_score = cached_lime(model, np.array(image), transform)

        progress.progress(100)
        st.image(lime_img, use_container_width=True)
        st.caption(f"Score: {float(lime_score):.3f}")
    except Exception as e:
        st.error(f"LIME error: {e}")

    st.divider()

    # ---------- Grad-CAM ----------
    try:
        st.markdown("### 🔥 Grad-CAM Heatmap")
        progress = st.progress(0)
        for i in range(0, 60, 20):
            progress.progress(i)
            time.sleep(0.1)

        gradcam_img, gradcam_score = generate_gradcam(model, img_tensor, image)

        progress.progress(100)
        st.image(gradcam_img, use_container_width=True)
        st.caption(f"Score: {float(gradcam_score):.3f}")
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")

    st.divider()

    # ---------- SHAP ----------
    try:
        st.markdown("### 📊 SHAP Explanation")
        progress = st.progress(0)
        for i in range(0, 60, 20):
            progress.progress(i)
            time.sleep(0.1)

        shap_values, shap_score = cached_shap(model, img_tensor)

        progress.progress(100)

        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.detach().cpu().numpy()

        if shap_values.ndim > 1:
            shap_values = np.mean(np.abs(shap_values), axis=(0, 1))

        labels = ["Red", "Green", "Blue"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, shap_values[:3])
        ax.set_title("SHAP Feature Importance")
        ax.set_ylabel("Impact")
        plt.tight_layout()

        st.pyplot(fig)
        st.caption(f"Score: {float(shap_score):.3f}")

    except Exception as e:
        st.error(f"SHAP error: {e}")

    st.divider()

    # ---------- E-Score ----------
    try:
        escore_value = e_score(model, img_tensor, label)
        st.info(f"E-Score: {float(escore_value):.3f}")
    except Exception as e:
        st.error(f"E-Score error: {e}")

    st.divider()

    # ---------- Doctor Feedback ----------
    try:
        feedback_img, feedback_text = doctor_feedback(label)
        st.image(feedback_img, width=350)
        st.caption(feedback_text)
    except Exception:
        st.warning("No doctor feedback available.")

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
