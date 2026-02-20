import streamlit as st
import torch
from PIL import Image
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

# ---------- PROJECT IMPORTS ----------
from models.fusion_model import FusionModel
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
model = FusionModel().to(device)
try:
    state_dict = torch.load("weights/resnet18_pneumonia_classifier.pth", map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
    model.cnn.load_state_dict(state_dict)
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
    """
    Generate a SHAP bar chart for RGB channels
    """
    try:
        # Run your original SHAP function
        shap_values, score = run_shap(_model, _img_tensor)  # shap_values can be tensor or array

        # Ensure it's on CPU and numpy array
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.detach().cpu().numpy()

        # Take mean absolute value across spatial dims if needed
        if shap_values.ndim > 1:
            shap_values = np.mean(np.abs(shap_values), axis=(0, 1))

        # Labels for RGB
        labels = ["Red", "Green", "Blue"]

        # Build bar chart
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, shap_values[:3], color=["red", "green", "blue"])
        ax.set_title("SHAP Feature Importance")
        ax.set_ylabel("Impact")
        plt.tight_layout()

        # Score: mean of shap values
        shap_score = float(np.mean(shap_values[:3]))

        return fig, shap_score

    except Exception as e:
        # Fallback: empty figure
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, 'SHAP not available', ha='center', va='center')
        return fig, 0.0

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
    # ---------- IMAGE ----------
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Medical Image", width=400)

    # ---------- PREPROCESS ----------
    transform = get_transform()
    img_tensor = preprocess_image(image, transform)

    # ---------- PREDICTION ----------
    with st.spinner("🩺 Running model prediction..."):
        pred_class, confidence, label = cached_prediction(model, img_tensor)
        st.success(f"🩺 Prediction: **{label}**  (Confidence: {confidence:.2f})")

    st.divider()
    st.subheader("🔍 Explainability Results")

    # ---------- LIME ----------
    try:
        st.markdown("<h4 style='text-align:center;'>🧩 LIME Explanation</h4>", unsafe_allow_html=True)
        progress_lime = st.progress(0)
        for i in range(0, 101, 20):
            progress_lime.progress(i)
            time.sleep(0.1)
        lime_img, lime_score = cached_lime(model, np.array(image), transform)
        progress_lime.progress(100)
        st.image(lime_img, use_container_width=True)
        st.caption(f"Score: {float(lime_score):.3f}")
    except Exception as e:
        st.error(f"LIME error: {e}")

    st.divider()

    # ---------- Grad-CAM ----------
    try:
        st.markdown("<h4 style='text-align:center;'>🔥 Grad-CAM Heatmap</h4>", unsafe_allow_html=True)
        progress_gradcam = st.progress(0)
        for i in range(0, 101, 20):
            progress_gradcam.progress(i)
            time.sleep(0.1)
        gradcam_img, gradcam_score = generate_gradcam(model, img_tensor, image)
        progress_gradcam.progress(100)
        st.image(gradcam_img, use_container_width=True)
        st.caption(f"Score: {float(gradcam_score):.3f}")
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")

    st.divider()

    # ---------- SHAP ----------
    try:
        st.markdown("<h4 style='text-align:center;'>📊 SHAP Explanation</h4>", unsafe_allow_html=True)
        progress_shap = st.progress(0)
        for i in range(0, 101, 20):
            progress_shap.progress(i)
            time.sleep(0.1)
        shap_fig, shap_score = cached_shap(model, img_tensor)
        progress_shap.progress(100)
        if shap_fig is not None:
            st.pyplot(shap_fig, clear_figure=True)
        else:
            st.warning("SHAP figure not generated")
        st.caption(f"Score: {float(shap_score):.3f}")
    except Exception as e:
        st.error(f"SHAP error: {e}")

    st.divider()

    # ---------- E-Score ----------
    try:
        st.markdown("<h4 style='text-align:center;'>📏 E-Score</h4>", unsafe_allow_html=True)
        with st.spinner("📏 Calculating E-Score..."):
            escore_value = e_score(model, img_tensor, label)
        st.info(f"E-Score: {float(escore_value):.3f}")
    except Exception as e:
        st.error(f"E-Score error: {e}")

    st.divider()

    # ---------- Doctor Feedback ----------
    try:
        st.markdown("<h4 style='text-align:center;'>👨‍⚕️ Doctor Feedback</h4>", unsafe_allow_html=True)
        with st.spinner("👨‍⚕️ Loading Doctor Feedback..."):
            feedback_img, feedback_text = doctor_feedback(label)
        st.image(feedback_img, width=350)
        st.caption(feedback_text)
    except Exception:
        st.warning("No doctor feedback available.")

    st.divider()

    # ---------- VALIDATION & CLINICAL FEEDBACK ----------
    st.subheader("🩺 Validation and Clinical Feedback")
    st.info("""
    Clinicians review model explanations such as Grad-CAM, LIME, and SHAP visualizations    
    to verify whether they correspond to medically relevant regions.
    """)

    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        st.checkbox("Grad-CAM aligns with medical knowledge", key="val_gradcam")
        st.checkbox("LIME highlights relevant regions", key="val_lime")
    with feedback_col2:
        st.checkbox("SHAP top features are clinically meaningful", key="val_shap")
        st.checkbox("Model explanation supports diagnosis", key="val_model")

    st.info("""
    If direct hospital access is limited, expert feedback can be collected from    
    publicly available annotated datasets or published studies.
    """)

    st.divider()

    # ---------- SAVE & SUBMIT ----------
    if st.button("💾 Save & Submit Analysis"):
        # Save patient data
        row = [
            patient_id, patient_name, patient_age, disease_present, major_surgeries,
            diabetes, bp, thyroid, cholesterol, asthma, label, confidence, float(escore_value)
        ]
        with open("patient_records.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Save validation feedback
        validation_row = [
            st.session_state.get("val_gradcam", False),
            st.session_state.get("val_lime", False),
            st.session_state.get("val_shap", False),
            st.session_state.get("val_model", False),
        ]
        with open("validation_feedback.csv", "a", newline="") as f:
            csv.writer(f).writerow(validation_row)

        st.success("✅ Patient data, analysis, E-Score, doctor feedback, and validation submitted successfully!")

        # 🔹 Refresh the app for next patient
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.stop()