import streamlit as st import os
import numpy as np import pandas as pd import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image from tensorflow.keras.models import load_model import tempfile
import SimpleITK as sitk import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Stroke Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- Custom Styling ---
st.markdown("""
    <style>
        .big-font { font-size:20px !important; }
        .stButton > button { width: 100%; }
        .stTextInput, .stNumberInput { width: 100%; }
        .stFileUploader { border: 2px dashed #2E86C1; padding: 15px; }
        .stroke-risk { font-size:22px; font-weight:bold; color:#E74C3C; }
        .success-text { font-size:22px; font-weight:bold; color:#2ECC71; }
    </style>
""", unsafe_allow_html=True)
# --- Load Pre-trained CNN Model ---
@st.cache_resource
def load_cnn_model():
    model_path = "brain_stroke_model.h5"  # Ensure the model file exists
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"🚨 Error: Model file not found at {model_path}.")
        return None
# --- Preprocess MRI Image ---
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array
# --- Predict Stroke from MRI Image ---
def predict_image(model, img_array):
    if model is None:
        return "Model not loaded.", 0
    
    prediction = model.predict(img_array)[0][0]
    class_label = "🛑 Stroke Detected" if prediction > 0.5 else "✅ No Stroke"
    
    # Generate a simulated mRS score (Replace this with an actual model-based score if available)
    mRS_score = np.random.randint(0, 6) if prediction > 0.5 else np.random.randint(0, 3)
    mRS_outcome = "✅ Favorable Outcome (mRS ≤ 2)" if mRS_score <= 2 else "🛑 Unfavorable Outcome (mRS > 2)"
    
    return class_label, mRS_score, mRS_outcome

# --- Radiomics Feature Extraction ---
def extract_radiomics_features(image_path):
    """
    Extracts radiomics features from the provided image.

    Args:
        image_path (str): Path to the MRI image file.

    Returns:
        dict: A dictionary containing the extracted radiomics features.
    """

    try:
        # Initialize feature extractor (can load settings from a parameter file)
        # Here, we use default settings for demonstration
        # You might want to customize these for your specific application
        params = {} # You may need to specify a parameter file here
        extractor = RadiomicsFeatureExtractor(**params) # Use params dictionary here

        # Load image using SimpleITK
        image = sitk.ReadImage(image_path)

        # Create a dummy mask (all pixels are part of the region)
        size = image.GetSize()
        mask = sitk.Image(size, sitk.sitkUInt8)
        mask.SetSpacing(image.GetSpacing())
        mask.SetOrigin(image.GetOrigin())
        mask.SetDirection(image.GetDirection())

        # Set all pixels to 1 (inside the region)
        mask_array = np.ones(size, dtype=np.uint8)
        mask = sitk.GetImageFromArray(mask_array)

        # Extract features
        feature_vector = extractor.execute(image, mask)

        return feature_vector

    except Exception as e:
        st.error(f"🚨 Radiomics Feature Extraction Error: {e}")
        return None  # Or an empty dictionary, depending on how you want to handle the error

# --- Collect Clinical Features ---
def get_clinical_features():
    with st.sidebar:
        st.subheader("🩺 Enter Clinical Information")

        age = st.number_input("📅 Age", min_value=1, max_value=120, value=50)
        blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
        cholesterol = st.number_input("🩸 Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=150)
        diabetes = st.number_input("🍬 Diabetes Level (mg/dL)", min_value=50, max_value=500, value=100)
        physical_activity = st.slider("🏃 Physical Activity (hours per week)", min_value=0, max_value=20, value=5)
        heart_disease = st.radio("❤ History of Heart Disease", ["Yes", "No"])
        smoking = st.radio("🚬 Smoking Habit", ["Smoker", "Non-Smoker", "Ex-Smoker"])

        return {
            "Age": age,
            "Blood Pressure": blood_pressure,
            "Cholesterol": cholesterol,
            "Diabetes": diabetes,
            "Physical Activity": physical_activity,
            "Heart Disease": 1 if heart_disease == "Yes" else 0,
            "Smoking": smoking
        }

# --- Predict Stroke Risk from Clinical Data ---
def predict_stroke_risk(clinical_data):
    score = 0
    if clinical_data["Age"] > 55:
        score += 1.5
    if clinical_data["Blood Pressure"] > 140:
        score += 1.5
    if clinical_data["Cholesterol"] > 240:
        score += 1.3
    if clinical_data["Diabetes"] > 180:
        score += 1.6
    if clinical_data["Physical Activity"] < 30:
        score += 1.0
    if clinical_data["Heart Disease"] == 1:
        score += 1.8
    if clinical_data["Smoking"] == "Smoker":
        score += 1.4

    risk_category = "✅ Low Stroke Risk"
    if score >= 3:
        risk_category = "⚠ Moderate Stroke Risk"
    if score >= 5:
        risk_category = "🚨 High Stroke Risk"
    return score, risk_category

# --- Main App Function ---
def main():
    st.title("🧠AI Powered Model for Predicting Ischemic Stroke using MRI images and Clinical data")
    st.write("A multi-method approach to detecting stroke using MRI scans, radiomics features and clinical data analysis.")
    
    # --- Sidebar Instructions ---
    with st.sidebar:
        st.info("📌 Instructions:\n- **MRI Analysis: Upload an MRI scan.\n- **Clinical Analysis: Enter patient details.\n- Click Predict to see results.")

    # --- Tab Navigation ---
    tab1, tab2 = st.tabs(["🖼 MRI Image Analysis", "📋 Clinical Data Analysis"])
    cnn_model = load_cnn_model()

    # --- MRI Image Analysis ---
    with tab1:
        st.subheader("🖼 Upload MRI Image")
        image_types = ["nii", "nii.gz", "dcm", "png", "jpg", "jpeg"]
        uploaded_image = st.file_uploader("Upload an MRI Image (NIfTI, DICOM, PNG, JPG, JPEG)", type=image_types)

        if uploaded_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as tmp_image_file:
                tmp_image_file.write(uploaded_image.read())
                temp_image_path = tmp_image_file.name

            st.image(uploaded_image, caption="Uploaded MRI Image", width=300)

            if st.button("🔍 Analyze MRI"):
                if cnn_model:
                    processed_image = preprocess_image(temp_image_path)
                    prediction, mRS_score, mRS_outcome = predict_image(cnn_model, processed_image)

                    st.subheader("🔎 MRI Prediction Results (CNN)")
                    st.write(f"Prediction: {prediction}")
                    st.write(f"mRS Score: {mRS_score}")
                    st.markdown(f"<p class='stroke-risk'>{mRS_outcome}</p>", unsafe_allow_html=True)

                    try:
                        # Radiomics analysis
                        if uploaded_image.name.lower().endswith((".nii", ".nii.gz", ".dcm")):
                            radiomics_features = extract_radiomics_features(temp_image_path)  # Pass temporary paths

                            if radiomics_features:
                                st.subheader("📊 Radiomics Features")
                                # Display some extracted features (or process them further)
                                num_features_to_display = 10  # Limiting number of features for demonstration
                                feature_names = list(radiomics_features.keys())

                                for i in range(min(num_features_to_display, len(feature_names))):
                                    feature_name = feature_names[i]
                                    st.write(f"Feature: {feature_name}, Value: {radiomics_features[feature_name]}")
                            else:
                                st.warning("Radiomics feature extraction failed.")
                        else:
                            st.info("")

                    except Exception as e:
                        st.error(f"🚨 Error during Radiomics analysis: {e}")

                os.remove(temp_image_path)

    # --- Clinical Data Analysis ---
    with tab2:
        st.subheader("📋 Patient Clinical Data")
        patient_data = get_clinical_features()
        if st.button("🩺 Analyze Clinical Data"):
            st.subheader("📊 Clinical Data-Based Stroke Risk Assessment")
            
            # Display input values
            st.write(f"🧓 Age: {patient_data['Age']} years")
            st.write(f"💓 Blood Pressure: {patient_data['Blood Pressure']} mmHg")
            st.write(f"🩸 Cholesterol: {patient_data['Cholesterol']} mg/dL")
            st.write(f"🏃 Physical Activity: {patient_data['Physical Activity']}")
            st.write(f"❤ Heart Disease: {'✅ Yes' if patient_data['Heart Disease'] == 1 else '❌ No'}")
            st.write(f"🍬 Diabetes: {'✅ Yes' if patient_data['Diabetes'] == 1 else '❌ No'}")
            st.write(f"🚬 Smoking Habit: {patient_data['Smoking']}")

            stroke_risk_score, risk_category = predict_stroke_risk(patient_data)
            st.markdown(f"<p style='font-size:22px; font-weight:bold; color:#E74C3C;'>{risk_category}</p>", unsafe_allow_html=True)

if _name_ == "_main_":
    main()