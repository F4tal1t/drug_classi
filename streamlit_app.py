import streamlit as st
import pickle
import numpy as np
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model and encoders
@st.cache_resource
def load_model_and_encoders():
    model_path = os.path.join(current_dir, "drug_classifier_model.pkl")
    encoders_path = os.path.join(current_dir, "label_encoders.pkl")

    # Load the model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Load the encoders
    with open(encoders_path, "rb") as encoders_file:
        encoders = pickle.load(encoders_file)

    return model, encoders

# Load model and encoders
model, encoders = load_model_and_encoders()

# Streamlit app interface
st.title("Drug Classification App")
st.write("This app predicts the drug type for a patient based on their medical features.")

# Input features
st.header("Input Patient Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
sex = st.selectbox("Sex", options=["Female", "Male"])
bp = st.selectbox("Blood Pressure", options=["Low", "Normal", "High"])
cholesterol = st.selectbox("Cholesterol Level", options=["Normal", "High"])
na_to_k = st.number_input("Sodium to Potassium Ratio", min_value=0.0, step=0.1, value=10.0)

# Transform categorical inputs using encoders
sex_encoded = encoders["Sex"].transform([sex])[0]
bp_encoded = encoders["BP"].transform([bp])[0]
cholesterol_encoded = encoders["Cholesterol"].transform([cholesterol])[0]

# Prepare the input array
input_features = np.array([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    predicted_drug = encoders["Drug"].inverse_transform([prediction])[0]
    st.success(f"The predicted drug type is: **{predicted_drug}**")

# Footer
st.write("---")



