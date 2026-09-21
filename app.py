import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Diabetes Prediction App")

st.write("Enter the patient details to predict diabetes.")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)

Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)

BloodPressure = st.number_input(
    "Blood Pressure", min_value=0, max_value=200, value=70
)

SkinThickness = st.number_input(
    "Skin Thickness", min_value=0, max_value=100, value=20
)

Insulin = st.number_input(
    "Insulin", min_value=0, max_value=900, value=80
)

BMI = st.number_input(
    "BMI", min_value=0.0, max_value=70.0, value=25.0
)

DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

Age = st.number_input(
    "Age", min_value=1, max_value=120, value=30
)

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age]
    })

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No Diabetes")