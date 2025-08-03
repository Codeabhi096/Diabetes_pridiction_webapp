import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('svm_diabetes_model.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="🩺 Diabetes Prediction", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Diabetes Prediction Web App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your health parameters below to predict diabetes risk.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=1000)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120)

    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error(" The model predicts: **Diabetic**")
    else:
        st.success(" The model predicts: **Not Diabetic**")
