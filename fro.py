import streamlit as st
import numpy as np
import pickle

# Load the trained model
model_filename = "diabetesmodel.sav"
try:
    loaded_model = pickle.load(open(model_filename, 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'diabetesmodel.sav' exists in the directory.")
    st.stop()


st.title("Diabetes Prediction App")
st.write("Enter the details below to predict the risk of diabetes.")


pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=60.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)


input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

if st.button("Predict"):
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        st.error(" (1) The model predicts a high risk of diabetes.")
    else:
        st.success(" (0) The model predicts a low risk of diabetes.")
