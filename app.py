import streamlit as st
import numpy as np
import joblib

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🎓 Student Performance Predictor")

st.write("Enter student details below:")

study_hours = st.slider("Study Hours per Day", 1, 10)
attendance = st.slider("Attendance (%)", 50, 100)
previous_marks = st.slider("Previous Marks", 0, 100)
sleep_hours = st.slider("Sleep Hours", 4, 10)
activity_option = st.radio(
    "Participates in Activities?",
    ["Yes", "No"]
)

activities = 1 if activity_option == "Yes" else 0

if st.button("Predict Performance"):

    input_data = np.array([[study_hours, attendance, previous_marks, sleep_hours, activities]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    result = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Performance: {result[0]}")