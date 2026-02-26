import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",   # makes it responsive
    initial_sidebar_state="collapsed"
)

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import numpy as np

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

import streamlit as st



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

if st.sidebar.button("Predict Performance"):

    input_data = np.array([[study_hours, attendance, previous_marks, sleep_hours, activities]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]

    class_labels = label_encoder.inverse_transform([0,1,2])

    result = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(probabilities) * 100

    st.markdown("## 🎯 Prediction Result")

    if result == "High":
        st.success("🏆 High Performer")
        st.balloons()
    elif result == "Medium":
        st.info("📘 Medium Performer")
    else:
        st.warning("⚠ Low Performer")

    st.markdown(f"### 🔍 Model Confidence: {confidence:.2f}%")

    # Progress bar
    progress_bar = st.progress(0)
    for i in range(int(confidence)):
        progress_bar.progress(i + 1)

    # -----------------------
    # Probability Chart
    # -----------------------
    st.markdown("## 📊 Class Probability Distribution")

    prob_df = pd.DataFrame({
        "Performance Level": class_labels,
        "Probability": probabilities * 100
    })

    st.bar_chart(prob_df.set_index("Performance Level"))

