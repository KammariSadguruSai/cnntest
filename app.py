import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load trained components
model = tf.keras.models.load_model('stroke_cnn_model.h5')
scaler = joblib.load('scaler.save')
label_encoders = joblib.load('label_encoders.save')

# Input features (EXCLUDING target column 'Stroke History')
features = [
    'Age', 'Gender', 'Hypertension', 'Heart Disease', 'Marital Status',
    'Work Type', 'Residence Type', 'Average Glucose Level', 'Body Mass Index (BMI)',
    'Smoking Status', 'Alcohol Intake', 'Physical Activity',
    'Family History of Stroke', 'Dietary Habits', 'Stress Levels',
    'Blood Pressure Levels', 'Cholesterol Levels', 'Symptoms'
]

st.set_page_config(page_title="Stroke Risk Prediction", page_icon="🧠")
st.title("🧠 Stroke Risk Prediction App")
st.markdown("This app predicts the **likelihood of stroke** based on patient health data.")

# Input form for user
user_input = {}
with st.form("stroke_form"):
    for feat in features:
        if feat in label_encoders:
            options = label_encoders[feat].classes_
            user_input[feat] = st.selectbox(f"{feat}", options)
        else:
            user_input[feat] = st.number_input(f"{feat}", min_value=0.0, step=0.1)
    
    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

if submitted:
    try:
        # Encode categorical features
        input_vals = []
        for feat in features:
            val = user_input[feat]
            if feat in label_encoders:
                val = label_encoders[feat].transform([val])[0]
            input_vals.append(val)

        # Scale and reshape for CNN
        input_array = scaler.transform([input_vals])
        input_array = input_array.reshape((1, input_array.shape[1], 1))

        # Predict
        pred_prob = model.predict(input_array)[0][0]
        prediction = "⚠️ Stroke Risk Detected" if pred_prob > 0.5 else "✅ No Stroke Risk Detected"

        st.subheader("Prediction Result:")
        st.success(prediction)
        st.info(f"Prediction Confidence: `{pred_prob:.2f}`")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
