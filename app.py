import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load model, scaler, and encoders
model = tf.keras.models.load_model('stroke_cnn_model.h5')
scaler = joblib.load('scaler.save')
label_encoders = joblib.load('label_encoders.save')

# Input features
features = [
    'Age', 'Gender', 'Hypertension', 'Heart Disease', 'Marital Status',
    'Work Type', 'Residence Type', 'Average Glucose Level', 'Body Mass Index (BMI)',
    'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Stroke History',
    'Family History of Stroke', 'Dietary Habits', 'Stress Levels',
    'Blood Pressure Levels', 'Cholesterol Levels', 'Symptoms'
]

st.title("🧠 Stroke Prediction App")
st.write("Enter patient details below to predict the risk of stroke.")

# Input form
user_input = {}
for feat in features:
    if feat in label_encoders:
        options = label_encoders[feat].classes_
        user_input[feat] = st.selectbox(feat, options)
    else:
        user_input[feat] = st.number_input(feat, min_value=0.0)

# Predict button
if st.button("Predict Stroke Risk"):
    try:
        # Encode input
        input_vals = []
        for feat in features:
            val = user_input[feat]
            if feat in label_encoders:
                val = label_encoders[feat].transform([val])[0]
            input_vals.append(val)

        # Scale and reshape
        input_array = scaler.transform([input_vals])
        input_array = input_array.reshape((1, input_array.shape[1], 1))

        # Predict
        pred_prob = model.predict(input_array)[0][0]
        prediction = "⚠️ Stroke Risk" if pred_prob > 0.5 else "✅ No Stroke Risk"

        st.subheader("Prediction:")
        st.write(prediction)
        st.write(f"Prediction Probability: `{pred_prob:.2f}`")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
