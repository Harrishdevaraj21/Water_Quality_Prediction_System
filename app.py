# 📦 Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# 🔍 Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# 🧪 Web App Title
st.markdown("<h1 style='color:#1f77b4;'>💧 Water Pollutants Predictor 💧</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>🔬 Predict the water pollutants based on <b>Year</b> and <b>Station ID</b></p>", unsafe_allow_html=True)

# 📅 User inputs
year_input = st.number_input("📆 Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("🏢 Enter Station ID", value='1')

# 🧮 Prediction Section
if st.button('🚀 Predict'):
    if not station_id:
        st.warning('⚠️ Please enter the station ID')
    else:
        # 🛠 Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # 🧩 Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # 🤖 Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O₂ 🫧', 'NO₃ ❄️', 'NO₂ 🌫️', 'SO₄ 🌋', 'PO₄ 🧪', 'Cl 🧂']

        st.markdown(f"<h3 style='color:#2ca02c;'>📊 Predicted pollutant levels for Station <b>{station_id}</b> in <b>{year_input}</b>:</h3>", unsafe_allow_html=True)
        
        # Display each pollutant with value
        for p, val in zip(pollutants, predicted_pollutants):
            st.markdown(f"<p style='font-size:17px;'>✅ <b>{p}:</b> <span style='color:#d62728;'>{val:.2f}</span></p>", unsafe_allow_html=True)
