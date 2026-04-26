import streamlit as st
import joblib
import numpy as np

# 🔗 Load trained components
model = joblib.load("models/crop_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/risk_encoder.pkl")

# 🌾 UI Title
st.title("🌾 Crop Risk Prediction System")

st.write("Enter soil and weather details:")

# 🧾 Input fields
N = st.number_input("Nitrogen (N)", 0.0)
P = st.number_input("Phosphorus (P)", 0.0)
K = st.number_input("Potassium (K)", 0.0)

temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall (mm)")

# 🚀 Prediction
if st.button("Predict"):

    # 🔥 Feature Engineering (MUST MATCH TRAINING)
    water_stress = temperature - rainfall * 0.1
    soil_quality = (N + P + K) / 3

    sample = np.array([[
        N, P, K, temperature, humidity, ph, rainfall,
        water_stress, soil_quality
    ]])

    # 🔗 Scaling
    sample_scaled = scaler.transform(sample)

    # 🤖 Prediction
    pred = model.predict(sample_scaled)
    result = encoder.inverse_transform(pred)[0]

    # 🎯 Output
    st.success(f"🌱 Predicted Risk: {result}")

    # 💡 Suggestion
    if result == "High Risk":
        st.warning("⚠️ Use strong pesticide + increase irrigation")
    elif result == "Medium Risk":
        st.info("⚡ Monitor crop and use moderate pesticide")
    else:
        st.success("✅ Crop is healthy, no action needed")