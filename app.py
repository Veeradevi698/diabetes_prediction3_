import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import pandas as pd

# Load saved model and scaler
with open("finalized_model.sav", "rb") as f:
    model_rf = pickle.load(f)

with open("scaler_model.sav", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below to predict diabetes outcome:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    # Prepare input
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]).reshape(1, -1)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model_rf.predict(scaled_data)
    probability = model_rf.predict_proba(scaled_data)[0][1]

    # Output with health advice
    if prediction[0] == 1:
        st.error(f"⚠️ Diabetes Positive (Probability: {probability*100:.2f}%)")
        st.warning("⚠️ High risk of Diabetes detected.")
        st.write("👉 Suggested actions:")
        st.write("- Consult a doctor for medical advice")
        st.write("- Maintain a healthy diet (low sugar, high fiber)")
        st.write("- Exercise regularly (30 mins/day)")
        st.write("- Monitor glucose levels frequently")
    else:
        st.success(f"✅ Diabetes Negative (Probability: {(1-probability)*100:.2f}%)")
        st.success("✅ Low risk of Diabetes.")
        st.write("👉 Suggested actions:")
        st.write("- Continue healthy lifestyle")
        st.write("- Regular checkups")
        st.write("- Balanced diet and exercise")

    # Feature Importance (built-in)
    st.subheader("🔍 Feature Importance")
    importances = model_rf.feature_importances_
    features = ["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","DPF","Age"]
    st.bar_chart(dict(zip(features, importances)))

    # Patient History Tracking
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Prediction": "Positive" if prediction[0]==1 else "Negative",
        "Probability": probability
    })
    st.subheader("📊 Patient History")
    st.dataframe(st.session_state["history"])

    # Lifestyle Simulator
    st.subheader("⚡ Lifestyle Simulator")
    sim_glucose = st.slider("Simulate Glucose Level", 50, 200, glucose)
    sim_bmi = st.slider("Simulate BMI", 15.0, 40.0, bmi)
    sim_input = np.array([pregnancies, sim_glucose, blood_pressure, skin_thickness,
                          insulin, sim_bmi, dpf, age]).reshape(1, -1)
    sim_scaled = scaler.transform(sim_input)
    sim_pred = model_rf.predict(sim_scaled)
    sim_prob = model_rf.predict_proba(sim_scaled)[0][1]
    st.write(f"Simulation Result: {'Positive' if sim_pred[0]==1 else 'Negative'} (Risk {sim_prob*100:.2f}%)")

    # Risk Level Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': "Risk Level (%)"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "red" if prediction[0]==1 else "green"}}
    ))
    st.plotly_chart(fig_gauge)

    # Healthy range checks
    healthy_ranges = {
        "Glucose": (70, 140),
        "Blood Pressure": (60, 120),
        "BMI": (18.5, 24.9)
    }
    for feature, (low, high) in healthy_ranges.items():
        value = {"Glucose": glucose, "Blood Pressure": blood_pressure, "BMI": bmi}[feature]
        if value < low or value > high:
            st.warning(f"{feature} = {value} is outside healthy range ({low}-{high})")

    # Health tip of the day
    tips = [
        "Drink plenty of water 💧",
        "Exercise for at least 30 minutes daily 🏃",
        "Eat more fiber and vegetables 🥦",
        "Avoid sugary drinks 🍹",
        "Get regular health checkups 🩺"
    ]
    st.info("Tip of the Day: " + random.choice(tips))

    # Download Report
    report = f"""
    Diabetes Prediction Report
    --------------------------
    Pregnancies: {pregnancies}
    Glucose: {glucose}
    Blood Pressure: {blood_pressure}
    Skin Thickness: {skin_thickness}
    Insulin: {insulin}
    BMI: {bmi}
    DPF: {dpf}
    Age: {age}

    Prediction: {'Positive' if prediction[0]==1 else 'Negative'}
    Probability: {probability*100:.2f}%
    Advice: Maintain healthy lifestyle, regular checkups, balanced diet.
    """
    st.download_button("📄 Download Report", report, file_name="diabetes_report.txt")

    # Gamification
    if prediction[0] == 0:
        st.success("🎉 Added to Healthy Leaderboard!")

    # Doctor Connect (Prototype)
    if st.button("👨‍⚕️ Connect to Doctor"):
        st.info("Demo: This would connect you to a healthcare provider portal.")

# Batch upload for multiple patients
uploaded_file = st.file_uploader("📂 Upload patient data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    scaled = scaler.transform(df.values)
    preds = model_rf.predict(scaled)
    df["Prediction"] = preds
    st.write(df)
    st.download_button("📄 Download Results", df.to_csv(index=False), "results.csv")
