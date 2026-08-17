# 🩺 Diabetes Prediction App

An interactive **Streamlit-based web application** that predicts diabetes risk using a trained machine learning model.  
Built for hackathons and educational purposes, this app combines **prediction, simulation, health tips, and a virtual assistant (VANI)** to raise awareness about diabetes.

---

## 🌐 Live Demo
👉 Try the app here: [Diabetes Prediction App on Streamlit](https://diabetesprediction3-27.streamlit.app/)

---

## 🚀 Features
- **Diabetes Risk Prediction**  
  Enter patient details (glucose, BMI, age, etc.) and get a prediction with probability.

- **Health Advice**  
  Personalized lifestyle suggestions based on prediction results.

- **Feature Importance**  
  Visual bar chart showing which health factors influenced the prediction.

- **Patient History Tracking**  
  Keeps a record of past predictions during the session.

- **Lifestyle Simulator**  
  Adjust glucose and BMI sliders to see how lifestyle changes affect risk.

- **Risk Gauge**  
  Interactive Plotly gauge chart to visualize risk percentage.

- **Healthy Range Checks**  
  Warns if glucose, blood pressure, or BMI are outside healthy ranges.

- **Health Tip of the Day**  
  Random motivational tips for better health.

- **Report Download**  
  Generate and download a text report of patient results.

- **Gamification**  
  Adds low-risk patients to a “Healthy Leaderboard.”

- **Doctor Connect (Prototype)**  
  Demo button to simulate connecting with healthcare providers.

- **Batch Upload**  
  Upload CSV files for multiple patient predictions at once.

- **🤖 VANI Assistant**  
  A rule-based chatbot that explains risk scores, BMI, glucose, and how the ML model works.

---

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python (Random Forest model with scikit-learn)
- **Visualization**: Matplotlib, Plotly
- **Data Handling**: Pandas, NumPy
- **Model Files**: `finalized_model.sav`, `scaler_model.sav`

---

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-app.git
   cd diabetes-prediction-app
