import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DiabetesGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f7fbff 0%,
            #eef8ff 50%,
            #ffffff 100%
        );
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #087ea4;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        padding: 22px;
        border-radius: 18px;
        background: white;
        border: 1px solid #dbeafe;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #087ea4;
    }

    /* VANI */
    .vani-card {
        padding: 20px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #eefaff,
            #ffffff
        );
        border: 1px solid #ccecf7;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
    }

    .vani-avatar {
        font-size: 52px;
        text-align: center;
    }

    .vani-title {
        font-size: 25px;
        font-weight: 800;
        color: #087ea4;
        text-align: center;
    }

    .vani-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 13px;
    }

    /* Disclaimer */
    .disclaimer {
        padding: 15px;
        border-radius: 12px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        font-size: 13px;
        margin-top: 20px;
    }

    /* Metric cards */
    .metric-card {
        padding: 20px;
        border-radius: 16px;
        background: white;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #087ea4;
    }

    .metric-label {
        font-size: 13px;
        color: #64748b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model():

    try:
        with open("finalized_model.sav", "rb") as f:
            model = pickle.load(f)

        with open("scaler_model.sav", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except FileNotFoundError:

        st.error(
            "Model files were not found. "
            "Please make sure finalized_model.sav and "
            "scaler_model.sav are in the same folder as app.py."
        )

        st.stop()


model_rf, scaler = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_probability" not in st.session_state:
    st.session_state.last_probability = None

if "last_risk_level" not in st.session_state:
    st.session_state.last_risk_level = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

if "vani_messages" not in st.session_state:

    st.session_state.vani_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm VANI 🤖, your Diabetes Risk Assistant.\n\n"
                "I can help you understand your model-estimated "
                "risk, explain health parameters and answer "
                "general diabetes-awareness questions."
            )
        }
    ]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

FEATURES = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree Function",
    "Age"
]


def get_risk_level(probability):
    """
    Application-defined risk categories.

    These are NOT clinical diagnostic categories.
    """

    if probability < 0.30:
        return "Lower"

    elif probability < 0.60:
        return "Moderate"

    else:
        return "Higher"


def get_prediction_text(prediction, probability):

    if prediction == 1:
        return (
            "Higher model-estimated risk",
            f"{probability * 100:.2f}%"
        )

    return (
        "Lower model-estimated risk",
        f"{(1 - probability) * 100:.2f}%"
    )


def make_prediction(input_values):

    input_data = np.array(input_values).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    prediction = int(model_rf.predict(scaled_data)[0])

    probability = float(
        model_rf.predict_proba(scaled_data)[0][1]
    )

    risk_level = get_risk_level(probability)

    return prediction, probability, risk_level


def vani_response(
    question,
    risk_probability=None,
    risk_level=None,
    prediction=None,
    inputs=None
):

    q = question.lower().strip()

    # --------------------------------------------------------
    # CURRENT RESULT
    # --------------------------------------------------------

    if (
        "risk score" in q
        or "risk" in q
        or "prediction" in q
        or "result" in q
        or "probability" in q
    ):

        if risk_probability is None:

            return (
                "Please complete the diabetes risk assessment "
                "first. Once you have a prediction, I can "
                "explain your result."
            )

        percentage = risk_probability * 100

        return (
            f"Your current model-estimated probability is "
            f"**{percentage:.2f}%**.\n\n"
            f"The application classifies this as **{risk_level}** "
            f"according to its current screening thresholds.\n\n"
            "This is an ML-based screening estimate, not a "
            "medical diagnosis."
        )

    # --------------------------------------------------------
    # WHY HIGH
    # --------------------------------------------------------

    if (
        "why" in q
        and (
            "high" in q
            or "higher" in q
            or "risk" in q
        )
    ):

        if inputs is None:

            return (
                "Complete a prediction first and I can help "
                "explain the result."
            )

        important_features = []

        if inputs["Glucose"] >= 140:
            important_features.append("glucose")

        if inputs["BMI"] >= 25:
            important_features.append("BMI")

        if inputs["Age"] >= 45:
            important_features.append("age")

        if inputs["Blood Pressure"] >= 80:
            important_features.append("blood pressure")

        if important_features:

            factors = ", ".join(important_features)

            return (
                "Some of the entered values that may be "
                f"important to the model include **{factors}**.\n\n"
                "These observations describe model inputs and "
                "do not establish medical causation or a diagnosis."
            )

        return (
            "The prediction is based on the combination of "
            "all input features learned by the machine-learning "
            "model. A single feature should not be interpreted "
            "as the cause of the prediction."
        )

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if "bmi" in q:

        return (
            "**BMI** stands for Body Mass Index.\n\n"
            "It is calculated using height and weight and is "
            "commonly used as one general screening measurement.\n\n"
            "BMI alone does not provide a complete assessment "
            "of a person's health."
        )

    # --------------------------------------------------------
    # GLUCOSE
    # --------------------------------------------------------

    if (
        "glucose" in q
        or "blood sugar" in q
        or "sugar level" in q
    ):

        return (
            "**Glucose** is a type of sugar used by the body "
            "for energy.\n\n"
            "Blood glucose measurements are important when "
            "evaluating diabetes, but their interpretation "
            "depends on factors such as whether the measurement "
            "was fasting and the individual's clinical situation."
        )

    # --------------------------------------------------------
    # BLOOD PRESSURE
    # --------------------------------------------------------

    if (
        "blood pressure" in q
        or "bp" in q
    ):

        return (
            "Blood pressure measures the force of blood against "
            "the walls of your arteries.\n\n"
            "It is one of the health parameters included in "
            "this screening model. Individual readings should "
            "be interpreted in their proper clinical context."
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if (
        "how" in q
        and (
            "model" in q
            or "work" in q
            or "prediction" in q
            or "machine learning" in q
        )
    ):

        return (
            "The application uses a trained machine-learning "
            "classification model.\n\n"
            "Your entered health parameters are first transformed "
            "using the saved preprocessing scaler. The processed "
            "data is then passed to the trained model, which "
            "produces a prediction and estimated probability."
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if (
        "feature" in q
        or "important factor" in q
        or "important factors" in q
    ):

        return (
            "The application displays Random Forest feature "
            "importance values to show which input variables "
            "were relatively important to the trained model.\n\n"
            "Feature importance indicates model influence; it "
            "does not mean that a feature directly causes diabetes."
        )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    if (
        "diagnos" in q
        or "doctor" in q
        or "medical" in q
    ):

        return (
            "No. 🤍\n\n"
            "VANI and this application do **not diagnose diabetes**.\n\n"
            "The system provides an ML-based screening estimate "
            "for educational and awareness purposes. A qualified "
            "healthcare professional should evaluate your health "
            "and appropriate clinical tests before making a diagnosis."
        )

    # --------------------------------------------------------
    # LIFESTYLE
    # --------------------------------------------------------

    if (
        "lifestyle" in q
        or "exercise" in q
        or "diet" in q
        or "food" in q
        or "prevent" in q
        or "prevention" in q
    ):

        return (
            "General healthy habits can include regular physical "
            "activity, balanced nutrition, adequate sleep, and "
            "following advice from your healthcare professional.\n\n"
            "For personalized diet or exercise recommendations, "
            "please consult a qualified healthcare professional."
        )

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if q in ["hi", "hello", "hey", "hii", "good morning"]:

        return (
            "Hi! 👋 I'm VANI.\n\n"
            "I can help explain your diabetes-risk screening "
            "result, health parameters and how the ML model works."
        )

    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if "thank" in q:

        return (
            "You're welcome! 🤖💙\n\n"
            "I'm here to help explain your screening result."
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "I'm VANI 🤖. I can help with:\n\n"
        "• What your risk score means\n"
        "• Why the model produced a result\n"
        "• BMI and glucose information\n"
        "• Feature importance\n"
        "• How the ML model works\n"
        "• Whether the result is a diagnosis\n"
        "• General lifestyle awareness\n\n"
        "Try asking: **What does my risk score mean?**"
    )


# ============================================================
# VANI ASSISTANT
# ============================================================

def vani_assistant():

    risk_probability = st.session_state.last_probability
    risk_level = st.session_state.last_risk_level
    prediction = st.session_state.last_prediction
    inputs = st.session_state.last_inputs

    with st.sidebar:

        st.markdown(
            """
            <div class="vani-card">

                <div class="vani-avatar">🤖</div>

                <div class="vani-title">
                    VANI
                </div>

                <div class="vani-subtitle">
                    Virtual AI Nutrition & Health Assistant
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        if risk_probability is not None:

            st.info(
                f"Current model estimate: "
                f"**{risk_probability * 100:.1f}%**"
            )

            st.write(
                f"App-defined category: **{risk_level}**"
            )

        else:

            st.info(
                "Complete a prediction to let VANI "
                "explain your result."
            )

        st.markdown("---")

        st.markdown("### 💬 Quick Questions")

        quick_question = st.selectbox(
            "Choose a question",
            [
                "Choose a question",
                "What does my risk score mean?",
                "Why is my risk high?",
                "What is BMI?",
                "What is glucose?",
                "How does the prediction work?",
                "What are the important features?",
                "Is this a medical diagnosis?",
                "What lifestyle habits are helpful?"
            ],
            key="vani_quick_question"
        )

        if (
            quick_question != "Choose a question"
            and quick_question != st.session_state.get(
                "last_quick_question"
            )
        ):

            st.session_state.last_quick_question = quick_question

            st.session_state.vani_messages.append(
                {
                    "role": "user",
                    "content": quick_question
                }
            )

            response = vani_response(
                quick_question,
                risk_probability,
                risk_level,
                prediction,
                inputs
            )

            st.session_state.vani_messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        st.markdown("### Conversation")

        for message in st.session_state.vani_messages:

            avatar = (
                "🤖"
                if message["role"] == "assistant"
                else "👤"
            )

            with st.chat_message(
                message["role"],
                avatar=avatar
            ):
                st.markdown(message["content"])

        # ----------------------------------------------------
        # CHAT INPUT
        # ----------------------------------------------------

        user_question = st.chat_input(
            "Ask VANI something...",
            key="vani_chat_input"
        )

        if user_question:

            st.session_state.vani_messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            response = vani_response(
                user_question,
                risk_probability,
                risk_level,
                prediction,
                inputs
            )

            st.session_state.vani_messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 DiabetesGuard AI")

    st.caption(
        "Machine-learning based diabetes risk screening"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Risk Assessment",
            "📊 Patient History",
            "📂 Batch Prediction",
            "🤖 VANI Assistant"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        **Important**

        This application provides an ML-based screening
        estimate for educational and awareness purposes.

        It is not a medical diagnosis.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 DiabetesGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered diabetes risk screening with VANI assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# RISK ASSESSMENT PAGE
# ============================================================

if page == "🏠 Risk Assessment":

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        🔎 Enter Health Information
        </div>

        <p>
        Enter the required health parameters and generate
        an ML-based diabetes risk screening estimate.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=300,
            value=120,
            step=1
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70,
            step=1
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80,
            step=1
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Assess Diabetes Risk",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_values = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]

        prediction, probability, risk_level = make_prediction(
            input_values
        )

        # Save prediction
        st.session_state.last_prediction = prediction
        st.session_state.last_probability = probability
        st.session_state.last_risk_level = risk_level

        st.session_state.last_inputs = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "Blood Pressure": blood_pressure,
            "Skin Thickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "Diabetes Pedigree Function": dpf,
            "Age": age
        }

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Glucose": glucose,
                "BMI": bmi,
                "Age": age,
                "Prediction": (
                    "Higher model-estimated risk"
                    if prediction == 1
                    else "Lower model-estimated risk"
                ),
                "Model Probability": round(
                    probability * 100, 2
                ),
                "Category": risk_level
            }
        )

    # ========================================================
    # DISPLAY LAST RESULT
    # ========================================================

    if st.session_state.last_probability is not None:

        prediction = st.session_state.last_prediction
        probability = st.session_state.last_probability
        risk_level = st.session_state.last_risk_level
        inputs = st.session_state.last_inputs

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {probability * 100:.1f}%
                </div>

                <div class="metric-label">
                Model-estimated probability
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            result_text = (
                "Higher"
                if prediction == 1
                else "Lower"
            )

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {result_text}
                </div>

                <div class="metric-label">
                Model-estimated risk direction
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col3:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {risk_level}
                </div>

                <div class="metric-label">
                App-defined category
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("")

        if prediction == 1:

            st.warning(
                "⚠️ The model predicts a higher-risk outcome "
                "for the entered information."
            )

        else:

            st.success(
                "✅ The model predicts a lower-risk outcome "
                "for the entered information."
            )

        st.info(
            "This result is an ML-based screening estimate "
            "and is not a medical diagnosis."
        )

        # ----------------------------------------------------
        # RISK GAUGE
        # ----------------------------------------------------

        st.subheader("🎯 Risk Visualization")

        gauge_color = (
            "#dc2626"
            if prediction == 1
            else "#16a34a"
        )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={
                    "text": "Model Probability (%)"
                },
                number={
                    "suffix": "%"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": gauge_color
                    },
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#dcfce7"
                        },
                        {
                            "range": [30, 60],
                            "color": "#fef9c3"
                        },
                        {
                            "range": [60, 100],
                            "color": "#fee2e2"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            fig_gauge,
            use_container_width=True
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.subheader("🔍 Feature Importance")

        if hasattr(model_rf, "feature_importances_"):

            importances = model_rf.feature_importances_

            importance_df = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Importance": importances
                }
            ).sort_values(
                "Importance",
                ascending=False
            )

            st.bar_chart(
                importance_df.set_index("Feature")
            )

            st.caption(
                "Feature importance shows the relative influence "
                "of input variables within the trained Random "
                "Forest model. It does not establish causation."
            )

        # ----------------------------------------------------
        # REFERENCE RANGE ALERTS
        # ----------------------------------------------------

        st.subheader("📌 Reference Range Alerts")

        reference_ranges = {
            "Glucose": (70, 140),
            "Blood Pressure": (60, 120),
            "BMI": (18.5, 24.9)
        }

        values = {
            "Glucose": glucose,
            "Blood Pressure": blood_pressure,
            "BMI": bmi
        }

        for feature, (low, high) in reference_ranges.items():

            value = values[feature]

            if value < low or value > high:

                st.warning(
                    f"{feature} = {value} is outside the "
                    f"application reference range "
                    f"({low}–{high})."
                )

            else:

                st.success(
                    f"{feature} = {value} is within the "
                    f"application reference range."
                )

        # ----------------------------------------------------
        # LIFESTYLE SIMULATOR
        # ----------------------------------------------------

        st.subheader("⚡ Lifestyle What-If Simulator")

        st.write(
            "Change glucose or BMI to explore how the "
            "model output changes for a hypothetical scenario."
        )

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:

            sim_glucose = st.slider(
                "Simulate Glucose",
                min_value=50,
                max_value=200,
                value=int(glucose)
            )

        with sim_col2:

            sim_bmi = st.slider(
                "Simulate BMI",
                min_value=15.0,
                max_value=40.0,
                value=float(bmi)
            )

        sim_values = [
            pregnancies,
            sim_glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            sim_bmi,
            dpf,
            age
        ]

        sim_prediction, sim_probability, _ = make_prediction(
            sim_values
        )

        st.info(
            f"Simulation result: "
            f"**{'Higher-risk outcome' if sim_prediction == 1 else 'Lower-risk outcome'}** "
            f"with model probability "
            f"**{sim_probability * 100:.2f}%**."
        )

        st.caption(
            "This is a hypothetical model simulation and "
            "does not predict actual future health outcomes."
        )

        # ----------------------------------------------------
        # GENERAL HEALTH GUIDANCE
        # ----------------------------------------------------

        st.subheader("🌱 General Health Awareness")

        if prediction == 1:

            st.write(
                "• Consider discussing the result with a "
                "qualified healthcare professional."
            )

            st.write(
                "• Maintain balanced nutrition and regular "
                "physical activity as appropriate."
            )

            st.write(
                "• Follow medical advice and recommended "
                "health checkups."
            )

        else:

            st.write(
                "• Continue healthy lifestyle habits."
            )

            st.write(
                "• Maintain balanced nutrition and regular "
                "physical activity."
            )

            st.write(
                "• Continue appropriate routine health checkups."
            )

        # ----------------------------------------------------
        # VANI CONTEXT CARD
        # ----------------------------------------------------

        st.subheader("🤖 Ask VANI About Your Result")

        st.markdown(
            f"""
            <div class="vani-card">

            <div style="font-size:40px;">
            🤖
            </div>

            <b>Hi! I'm VANI.</b>

            <p>
            Your current model-estimated probability is
            <b>{probability * 100:.1f}%</b>.
            </p>

            <p>
            You can ask me why the model produced this result,
            what BMI or glucose means, or whether this is a
            medical diagnosis.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        st.subheader("📄 Download Assessment Report")

        prediction_label = (
            "Higher model-estimated risk"
            if prediction == 1
            else "Lower model-estimated risk"
        )

        report = f"""
DIABETESGUARD AI
Diabetes Risk Screening Report
================================

Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PATIENT INPUTS
--------------

Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {blood_pressure}
Skin Thickness: {skin_thickness}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}

MODEL RESULT
------------

Prediction:
{prediction_label}

Model-estimated probability:
{probability * 100:.2f}%

Application-defined category:
{risk_level}

IMPORTANT
---------

This application provides an ML-based screening estimate
for educational and health-awareness purposes.

It is NOT a medical diagnosis.

Please consult a qualified healthcare professional
for medical evaluation and diagnosis.
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="diabetesguard_report.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# PATIENT HISTORY PAGE
# ============================================================

elif page == "📊 Patient History":

    st.subheader("📊 Patient Assessment History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download History CSV",
            history_df.to_csv(index=False),
            file_name="patient_history.csv",
            mime="text/csv"
        )

        if st.button(
            "🗑️ Clear History",
            type="secondary"
        ):

            st.session_state.history = []

            st.success(
                "History cleared."
            )

            st.rerun()

    else:

        st.info(
            "No predictions have been recorded in this session yet."
        )


# ============================================================
# BATCH PREDICTION PAGE
# ============================================================

elif page == "📂 Batch Prediction":

    st.subheader("📂 Batch Patient Prediction")

    st.write(
        "Upload a CSV file containing the eight model input "
        "columns to generate predictions for multiple records."
    )

    st.code(
        """
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
        """,
        language="text"
    )

    uploaded_file = st.file_uploader(
        "Upload patient CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            missing_columns = [
                feature
                for feature in FEATURES
                if feature not in df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                batch_input = df[
                    FEATURES
                ].copy()

                scaled = scaler.transform(
                    batch_input.values
                )

                predictions = model_rf.predict(
                    scaled
                )

                probabilities = (
                    model_rf.predict_proba(
                        scaled
                    )[:, 1]
                )

                df["Prediction"] = [
                    (
                        "Higher model-estimated risk"
                        if p == 1
                        else "Lower model-estimated risk"
                    )
                    for p in predictions
                ]

                df["Model Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                df["Category"] = [
                    get_risk_level(p)
                    for p in probabilities
                ]

                st.success(
                    f"Successfully processed {len(df)} records."
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "📄 Download Batch Results",
                    df.to_csv(index=False),
                    file_name="diabetes_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Could not process the CSV file: {e}"
            )


# ============================================================
# VANI PAGE
# ============================================================

elif page == "🤖 VANI Assistant":

    st.subheader("🤖 VANI — Your Diabetes Risk Assistant")

    st.markdown(
        """
        <div class="vani-card">

        <div class="vani-avatar">
        🤖
        </div>

        <div class="vani-title">
        Meet VANI
        </div>

        <div class="vani-subtitle">
        Virtual AI Nutrition & Health Assistant
        </div>

        <br>

        VANI helps explain your machine-learning screening
        result and provides general diabetes-awareness
        information.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    # Current result
    if st.session_state.last_probability is not None:

        st.info(
            f"Current model estimate: "
            f"{st.session_state.last_probability * 100:.2f}% "
            f"| Category: "
            f"{st.session_state.last_risk_level}"
        )

    # Chat history

    for message in st.session_state.vani_messages:

        avatar = (
            "🤖"
            if message["role"] == "assistant"
            else "👤"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.markdown(
                message["content"]
            )

    user_question = st.chat_input(
        "Ask VANI something..."
    )

    if user_question:

        st.session_state.vani_messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        response = vani_response(
            user_question,
            st.session_state.last_probability,
            st.session_state.last_risk_level,
            st.session_state.last_prediction,
            st.session_state.last_inputs
        )

        st.session_state.vani_messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        st.rerun()

    if st.button(
        "🗑️ Clear VANI Conversation"
    ):
        st.session_state.vani_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm VANI 🤖. "
                    "How can I help you understand "
                    "your diabetes-risk screening result?"
                )
            }
        ]

        st.rerun()


# ============================================================
# SIDEBAR VANI
# ============================================================

if page != "🤖 VANI Assistant":

    # Show VANI in sidebar
    vani_assistant()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    ⚠️ <b>Health Disclaimer:</b>
    DiabetesGuard AI provides a machine-learning-based
    screening estimate for educational and awareness
    purposes. It is not intended to diagnose, treat,
    cure or prevent any disease. Please consult a qualified
    healthcare professional for medical evaluation.

    </div>
    """,
    unsafe_allow_html=True
)
