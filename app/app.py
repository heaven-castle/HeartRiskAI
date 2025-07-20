import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from prediction import load_model, predict_heart_attack
from preprocessing import preprocess_input

# LangChain & Replicate setup
from langchain_community.llms import Replicate
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# ========== PAGE CONFIGURATION ==========
st.set_page_config(page_title="Heart Risk Predictor", layout="centered")
st.title("💓 Heart Attack Risk Prediction")
st.markdown("""
This application uses a Machine Learning model to predict whether a person is at high risk of a **heart attack** based on their health data.
""")

# ========== INPUT FORM ==========
with st.form("input_form"):
    st.subheader("🩺 Enter Your Health Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 10, 100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Past", "Current"])
        blood_pressure_systolic = st.number_input("Systolic Blood Pressure", 80, 200, value=120)
        physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        hypertension = st.selectbox("Hypertension", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("Diabetes", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        obesity = st.selectbox("Obesity", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        cholesterol_hdl = st.number_input("HDL Cholesterol", 10, 100, value=55)
        cholesterol_ldl = st.number_input("LDL Cholesterol", 50, 300, value=130)
        EKG_results = st.selectbox("EKG Results", ["Normal", "Abnormal", "ST-T Abnormality"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
        air_pollution_exposure = st.selectbox("Air Pollution Exposure", ["Low", "Moderate", "High"])
        dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Healthy"])
        stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        sleep_hours = st.slider("Sleep Hours per Day", 0, 15, 7)
        previous_heart_disease = st.selectbox("History of Heart Disease", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        medication_usage = st.selectbox("Currently on Medication", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        family_history = st.selectbox("Family History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("🔍 Predict Risk")

# ========== PREDICTION ==========
if submitted:
    input_data = {
        "age": age,
        "gender": gender,
        "smoking_status": smoking_status,
        "blood_pressure_systolic": blood_pressure_systolic,
        "cholesterol_hdl": cholesterol_hdl,
        "cholesterol_ldl": cholesterol_ldl,
        "physical_activity": physical_activity,
        "EKG_results": EKG_results,
        "alcohol_consumption": alcohol_consumption,
        "air_pollution_exposure": air_pollution_exposure,
        "dietary_habits": dietary_habits,
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "obesity": obesity,
        "previous_heart_disease": previous_heart_disease,
        "medication_usage": medication_usage,
        "family_history": family_history,
    }

    df_input = pd.DataFrame([input_data])
    df_processed = preprocess_input(df_input)
    model = load_model()
    prediction, probability = predict_heart_attack(model, df_processed, threshold=0.35)

    # ========== OUTPUT ==========
    st.subheader("📊 Risk Prediction Result")
    probability = float(min(max(probability, 0.0), 1.0))
    risk_percent = probability * 100

    if probability >= 0.7:
        label = "🔴 High Risk"
        box = st.error
    elif probability >= 0.35:
        label = "🟠 Medium Risk"
        box = st.warning
    else:
        label = "🟢 Low Risk"
        box = st.success

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Probability", f"{risk_percent:.2f}%")
    with col2:
        box(f"**{label}**")

    st.markdown("**Visual Risk Level:**")
    st.progress(probability)

    st.caption("📌 This prediction is based on a Machine Learning model with a threshold of 0.35. This is not a medical diagnosis. Please consult a doctor for a professional opinion.")

    # ========== AI INSIGHT (IBM GRANITE) ==========
    st.subheader("🤖 Insight from IBM Granite AI")
    replicate_token = os.getenv("REPLICATE_API_TOKEN")

    if not replicate_token:
        st.warning("🔐 REPLICATE_API_TOKEN is not set in the environment. Insight AI is unavailable.")
    else:
        try:
            llm = Replicate(
                model="ibm-granite/granite-3.3-8b-instruct",
                replicate_api_token=replicate_token
            )

            prompt_template = """
You are an intelligent medical assistant helping assess heart attack risk based on the following patient data:

Age: {age}
Gender: {gender}
Systolic Blood Pressure: {blood_pressure_systolic}
HDL: {cholesterol_hdl}
LDL: {cholesterol_ldl}
Smoking: {smoking_status}
Physical Activity: {physical_activity}
EKG: {EKG_results}
Heart Disease History: {previous_heart_disease}
Obesity: {obesity}
Hypertension: {hypertension}
Diabetes: {diabetes}
Air Pollution Exposure: {air_pollution_exposure}
Alcohol Consumption: {alcohol_consumption}
Dietary Habits: {dietary_habits}
Stress: {stress_level}
Sleep Hours: {sleep_hours}
Family History: {family_history}
Medication: {medication_usage}

Summarize the condition in the following structured format:

### 🔍 Key Points:
- ...

### 🧠 Decision Made:
- ...

### 💡 Suggestions:
- ...
            """

            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            result = chain.invoke(input_data)

            st.info(result.strip())

        except Exception as e:
            st.error(f"❌ Error occurred while fetching AI Insight:\n\n{e}")
