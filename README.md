# 💓 HeartRiskAI – Heart Attack Risk Prediction with AI Insights

HeartRiskAI is a smart health screening web app that predicts the risk of a heart attack using a machine learning model and provides instant, structured insights using IBM Granite (via Replicate API). Built with an intuitive Streamlit interface, this project merges healthcare analytics with generative AI to promote early awareness and preventive action.

---

## 📌 Project Overview

This project addresses the real-world need for early cardiovascular risk detection by combining predictive modeling and AI summarization. Users input their health data, receive a personalized risk score, and get professional AI-generated health recommendations.

### 🎯 Goals

- Predict heart attack risk based on lifestyle and biometric data.
- Provide actionable AI explanations using a Large Language Model (LLM).
- Empower users to understand and reduce their personal health risk.

---

## ⚙️ Features & Workflow

1. **User Input**  
   Health-related form including age, blood pressure, cholesterol, physical activity, stress level, etc.

2. **ML Risk Prediction**  
   Logistic regression model classifies into Low, Moderate, or High risk using a 0.35 threshold.

3. **AI-Powered Insight**  
   IBM Granite LLM (via Replicate) provides structured insights using SOFP (Structured Output Format Prompting):
   - **Key Points**: Highlights of major health risks.
   - **Decision Made**: Summary of why risk is classified.
   - **Suggestions**: Personalized health advice.

4. **Visual Risk Indicator**  
   Displays probability score and color-coded feedback (🔴, 🟠, 🟢).

---

## 💡 Insights & Findings

- Health risks can be communicated better with clear language.
- AI summaries improve user understanding and motivation.
- Structured prompts enhance LLM output relevance in healthcare.

---

## 🛠️ Tech Stack

- Python (Scikit-learn, Pandas)
- Streamlit (UI)
- LangChain (PromptTemplate)
- Replicate API – IBM Granite 3.3 8B Instruct (LLM)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/heaven-castle/HeartRiskAI
cd heart-risk-ai

### 2. Model And Data

data https://drive.google.com/drive/folders/11F-G2TdXi8qyiltzMW3tBV2uMukowxSK

### 3. App web
https://heartriskai.streamlit.app/


