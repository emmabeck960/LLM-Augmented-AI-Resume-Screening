import streamlit as st
import joblib
from utils.resume_parser import clean_text
from utils.llm_explainer import explain_resume

# Load trained ML model
model = joblib.load("models/resume_model.pkl")

st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("🧠 AI Resume Screening System")
st.write("Upload a resume and check if it gets shortlisted")

uploaded_file = st.file_uploader("Upload Resume (TXT only)", type=["txt"])

if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")
    cleaned_text = clean_text(resume_text)

    # Get probability
    probability = model.predict_proba([cleaned_text])[0][1]

    # Threshold-based decision
    prediction = 1 if probability >= 0.35 else 0

    st.subheader("Result")

    if prediction == 1:
        st.success(f"✅ Shortlisted (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Not Shortlisted (Confidence: {probability:.2f})")

    # ---- LLM EXPLANATION (phi-3:mini) ----
    with st.spinner("🧠 phi-3:mini is explaining the decision..."):
        explanation = explain_resume(resume_text, prediction, probability)

    st.subheader("🤖 AI Explanation (phi-3:mini)")
    st.write(explanation)
