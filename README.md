# 🧠 AI-Powered Resume Screening System

An AI-powered Resume Screening System that uses **Machine Learning + NLP + Generative AI (phi-3:mini)** to automatically shortlist resumes and explain decisions in natural language.

This project is built **100% free**, using open-source tools and Kaggle data.

---

## 🚀 Features

- 📄 Resume classification (Shortlisted / Not Shortlisted)
- 🧠 NLP-based resume understanding
- 📊 Confidence score using ML model
- 🤖 LLM explanation using **phi-3:mini**
- 🌐 Interactive dashboard with **Streamlit**
- 🔓 Fully free & offline (no paid APIs)

---

## 🛠 Tech Stack

- **Python**
- **scikit-learn**
- **spaCy**
- **Streamlit**
- **Ollama (phi-3:mini)**
- **Kaggle Resume Dataset**

---

## 📂 Project Structure

ai_resume_screening/
│
├── app.py # Streamlit app
├── train_model.py # Model training script
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore
│
├── models/
│ └── resume_model.pkl # Trained ML model
│
├── utils/
│ ├── resume_parser.py # Text cleaning & preprocessing
│ └── llm_explainer.py # phi-3 LLM explanation logic
│
└── data/
└── resumes.csv # Kaggle dataset

---

## 📊 Dataset

- Source: **Kaggle Resume Dataset**
- Labels converted to:
  - `1` → Hired / Shortlisted
  - `0` → Not Hired

## ⏳ Note on LLM Response Time

This project uses the **free and local LLM (phi-3:mini via Ollama)**.

- The **first request may take some time** to respond
- Model loading depends on:
  - System RAM & CPU
  - First-time model initialization
- Subsequent responses are **much faster**

⚠️ This delay is **expected behavior** and not a bug.
