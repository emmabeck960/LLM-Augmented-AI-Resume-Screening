import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.resume_parser import clean_text

# Load dataset
df = pd.read_csv("data/Resume.csv")

# Clean resume text
df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

# Create Hired label based on category
shortlisted_roles = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "BANKING",
    "FINANCE",
    "ACCOUNTANT",
    "CONSULTANT",
    "DIGITAL-MEDIA",
    "SALES"
]

df["Hired"] = df["Category"].apply(
    lambda x: 1 if x in shortlisted_roles else 0
)

print("Label distribution:")
print(df["Hired"].value_counts())

# Features & labels
X = df["cleaned_resume"]
y = df["Hired"]

# ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=6000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/resume_model.pkl")

print("✅ Resume Screening Model Trained Successfully!")


