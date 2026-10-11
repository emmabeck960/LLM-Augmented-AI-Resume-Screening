import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    return text

def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            keywords.append(token.text)
    return " ".join(keywords)
