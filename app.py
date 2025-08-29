import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re

# Load model + tokenizer
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to clean text
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9%$., ]', '', text)

# Function to get sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    return probs

# Function to extract % change from text
def extract_percentage(text):
    matches = re.findall(r'(\d+\.?\d*)\s*%', text)
    if matches:
        return float(matches[0])
    return 0.0

# Function to predict stock movement with cap
def predict_stock_movement(text, sentiment_label):
    raw_pct = extract_percentage(text)

    # Cap stock movement at +/- 10%
    cap = 10.0
    movement = min(raw_pct, cap)

    if sentiment_label == "Positive":
        return round(movement, 2)
    elif sentiment_label == "Negative":
        return round(-movement, 2)
    else:
        return 0.0

# Streamlit App
st.title("Finance News Sentiment & Stock Movement Predictor")
user_input = st.text_area("Enter stock news, tweets, or finance text:")

if user_input:
    cleaned = clean_text(user_input)
    probs = predict_sentiment(cleaned)

    labels = ["Negative", "Neutral", "Positive"]
    sentiment_idx = np.argmax(probs)
    sentiment_label = labels[sentiment_idx]

    stock_movement = predict_stock_movement(cleaned, sentiment_label)

    st.write("### Sentiment Probabilities:")
    for label, prob in zip(labels, probs):
        st.write(f"{label}: {prob:.4f}")

    st.write("### Predicted Sentiment & Stock Movement:")
    st.write(f"**Sentiment:** {sentiment_label}")
    st.write(f"**Predicted Stock Movement:** {stock_movement}%")
