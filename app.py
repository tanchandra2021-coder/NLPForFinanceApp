import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Finance News Sentiment & Stock Movement Predictor")
text = st.text_area("Enter stock news, tweets, or finance text:")

if st.button("Predict"):
    # Preprocess
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

    sentiment_labels = ["Positive", "Neutral", "Negative"]
    sentiment_idx = np.argmax(probs)
    sentiment = sentiment_labels[sentiment_idx]

    # Cap predicted stock movement to +/-10%
    if sentiment == "Positive":
        movement = min(10, round(float(probs[sentiment_idx]) * 10, 2))
    elif sentiment == "Negative":
        movement = -min(10, round(float(probs[sentiment_idx]) * 10, 2))
    else:
        movement = 0.0

    # Display
    st.subheader("Sentiment Probabilities:")
    for label, p in zip(sentiment_labels, probs):
        st.write(f"{label}: {p:.4f}")

    st.subheader("Predicted Sentiment & Stock Movement:")
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Predicted Stock Movement:** {movement}%")

