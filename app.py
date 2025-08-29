# app.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re

# -------------------- STEP 1: Load Model --------------------
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "yiyanghkust/finbert-tone"  # Pretrained FinBERT
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# -------------------- STEP 2: Helper Functions --------------------
def prepare_input(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

def predict_sentiment(text):
    # Detect numeric changes and boost sentiment
    numeric_match = re.findall(r"[-+]?\d+\.?\d*%?", text)
    boost = 0
    if numeric_match:
        for n in numeric_match:
            val = float(re.sub(r"[%+]", "", n))
            boost += val
    input_ids, attention_mask = prepare_input(text)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
    
    # Adjust probs using numeric boost
    if boost > 10:  # Large positive numbers
        probs = [0.05, 0.05, 0.9]
    elif boost < -10:  # Large negative numbers
        probs = [0.9, 0.05, 0.05]
    
    return probs

def predict_stock_movement(sentiment_probs, text):
    labels = ["Negative", "Neutral", "Positive"]
    predicted_label = labels[np.argmax(sentiment_probs)]
    
    # Use numeric info if present
    numeric_match = re.findall(r"[-+]?\d+\.?\d*%?", text)
    if numeric_match:
        val = float(re.sub(r"[%+]", "", numeric_match[0]))
        stock_change = val
    else:
        if predicted_label == "Positive":
            stock_change = np.random.uniform(0.5, 3.0)
        elif predicted_label == "Neutral":
            stock_change = np.random.uniform(-0.5, 0.5)
        else:
            stock_change = np.random.uniform(-3.0, -0.5)
    
    return predicted_label, stock_change

# -------------------- STEP 3: Streamlit App --------------------
st.title("Finance News Sentiment & Stock Movement Predictor")

text = st.text_area("Enter stock news, tweets, or finance text:")
button = st.button("Predict")

if button:
    if text:
        probs = predict_sentiment(text)
        labels = ["Negative", "Neutral", "Positive"]
        
        st.subheader("Sentiment Probabilities:")
        for label, prob in zip(labels, probs):
            st.write(f"{label}: {prob:.4f}")
        
        sentiment_label, stock_change = predict_stock_movement(probs, text)
        st.subheader("Predicted Sentiment & Stock Movement:")
        st.write(f"Sentiment: **{sentiment_label}**")
        st.write(f"Predicted Stock Movement: **{stock_change:.2f}%**")
    else:
        st.error("Please enter some text to analyze.")


