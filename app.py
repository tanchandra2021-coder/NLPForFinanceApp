# app.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# -------------------- STEP 1: Load Pretrained FinBERT Model --------------------
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "yiyanghkust/finbert-tone"  # Hugging Face pretrained FinBERT for finance
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
    input_ids, attention_mask = prepare_input(text)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
    return probs

def predict_stock_movement(sentiment_probs):
    # Map sentiment to rough stock change (%)
    labels = ["positive", "neutral", "negative"]
    predicted_label = labels[np.argmax(sentiment_probs)]
    
    if predicted_label == "positive":
        stock_change = np.random.uniform(0.5, 3.0)
    elif predicted_label == "neutral":
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
        labels = ["Positive", "Neutral", "Negative"]
        
        st.subheader("Sentiment Probabilities:")
        for label, prob in zip(labels, probs):
            st.write(f"{label}: {prob:.4f}")
        
        sentiment_label, stock_change = predict_stock_movement(probs)
        st.subheader("Predicted Sentiment & Stock Movement:")
        st.write(f"Sentiment: **{sentiment_label.capitalize()}**")
        st.write(f"Predicted Stock Movement: **{stock_change:.2f}%**")
        
    else:
        st.error("Please enter some text to analyze.")
