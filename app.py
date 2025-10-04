import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import base64

# --- Page config ---
st.set_page_config(page_title="Finance News Sentiment", layout="wide")

# --- Load FinBERT model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- Set background image ---
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/avif;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.2);
        z-index: -1;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg_local("stock_app.avif")

# --- Custom CSS ---
st.markdown("""
<style>
.title-bubble {
    background: #00e6ac;
    color: #000 !important;
    text-align: center;
    font-size: 2.2em;
    font-weight: bold;
    border-radius: 40px;
    padding: 25px 30px;
    margin: 40px auto 20px auto;
    max-width: 800px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.input-bubble {
    background: rgba(255,255,255,0.95);
    border-radius: 50px;
    padding: 25px 30px;
    margin: 20px auto;
    max-width: 700px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    font-family: "Helvetica Neue", sans-serif;
    color: #000;
    font-size: 1.2em;
}

.results-bubble {
    background: rgba(0,0,0,0.8);
    color: #fff !important;
    border-radius: 30px;
    padding: 25px;
    margin-top: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.results-bubble * {
    color: #fff !important;
}

textarea {
    font-size: 1.1em !important;
    border-radius: 15px !important;
    padding: 10px !important;
    border: 1px solid #ccc !important;
    color: #222 !important;
}

div.stButton > button {
    background-color: #000;
    color: #fff;
    border-radius: 12px;
    font-weight: bold;
    font-size: 1.1em;
    padding: 10px 20px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.4);
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #333;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# --- App layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

# Input bubble
st.markdown('<div class="input-bubble">💭 Paste your stock news, tweets, or finance text below (with a down arrow). We\'ll predict the impact this will have on the stock, generate a chart, and predict investor sentiment!</div>', unsafe_allow_html=True)
text = st.text_area("", "", key="finance_text")

# Prediction
if st.button("Predict 🚀"):
    if text.strip() != "":
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_idx = np.argmax(probs)
        sentiment = sentiment_labels[sentiment_idx]

        movement = 0.0
        if sentiment == "Positive":
            movement = min(10, round(float(probs[sentiment_idx]) * 10, 2))
        elif sentiment == "Negative":
            movement = -min(10, round(float(probs[sentiment_idx]) * 10, 2))

        st.markdown('<div class="results-bubble">', unsafe_allow_html=True)
        st.subheader("📊 Sentiment Probabilities")
        for label, p in zip(sentiment_labels, probs):
            st.write(f"**{label}:** {p:.4f}")

        st.subheader("🧠 Predicted Sentiment & Stock Movement")
        if sentiment == "Positive":
            st.write(f"📈 **Sentiment:** {sentiment}")
            st.write(f"📈 **Predicted Movement:** +{movement}%")
        elif sentiment == "Negative":
            st.write(f"📉 **Sentiment:** {sentiment}")
            st.write(f"📉 **Predicted Movement:** {movement}%")
        else:
            st.write(f"➖ **Sentiment:** {sentiment}")
            st.write(f"➖ **Predicted Movement:** {movement}%")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to predict!")

