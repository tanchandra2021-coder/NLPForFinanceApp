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

# --- Custom CSS for modern UI ---
st.markdown("""
<style>
/* Background image: stock market chart */
.stApp {
    background-image: url("https://cdn.pixabay.com/photo/2015/09/04/23/28/stock-923706_1280.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: "Helvetica Neue", sans-serif;
    color: #fff;
}

/* Ombré black overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    height: 100%; width: 100%;
    background: linear-gradient(to bottom, rgba(0,0,0,0.8), rgba(0,0,0,0.95));
    z-index: -1;
}

/* Thought bubble style */
.thought-bubble {
    background: white;
    color: #222;
    border-radius: 30px;
    padding: 25px 30px;
    margin: 30px auto;
    max-width: 700px;
    position: relative;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

/* Bubble tail */
.thought-bubble::after {
    content: "";
    position: absolute;
    bottom: -25px;
    left: 60px;
    border-width: 25px 25px 0;
    border-style: solid;
    border-color: white transparent transparent transparent;
}

/* Title bubble */
.title-bubble {
    background: #00e6ac;
    color: #000;
    text-align: center;
    font-size: 2.2em;
    font-weight: bold;
    border-radius: 40px;
    padding: 25px 30px;
    margin: 40px auto;
    max-width: 800px;
    position: relative;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.title-bubble::after {
    content: "";
    position: absolute;
    bottom: -25px;
    right: 120px;
    border-width: 25px 25px 0;
    border-style: solid;
    border-color: #00e6ac transparent transparent transparent;
}

/* Textarea inside bubble */
textarea {
    font-size: 1.1em !important;
}

/* Results bubble */
.results-bubble {
    background: #f9f9f9;
    color: #000 !important;   /* Force black text */
    border-radius: 30px;
    padding: 25px;
    margin-top: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    position: relative;
}
.results-bubble * {
    color: #000 !important;   /* Make ALL children black */
}
.results-bubble::after {
    content: "";
    position: absolute;
    bottom: -25px;
    left: 80px;
    border-width: 25px 25px 0;
    border-style: solid;
    border-color: #f9f9f9 transparent transparent transparent;
}

/* Black Predict button */
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

# --- App Layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

# Guidance bubble
st.markdown("""
<div class="thought-bubble">
    💭 Want to see how a piece of news or a tweet will impact your stock's performance?  
    Copy it below, and we’ll predict it!
</div>
""", unsafe_allow_html=True)

# Input bubble
st.markdown('<div class="thought-bubble">', unsafe_allow_html=True)
text = st.text_area("Paste stock news, tweets, or finance text here:", "")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict 🚀"):
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

    # Results bubble (now forced black text)
    st.markdown('<div class="thought-bubble results-bubble">', unsafe_allow_html=True)
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
