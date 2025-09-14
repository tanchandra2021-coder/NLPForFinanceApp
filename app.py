import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

st.set_page_config(page_title="Finance News Sentiment", layout="wide")

# --- Load model ---
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- Custom CSS ---
st.markdown("""
<style>
/* Full-page background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2015/09/04/23/28/stock-923706_1280.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    position: relative;
}

/* Semi-transparent overlay for readability */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.5);
    z-index: -1;
}

/* Title bubble */
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

/* Thought bubble for input */
.thought-bubble {
    background: rgba(255,255,255,0.95);
    color: #222 !important;
    border-radius: 30px;
    padding: 25px 30px;
    margin: 20px auto;
    max-width: 700px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
    position: relative;
}
.thought-bubble::after {
    content: "";
    position: absolute;
    bottom: -25px;
    left: 60px;
    border-width: 25px 25px 0;
    border-style: solid;
    border-color: rgba(255,255,255,0.95) transparent transparent transparent;
}

/* Force textarea text to be dark */
textarea {
    font-size: 1.1em !important;
    border-radius: 15px !important;
    padding: 10px !important;
    border: 1px solid #ccc !important;
    color: #222 !important;
}

/* Results bubble */
.results-bubble {
    background: rgba(249,249,249,0.95);
    color: #000 !important;
    border-radius: 30px;
    padding: 25px;
    margin-top: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    position: relative;
}
.results-bubble h2, 
.results-bubble h3, 
.results-bubble h4, 
.results-bubble h5, 
.results-bubble h6, 
.results-bubble p, 
.results-bubble span, 
.results-bubble div {
    color: #000 !important;
}

/* Predict button */
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

# --- Layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

# Input bubble
st.markdown('<div class="thought-bubble">', unsafe_allow_html=True)
text = st.text_area("💭 Paste your stock news, tweets, or finance text here:", "")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict 🚀"):
    # Process sentiment
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

    # Results bubble
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

