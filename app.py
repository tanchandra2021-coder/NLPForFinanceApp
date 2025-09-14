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
/* Background image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1561414927-6d86591d0c4f");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: "Helvetica Neue", sans-serif;
    color: #fff;
}

/* Overlay to make text readable */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    height: 100%; width: 100%;
    background: rgba(0,0,0,0.6);
    z-index: -1;
}

/* Title */
h1 {
    text-align: center;
    font-size: 3em;
    margin-bottom: 10px;
    color: #00e6ac;
    text-shadow: 2px 2px 6px #000;
}

/* Thought bubble for input */
.thought-bubble {
    background: white;
    color: #333;
    border-radius: 30px;
    padding: 20px;
    position: relative;
    margin: 30px auto;
    max-width: 600px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

/* Bubble tail */
.thought-bubble::after {
    content: "";
    position: absolute;
    bottom: -20px;
    left: 50px;
    border-width: 20px 20px 0;
    border-style: solid;
    border-color: white transparent transparent transparent;
}

/* Prompt bubble with arrow */
.prompt-bubble {
    background: #00e6ac;
    color: #000;
    border-radius: 20px;
    padding: 15px 25px;
    position: relative;
    display: inline-block;
    margin: 15px 40px;
    font-weight: bold;
    box-shadow: 0 6px 15px rgba(0,0,0,0.3);
}

.prompt-bubble::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -20px;
    transform: translateY(-50%);
    border-width: 10px 0 10px 20px;
    border-style: solid;
    border-color: transparent transparent transparent #00e6ac;
}

/* Results card */
.results-card {
    background: rgba(255,255,255,0.95);
    color: #111;
    border-radius: 20px;
    padding: 25px;
    margin-top: 25px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# --- App Layout ---
st.title("Finance News Sentiment & Stock Movement Predictor")

# Guidance bubble
st.markdown("""
<div class="prompt-bubble">
    💬 What tweet should I analyze? Let's see how your favorite stock will be impacted!
</div>
""", unsafe_allow_html=True)

# Thought bubble with text input
st.markdown('<div class="thought-bubble">', unsafe_allow_html=True)
text = st.text_area("Enter stock news, tweets, or finance text:", "")
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

    # Display results in a modern card
    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    st.subheader("📊 Sentiment Probabilities")
    for label, p in zip(sentiment_labels, probs):
        st.write(f"**{label}:** {p:.4f}")

    st.subheader("🧠 Predicted Sentiment & Stock Movement")
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Predicted Stock Movement:** {movement}%")
    st.markdown('</div>', unsafe_allow_html=True)


