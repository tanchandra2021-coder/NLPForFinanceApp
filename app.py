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
        background: rgba(0,0,0,0.3);
        z-index: -1;
    }}
    </style>
    """, unsafe_allow_html=True)

try:
    set_bg_local("stock_app.avif")
except FileNotFoundError:
    pass

# --- Custom CSS ---
st.markdown("""
<style>
/* Global Font */
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
}

/* Center everything */
.block-container {
    max-width: 800px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Main Title */
.title-bubble {
    background: linear-gradient(135deg, #00f2a9 0%, #00d4a0 100%);
    color: #000 !important;
    text-align: center;
    font-size: 2.2em;
    font-weight: 700;
    border-radius: 25px;
    padding: 30px 40px;
    margin: 40px auto 40px auto;
    box-shadow: 0 15px 50px rgba(0,242,169,0.4);
}

/* Blue iMessage Bubble Container */
.blue-bubble-container {
    display: flex;
    justify-content: flex-end;
    margin: 30px 0 50px 80px;
}

/* Blue Bubble (Sent Message - Right aligned) */
.input-bubble {
    position: relative;
    background: #0084ff;
    color: #fff !important;
    font-size: 1.1em;
    line-height: 1.5;
    border-radius: 18px;
    padding: 14px 18px;
    max-width: 85%;
    box-shadow: 0 2px 10px rgba(0,132,255,0.3);
}

/* Blue bubble tail - proper iMessage style */
.input-bubble::before {
    content: '';
    position: absolute;
    right: -7px;
    bottom: 0;
    width: 20px;
    height: 20px;
    background: #0084ff;
    border-radius: 0 0 0 18px;
}

.input-bubble::after {
    content: '';
    position: absolute;
    right: -10px;
    bottom: 0;
    width: 10px;
    height: 20px;
    background: transparent;
    border-radius: 0 0 0 10px;
}

/* Gray iMessage Bubble Container */
.gray-bubble-container {
    display: flex;
    justify-content: flex-start;
    margin: 0 80px 40px 0;
}

/* Text Area - Gray Bubble (Received Message - Left aligned) */
.stTextArea {
    width: 100%;
}

.stTextArea > div {
    display: flex;
    justify-content: flex-start;
}

.stTextArea > div > div {
    position: relative;
    max-width: 85%;
}

.stTextArea textarea {
    background: #e5e5ea !important;
    color: #000 !important;
    font-size: 1.1em !important;
    line-height: 1.5 !important;
    border-radius: 18px !important;
    border: none !important;
    box-shadow: 0 1px 5px rgba(0,0,0,0.1) !important;
    padding: 14px 18px !important;
}

/* Gray bubble tail - proper iMessage style */
.stTextArea > div > div::before {
    content: '';
    position: absolute;
    left: -7px;
    bottom: 6px;
    width: 20px;
    height: 20px;
    background: #e5e5ea;
    border-radius: 0 0 18px 0;
    z-index: -1;
}

.stTextArea > div > div::after {
    content: '';
    position: absolute;
    left: -10px;
    bottom: 6px;
    width: 10px;
    height: 20px;
    background: transparent;
    border-radius: 0 0 10px 0;
    z-index: 0;
}

/* Results Bubble */
.results-bubble {
    background: rgba(20,20,20,0.95);
    backdrop-filter: blur(20px);
    color: #fff !important;
    border-radius: 25px;
    padding: 30px;
    margin: 30px auto;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.1);
}

.results-bubble h3 {
    color: #00f2a9 !important;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Button */
div.stButton {
    display: flex;
    justify-content: center;
}

div.stButton > button {
    background: linear-gradient(135deg, #00f2a9 0%, #00d4a0 100%);
    color: #000;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.15em;
    padding: 14px 50px;
    border: none;
    box-shadow: 0 6px 20px rgba(0,242,169,0.4);
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0,242,169,0.6);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.results-bubble {
    animation: fadeIn 0.5s ease-out;
}
</style>
""", unsafe_allow_html=True)

# --- App layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

st.markdown('''
<div class="blue-bubble-container">
    <div class="input-bubble">
        💭 Paste your stock news, tweets, or finance text below ⬇️. We'll predict the impact this will have on the stock, generate a chart, and predict investor sentiment!
    </div>
</div>
''', unsafe_allow_html=True)

text = st.text_area("Enter your finance text here:", "", key="finance_text", height=150, label_visibility="collapsed")

col1, col2, col3 = st.columns([1,2,1])
with col2:
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
                st.write(f"📈 **Sentiment:** <span style='color: #00f2a9; font-weight: bold;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"📈 **Predicted Movement:** +{movement}%")
            elif sentiment == "Negative":
                st.write(f"📉 **Sentiment:** <span style='color: #ff6b6b; font-weight: bold;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"📉 **Predicted Movement:** {movement}%")
            else:
                st.write(f"➖ **Sentiment:** <span style='color: #ffd93d; font-weight: bold;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"➖ **Predicted Movement:** {movement}%")
            
            st.markdown(
                """
                <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 20px 0;">
                <p style="font-size: 0.85em; color: #888; text-align: center;">
                    <em>Disclaimer: This is a sentiment analysis prediction based on the FinBERT model and does not constitute financial advice.</em>
                </p>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to predict!")
