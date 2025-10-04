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
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif !important;
}

/* Main Title Bubble */
.title-bubble {
    background: linear-gradient(135deg, #00f2a9 0%, #00d4a0 100%);
    color: #000 !important;
    text-align: center;
    font-size: 2.5em;
    font-weight: 700;
    border-radius: 25px;
    padding: 30px 40px;
    margin: 40px auto 30px auto;
    max-width: 850px;
    box-shadow: 0 15px 50px rgba(0,242,169,0.4), 0 5px 15px rgba(0,0,0,0.3);
    letter-spacing: -0.5px;
}

/* iMessage Blue Bubble (Sent Message) */
.input-bubble {
    position: relative;
    background: #0084ff;
    color: #fff !important;
    font-size: 1.15em;
    line-height: 1.5;
    border-radius: 20px;
    padding: 16px 20px;
    margin: 25px auto 45px auto;
    max-width: 680px;
    box-shadow: 0 4px 12px rgba(0,132,255,0.3);
    margin-left: auto;
    margin-right: 50px;
}

.input-bubble * {
    color: #fff !important;
}

/* Proper iMessage tail for blue bubble (bottom right) */
.input-bubble::after {
    content: '';
    position: absolute;
    bottom: 0;
    right: -8px;
    width: 20px;
    height: 25px;
    background: #0084ff;
    border-bottom-left-radius: 16px 14px;
    transform: translateY(0);
}

.input-bubble::before {
    content: '';
    position: absolute;
    bottom: 0;
    right: -10px;
    width: 10px;
    height: 20px;
    background: transparent;
    border-bottom-left-radius: 10px;
}

/* Text Area Container */
.stTextArea > div {
    margin: 0 50px 40px 0;
    max-width: 680px;
    position: relative;
}

/* iMessage Gray Bubble (Received Message) */
.stTextArea textarea {
    background: #e5e5ea !important;
    color: #000 !important;
    font-size: 1.1em !important;
    line-height: 1.5 !important;
    border-radius: 20px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    padding: 16px 20px !important;
    transition: all 0.2s ease;
}

.stTextArea textarea:focus {
    box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
    outline: none !important;
}

/* Proper iMessage tail for gray bubble (bottom left) */
.stTextArea > div::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: -8px;
    width: 20px;
    height: 25px;
    background: #e5e5ea;
    border-bottom-right-radius: 16px 14px;
    z-index: 0;
}

.stTextArea > div::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: -10px;
    width: 10px;
    height: 20px;
    background: transparent;
    border-bottom-right-radius: 10px;
    z-index: 1;
}

/* Results Bubble */
.results-bubble {
    background: rgba(20,20,20,0.95);
    backdrop-filter: blur(20px);
    color: #fff !important;
    border-radius: 25px;
    padding: 30px 35px;
    margin: 30px auto;
    max-width: 700px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 80px rgba(0,242,169,0.1);
    border: 1px solid rgba(255,255,255,0.1);
}

.results-bubble h3 {
    color: #00f2a9 !important;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #00f2a9 0%, #00d4a0 100%);
    color: #000;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.15em;
    padding: 14px 40px;
    border: none;
    box-shadow: 0 6px 20px rgba(0,242,169,0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    letter-spacing: 0.5px;
}

div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,242,169,0.6);
}

div.stButton > button:active {
    transform: translateY(-1px);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Smooth animations */
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

st.markdown('<div class="input-bubble">💭 Paste your stock news, tweets, or finance text below ⬇️. We\'ll predict the impact this will have on the stock, generate a chart, and predict investor sentiment!</div>', unsafe_allow_html=True)

text = st.text_area("Enter your finance text here:", "", key="finance_text", height=150, help="e.g., 'Apple Inc. stock rises 5% after record-breaking Q4 iPhone sales.'", label_visibility="collapsed")

col1, col2, col3 = st.columns([1,1,1])
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
                    <em>Disclaimer: This is a sentiment analysis prediction based on the FinBERT model and does not constitute financial advice. Stock movement prediction is for illustrative purposes only.</em>
                </p>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to predict!")
