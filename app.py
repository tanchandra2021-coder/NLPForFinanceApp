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

# Note: This will only work if you have a local file named 'stock_app.avif'.
try:
    set_bg_local("stock_app.avif")
except FileNotFoundError:
    st.warning("⚠️ Background image 'stock_app.avif' not found. Using default background.")

# --- Custom CSS ---
st.markdown("""
<style>
/* ⚙️ GLOBAL FUTURISTIC FONT ⚙️ (Applied to text input) */
.stTextArea textarea, .stButton > button, .results-bubble {
    /* Use a futuristic font stack for an elegant, impressive look */
    font-family: 'Orbitron', 'Roboto Mono', monospace !important; 
}


/* Main Title Bubble (Green) */
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

/* 🟢 Instruction Bubble (Sent Message Blue/Right Tail) 🟢 */
.input-bubble {
    position: relative; 
    background: #0084ff; 
    color: #fff !important; 
    font-family: 'Orbitron', sans-serif; /* Apply futuristic font to instruction */
    font-size: 1.2em;
    line-height: 1.4;
    border-radius: 25px; 
    padding: 20px 25px;
    margin: 30px auto 40px auto; /* Increased margin for better tail visibility */
    max-width: 700px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3); 
}
.input-bubble * {
    color: #fff !important;
}

/* Tail for Instruction Bubble (Bottom-Right, SMOOTH CURVE) */
.input-bubble::after {
    content: '';
    position: absolute;
    z-index: 1; 
    bottom: -15px; /* Position the tail below the bubble */
    right: 5px; /* Position next to the bubble edge */
    width: 20px;
    height: 20px;
    background: #0084ff; 
    /* The trick for the smooth curve */
    border-bottom-left-radius: 20px;
    transform: rotate(30deg); 
    box-shadow: 4px 4px 5px rgba(0,0,0,0.1); 
}


/* Results Bubble */
.results-bubble {
    background: rgba(0,0,0,0.8);
    color: #fff !important;
    border-radius: 30px;
    padding: 25px;
    margin-top: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}


/* 💬 Text Area Input Bubble (Received Message Gray/Left Tail) 💬 */

/* The entire container for the text area widget */
.stTextArea > div {
    margin: 0 auto 30px auto; 
    max-width: 700px; 
    position: relative; 
}

/* The actual text area element */
.stTextArea textarea {
    background: #e5e5ea !important; /* Light Gray for received message */
    color: #222 !important;
    font-size: 1.1em !important;
    line-height: 1.5;
    
    /* Main body should be fully rounded */
    border-radius: 25px !important; 
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    padding: 15px 20px !important; 
    transition: all 0.3s ease;
}

/* Tail for Text Input Area (Bottom-Left, SMOOTH CURVE) */
.stTextArea > div::after {
    content: '';
    position: absolute;
    z-index: 1;
    bottom: -15px; /* Position the tail below the bubble */
    left: 5px; /* Position next to the bubble edge */
    width: 20px;
    height: 20px;
    background: #e5e5ea; 
    /* The trick for the smooth curve */
    border-bottom-right-radius: 20px; 
    transform: rotate(-30deg); /* Mirror the rotation for the left side */
    box-shadow: -4px 4px 5px rgba(0,0,0,0.1); 
}

/* Fix for Streamlit's inner padding issue that affects centering and tail positioning */
div[data-testid="stVerticalBlock"] > div:first-child > div:nth-child(2) > div:first-child {
    padding-top: 0 !important;
}

/* Button Styling */
div.stButton > button {
    background-color: #00e6ac; /* Use the accent color for the button */
    color: #000;
    border-radius: 12px;
    font-weight: bold;
    font-size: 1.1em;
    padding: 10px 20px;
    box-shadow: 0 5px 15px rgba(0,230,172,0.6); /* Green shadow for button */
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #00cc99;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# --- App layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

# 📣 Instruction Bubble (Sent Message Blue with smooth tail and down arrow emoji)
st.markdown('<div class="input-bubble">💭 Paste your stock news, tweets, or finance text below **(with a down arrow ⬇️)**. We\'ll predict the impact this will have on the stock, generate a chart, and predict investor sentiment!</div>', unsafe_allow_html=True)

# Text Area Widget (Received Message Gray with smooth tail)
text = st.text_area("Enter your finance text here:", "", key="finance_text", height=150, help="e.g., 'Apple Inc. stock rises 5% after record-breaking Q4 iPhone sales.'", label_visibility="collapsed")

# Prediction
col1, col2, col3 = st.columns([1,1,1])
with col2:
    if st.button("Predict 🚀"):
        if text.strip() != "":
            # --- Model Prediction Logic ---
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

            sentiment_labels = ["Positive", "Neutral", "Negative"]
            sentiment_idx = np.argmax(probs)
            sentiment = sentiment_labels[sentiment_idx]

            # --- Simplified Stock Movement Estimation ---
            movement = 0.0
            if sentiment == "Positive":
                movement = min(10, round(float(probs[sentiment_idx]) * 10, 2))
            elif sentiment == "Negative":
                movement = -min(10, round(float(probs[sentiment_idx]) * 10, 2))
            
            # --- Results Display ---
            st.markdown('<div class="results-bubble">', unsafe_allow_html=True)
            st.subheader("📊 Sentiment Probabilities")
            for label, p in zip(sentiment_labels, probs):
                st.write(f"**{label}:** {p:.4f}")

            st.subheader("🧠 Predicted Sentiment & Stock Movement")
            if sentiment == "Positive":
                st.write(f"📈 **Sentiment:** <span style='color: #00e6ac; font-weight: bold;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"📈 **Predicted Movement:** +{movement}%")
            elif sentiment == "Negative":
                st.write(f"📉 **Sentiment:** <span style='color: #ff6347; font-weight: bold;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"📉 **Predicted Movement:** {movement}%")
            else:
                st.write(f"➖ **Sentiment:** <span style='color: #ffd700; font-weight: bold;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"➖ **Predicted Movement:** {movement}%")
            
            # --- Disclaimer ---
            st.markdown(
                """
                <hr style="border: 0.5px solid rgba(255,255,255,0.2); margin: 15px 0;">
                <p style="font-size: 0.8em; color: #aaa; font-family: sans-serif !important;">
                    *Disclaimer: This is a sentiment analysis prediction based on the FinBERT model and does not constitute financial advice. Stock movement prediction is for illustrative purposes only.*
                </p>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to predict!")
