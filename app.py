import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Finance News Sentiment", layout="wide")

# --- Load FinBERT model ---
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- Generate sample stock data ---
def generate_stock_data(days=50):
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(days))
    open_price = price + np.random.randn(days)
    close_price = price + np.random.randn(days)
    high_price = np.maximum(open_price, close_price) + np.random.rand(days) * 2
    low_price = np.minimum(open_price, close_price) - np.random.rand(days) * 2
    df = pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.today(), periods=days),
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price
    })
    return df

df = generate_stock_data()

# --- Plotly candlestick chart ---
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red',
    opacity=0.3
)])
fig.update_layout(
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    margin=dict(l=0,r=0,t=0,b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- Custom CSS ---
st.markdown("""
<style>
.stApp {
    font-family: "Helvetica Neue", sans-serif;
    color: #fff;
    background: #000;  /* fallback */
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
    margin: 40px auto 20px auto;
    max-width: 800px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* Thought bubble for input */
.thought-bubble {
    background: rgba(255,255,255,0.95);
    color: #222;
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

/* Textarea inside bubble */
textarea {
    font-size: 1.1em !important;
    border-radius: 15px !important;
    padding: 10px !important;
    border: 1px solid #ccc !important;
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
.results-bubble * {
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

# --- App layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

# Input bubble
st.markdown('<div class="thought-bubble">', unsafe_allow_html=True)
text = st.text_area("💭 Paste your stock news, tweets, or finance text here:", "")
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

    if sentiment == "Positive":
        movement = min(10, round(float(probs[sentiment_idx]) * 10, 2))
    elif sentiment == "Negative":
        movement = -min(10, round(float(probs[sentiment_idx]) * 10, 2))
    else:
        movement = 0.0

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

