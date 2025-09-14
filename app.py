import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load pre-trained model ---
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- Generate sample stock data for candlestick ---
def generate_sample_stock():
    # Simulate 30 days of stock prices
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(30))
    open_price = price + np.random.randn(30)
    close_price = price + np.random.randn(30)
    high_price = np.maximum(open_price, close_price) + np.random.rand(30)*2
    low_price = np.minimum(open_price, close_price) - np.random.rand(30)*2
    df = pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.today(), periods=30),
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price
    })
    return df

df = generate_sample_stock()

# --- Plotly Candlestick Chart ---
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
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)

# --- Custom CSS for bubbles ---
st.markdown("""
<style>
.stApp {
    font-family: "Helvetica Neue", sans-serif;
    color: #fff;
}

/* Thought bubble style */
.thought-bubble {
    background: rgba(255,255,255,0.95);
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
    border-color: rgba(255,255,255,0.95) transparent transparent transparent;
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

st.markdown("""
<div class="thought-bubble">
    💭 Paste your stock news, tweets, or finance text below and predict its impact!
</div>
""", unsafe_allow_html=True)

text = st.text_area("Paste stock news or tweets here:", "")

if st.button("Predict 🚀"):
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
