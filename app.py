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

# --- Custom CSS for bubbles and input ---
st.markdown("""
<style>
.stApp {
    font-family: "Helvetica Neue", sans-serif;
    color: #fff;
}

/* Thought bubble style for input */
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
    backgro
