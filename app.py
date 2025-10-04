import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import base64
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# --- Page config ---
st.set_page_config(page_title="Finance News Sentiment", layout="wide")

# --- Load FinBERT model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- Get real-time stock data ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data():
    tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX']
    stock_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev_close = info.get('previousClose', current_price)
            
            if prev_close and prev_close > 0:
                change_percent = ((current_price - prev_close) / prev_close) * 100
            else:
                change_percent = 0
            
            stock_data.append({
                'ticker': ticker,
                'change': change_percent
            })
        except:
            # Fallback if API fails
            stock_data.append({
                'ticker': ticker,
                'change': 0
            })
    
    return stock_data

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
        background: rgba(0,0,0,0.4);
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Title with glow effect */
.title-bubble {
    background: linear-gradient(135deg, #00f2a9 0%, #00d98e 50%, #00c47a 100%);
    color: #000 !important;
    text-align: center;
    font-size: 2.5em;
    font-weight: 800;
    border-radius: 30px;
    padding: 35px 45px;
    margin: 30px auto 40px auto;
    box-shadow: 0 20px 60px rgba(0,242,169,0.5), 0 0 100px rgba(0,242,169,0.3);
    animation: titleGlow 3s ease-in-out infinite;
    letter-spacing: -0.5px;
}

@keyframes titleGlow {
    0%, 100% { box-shadow: 0 20px 60px rgba(0,242,169,0.5), 0 0 100px rgba(0,242,169,0.3); }
    50% { box-shadow: 0 20px 80px rgba(0,242,169,0.7), 0 0 120px rgba(0,242,169,0.5); }
}

/* Instruction box - cute and curved */
.instruction-box {
    background: linear-gradient(135deg, rgba(0,132,255,0.95) 0%, rgba(0,100,255,0.95) 100%);
    backdrop-filter: blur(20px);
    color: #fff !important;
    font-size: 1.2em;
    line-height: 1.6;
    border-radius: 25px;
    padding: 25px 30px;
    margin: 30px auto;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0,132,255,0.4);
    animation: slideDown 0.8s ease-out;
    border: 2px solid rgba(255,255,255,0.2);
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar cards */
.sidebar-card {
    background: linear-gradient(135deg, rgba(0,242,169,0.15) 0%, rgba(0,200,150,0.15) 100%);
    backdrop-filter: blur(15px);
    border: 2px solid rgba(0,242,169,0.3);
    border-radius: 20px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    animation: fadeInLeft 0.8s ease-out;
}

@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

.sidebar-card h4 {
    color: #00f2a9 !important;
    font-size: 1.1em;
    font-weight: 700;
    margin-bottom: 10px;
}

.sidebar-card p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.95em;
    line-height: 1.5;
}

/* Feature icons */
.feature-icon {
    font-size: 2em;
    display: inline-block;
    animation: bounce 2s ease-in-out infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

/* Stat counter */
.stat-counter {
    background: linear-gradient(135deg, rgba(0,132,255,0.2) 0%, rgba(0,100,255,0.2) 100%);
    border: 2px solid rgba(0,132,255,0.4);
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
    animation: fadeInRight 0.8s ease-out;
}

@keyframes fadeInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}

.stat-counter h3 {
    color: #0084ff !important;
    font-size: 2em;
    font-weight: 800;
    margin: 5px 0;
}

.stat-counter p {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.9em;
}

/* Text Area - clean and modern */
.stTextArea label {
    font-size: 1.1em !important;
    font-weight: 600 !important;
    color: #00f2a9 !important;
    margin-bottom: 10px !important;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.95) !important;
    backdrop-filter: blur(10px);
    color: #1a1a1a !important;
    font-size: 1.05em !important;
    line-height: 1.6 !important;
    border-radius: 20px !important;
    border: 2px solid rgba(0,242,169,0.3) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border: 2px solid #00f2a9 !important;
    box-shadow: 0 15px 40px rgba(0,242,169,0.3) !important;
    transform: translateY(-2px);
}

/* Button with pulse effect */
div.stButton > button {
    background: linear-gradient(135deg, #00f2a9 0%, #00d98e 100%);
    color: #000;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1.2em;
    padding: 18px 60px;
    border: none;
    box-shadow: 0 10px 30px rgba(0,242,169,0.5);
    transition: all 0.3s ease;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.03); }
}

div.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 15px 40px rgba(0,242,169,0.7);
}

/* Results card */
.results-card {
    background: linear-gradient(135deg, rgba(15,15,15,0.98) 0%, rgba(30,30,30,0.98) 100%);
    backdrop-filter: blur(30px);
    color: #fff !important;
    border-radius: 30px;
    padding: 35px;
    margin: 40px auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 100px rgba(0,242,169,0.1);
    border: 2px solid rgba(0,242,169,0.2);
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

.results-card h3 {
    color: #00f2a9 !important;
    font-weight: 700;
    font-size: 1.5em;
    margin-bottom: 20px;
    text-align: center;
}

.results-card h2 {
    color: #00f2a9 !important;
    font-weight: 700;
    font-size: 1.8em;
    margin-top: 30px;
    margin-bottom: 20px;
    text-align: center;
}

/* Sentiment badges */
.sentiment-badge {
    display: inline-block;
    padding: 10px 25px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1em;
    margin: 10px 5px;
    animation: bounceIn 0.6s ease-out;
}

@keyframes bounceIn {
    0% { transform: scale(0); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.positive-badge {
    background: linear-gradient(135deg, #00f2a9 0%, #00d98e 100%);
    color: #000;
    box-shadow: 0 5px 20px rgba(0,242,169,0.4);
}

.negative-badge {
    background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%);
    color: #fff;
    box-shadow: 0 5px 20px rgba(255,107,107,0.4);
}

.neutral-badge {
    background: linear-gradient(135deg, #ffd93d 0%, #ffb900 100%);
    color: #000;
    box-shadow: 0 5px 20px rgba(255,217,61,0.4);
}

/* Stats display */
.stat-box {
    background: rgba(0,242,169,0.1);
    border: 2px solid rgba(0,242,169,0.3);
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
    animation: scaleIn 0.5s ease-out;
}

@keyframes scaleIn {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}

.stat-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #00f2a9;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Demo tag */
.demo-tag {
    display: inline-block;
    background: rgba(255,215,0,0.2);
    color: #ffd700;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 600;
    margin-bottom: 15px;
    border: 2px solid rgba(255,215,0,0.4);
    animation: shimmer 2s ease-in-out infinite;
}

@keyframes shimmer {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

/* Floating finance graphics */
.finance-graphic {
    position: fixed;
    font-size: 3em;
    opacity: 0.15;
    pointer-events: none;
    z-index: 0;
}

.graphic-1 {
    top: 10%;
    left: 5%;
    animation: float1 8s ease-in-out infinite;
}

.graphic-2 {
    top: 20%;
    right: 5%;
    animation: float2 10s ease-in-out infinite;
}

.graphic-3 {
    bottom: 15%;
    left: 8%;
    animation: float3 12s ease-in-out infinite;
}

.graphic-4 {
    bottom: 25%;
    right: 8%;
    animation: float4 9s ease-in-out infinite;
}

.graphic-5 {
    top: 50%;
    left: 3%;
    animation: rotate1 15s linear infinite;
}

.graphic-6 {
    top: 60%;
    right: 3%;
    animation: rotate2 20s linear infinite;
}

@keyframes float1 {
    0%, 100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
    25% { transform: translateY(-20px) translateX(10px) rotate(5deg); }
    50% { transform: translateY(-10px) translateX(20px) rotate(-5deg); }
    75% { transform: translateY(10px) translateX(10px) rotate(3deg); }
}

@keyframes float2 {
    0%, 100% { transform: translateY(0px) translateX(0px) scale(1); }
    33% { transform: translateY(15px) translateX(-15px) scale(1.1); }
    66% { transform: translateY(-15px) translateX(-10px) scale(0.9); }
}

@keyframes float3 {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-25px) rotate(10deg); }
}

@keyframes float4 {
    0%, 100% { transform: translateX(0px) translateY(0px); }
    33% { transform: translateX(-20px) translateY(10px); }
    66% { transform: translateX(10px) translateY(-15px); }
}

@keyframes rotate1 {
    from { transform: rotate(0deg); opacity: 0.1; }
    50% { opacity: 0.2; }
    to { transform: rotate(360deg); opacity: 0.1; }
}

@keyframes rotate2 {
    from { transform: rotate(360deg); opacity: 0.15; }
    50% { opacity: 0.25; }
    to { transform: rotate(0deg); opacity: 0.15; }
}

/* Animated stock ticker */
.stock-ticker {
    position: fixed;
    bottom: 20px;
    left: 0;
    right: 0;
    background: linear-gradient(90deg, rgba(0,242,169,0.1) 0%, rgba(0,132,255,0.1) 100%);
    backdrop-filter: blur(10px);
    border-top: 2px solid rgba(0,242,169,0.3);
    padding: 12px 0;
    overflow: hidden;
    z-index: 1000;
}

.ticker-content {
    display: flex;
    animation: scroll 30s linear infinite;
    white-space: nowrap;
}

@keyframes scroll {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

.ticker-item {
    display: inline-flex;
    align-items: center;
    margin: 0 30px;
    font-size: 0.95em;
    font-weight: 600;
    color: rgba(255,255,255,0.8);
}

.ticker-up {
    color: #00f2a9 !important;
}

.ticker-down {
    color: #ff6b6b !important;
}
</style>
""", unsafe_allow_html=True)

# --- Main content ---
# Get real-time stock data
stock_data = get_stock_data()

# Floating finance graphics in corners
st.markdown('''
<div class="finance-graphic graphic-1">📈</div>
<div class="finance-graphic graphic-2">💰</div>
<div class="finance-graphic graphic-3">📊</div>
<div class="finance-graphic graphic-4">💹</div>
<div class="finance-graphic graphic-5">🎯</div>
<div class="finance-graphic graphic-6">💎</div>
''', unsafe_allow_html=True)

# Build ticker HTML with real data
ticker_items = ""
for stock in stock_data:
    arrow = "↗" if stock['change'] >= 0 else "↘"
    css_class = "ticker-up" if stock['change'] >= 0 else "ticker-down"
    sign = "+" if stock['change'] >= 0 else ""
    ticker_items += f'<span class="ticker-item {css_class}">{stock["ticker"]} {arrow} {sign}{stock["change"]:.2f}%</span>'

# Duplicate for seamless loop
ticker_items_doubled = ticker_items + ticker_items

# Animated stock ticker at bottom with real data
st.markdown(f'''
<div class="stock-ticker">
    <div class="ticker-content">
        {ticker_items_doubled}
    </div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

st.markdown('<div class="instruction-box">💭 Paste your stock news, tweets, or finance text below. We\'ll analyze sentiment, predict stock movement, and visualize the results with beautiful charts!</div>', unsafe_allow_html=True)

# Analysis mode selector
analysis_mode = st.radio(
    "Select Analysis Mode:",
    ["General Market Sentiment", "Tariff Impact Analysis"],
    horizontal=True,
    help="Choose between general financial sentiment or specialized tariff impact analysis"
)

# Demo mode toggle
demo_mode = st.checkbox("🎬 Show Demo Example", value=False)

if demo_mode:
    if analysis_mode == "Tariff Impact Analysis":
        st.markdown('<div class="demo-tag">🌟 DEMO MODE ACTIVE - Example: "New tariffs on steel imports"</div>', unsafe_allow_html=True)
        text = "President announces new 25% tariffs on steel and aluminum imports, citing national security concerns. Industry groups warn of supply chain disruptions and increased costs for manufacturers. European Union threatens retaliatory measures."
    else:
        st.markdown('<div class="demo-tag">🌟 DEMO MODE ACTIVE - Example: "Apple iPhone sales increased 50%"</div>', unsafe_allow_html=True)
        text = "Apple Inc. announces record-breaking Q4 earnings with revenue exceeding $120 billion, driven by strong iPhone 15 sales and robust services growth. The company exceeded analyst expectations by 15%, with CEO Tim Cook highlighting unprecedented demand in emerging markets. Stock surged 8% in after-hours trading as investors celebrated the stellar performance."
    st.text_area("📝 Demo Input - Try your own text!", text, key="finance_text", height=150)
else:
    if analysis_mode == "Tariff Impact Analysis":
        text = st.text_area("📝 Enter tariff-related news or social media post:", "", key="finance_text", height=150, 
                            placeholder="e.g., 'China announces retaliatory tariffs on U.S. agricultural products...'")
    else:
        text = st.text_area("📝 Enter your finance text here:", "", key="finance_text", height=150, 
                            placeholder="e.g., 'Tesla stock soars 20% after announcing breakthrough in battery technology...'")

col1, col2, col3 = st.columns([1,1,1])
with col2:
    predict_button = st.button("🚀 Predict Sentiment", use_container_width=True)
    
if predict_button or demo_mode:
    if text.strip() != "":
        with st.spinner('🔮 Analyzing sentiment...'):
            # Model prediction
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

            sentiment_labels = ["Positive", "Neutral", "Negative"]
            sentiment_idx = np.argmax(probs)
            sentiment = sentiment_labels[sentiment_idx]

            # More nuanced stock movement calculation
            confidence = float(probs[sentiment_idx])
            pos_prob = float(probs[0])
            neg_prob = float(probs[2])
            
            # Calculate sentiment strength and volatility
            sentiment_spread = abs(pos_prob - neg_prob)
            sentiment_certainty = max(probs) - sorted(probs)[-2]  # Gap between top 2
            
            movement = 0.0
            if sentiment == "Positive":
                # Scale movement based on confidence and sentiment strength
                base_movement = confidence * sentiment_spread * 8
                # Add boost if sentiment is very certain
                certainty_boost = sentiment_certainty * 2
                movement = min(7.5, round(base_movement + certainty_boost, 2))
            elif sentiment == "Negative":
                # Negative movements tend to be sharper
                base_movement = confidence * sentiment_spread * 9
                certainty_boost = sentiment_certainty * 2.5
                movement = -min(8.5, round(base_movement + certainty_boost, 2))
            else:
                # Neutral with slight bias based on underlying sentiment
                movement = round((pos_prob - neg_prob) * 2, 2)
            
            # Generate nuanced analysis text
            def get_sentiment_analysis(sentiment, confidence, movement, probs):
                pos_prob = float(probs[0])
                neg_prob = float(probs[2])
                neu_prob = float(probs[1])
                
                if sentiment == "Positive":
                    if confidence > 0.85:
                        strength = "strongly positive"
                        impact = "significant bullish momentum"
                    elif confidence > 0.70:
                        strength = "moderately positive"
                        impact = "cautious optimism"
                    else:
                        strength = "slightly positive"
                        impact = "mild upward pressure"
                    
                    if neu_prob > 0.25:
                        qualifier = f" though {neu_prob:.1%} neutral sentiment suggests some uncertainty"
                    else:
                        qualifier = " with strong conviction"
                elif sentiment == "Negative":
                    if confidence > 0.85:
                        strength = "strongly negative"
                        impact = "substantial bearish pressure"
                    elif confidence > 0.70:
                        strength = "moderately negative"
                        impact = "notable investor concern"
                    else:
                        strength = "slightly negative"
                        impact = "mild downward pressure"
                    
                    if neu_prob > 0.25:
                        qualifier = f" though {neu_prob:.1%} neutral sentiment indicates mixed signals"
                    else:
                        qualifier = " with high conviction"
                else:
                    if abs(pos_prob - neg_prob) < 0.15:
                        strength = "balanced neutral"
                        impact = "minimal market impact expected"
                        qualifier = " with equal positive and negative signals"
                    elif pos_prob > neg_prob:
                        strength = "neutral with positive lean"
                        impact = "slight upward bias"
                        qualifier = f" as positive sentiment edges out at {pos_prob:.1%}"
                    else:
                        strength = "neutral with negative lean"
                        impact = "slight downward bias"
                        qualifier = f" as negative sentiment dominates at {neg_prob:.1%}"
                
                return strength, impact, qualifier
            
            # Results display
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            
            # Get nuanced analysis
            strength, impact, qualifier = get_sentiment_analysis(sentiment, confidence, movement, probs)
            
            # Add tariff-specific analysis if in tariff mode
            if analysis_mode == "Tariff Impact Analysis":
                st.markdown('<h2>🌐 Tariff Impact on Investor Sentiment</h2>', unsafe_allow_html=True)
                
                # Calculate impact score
                if sentiment == "Positive":
                    impact_score = round(confidence * 10, 2)
                    st.markdown(f'<div style="text-align: center; margin: 20px 0;"><span style="background: rgba(0,242,169,0.2); padding: 15px 30px; border-radius: 15px; border: 2px solid #00f2a9; display: inline-block;"><span style="font-size: 1.3em; font-weight: 700; color: #00f2a9;">↑ Optimism among investors</span><br><span style="font-size: 0.9em; color: #aaa; margin-top: 5px; display: block;">+{impact_score} sentiment points</span></span></div>', unsafe_allow_html=True)
                    st.markdown(f'<p style="text-align: center; color: #aaa; font-size: 0.95em;">The tariff news suggests potential benefits for domestic producers or trade resolution, creating positive market sentiment{qualifier}.</p>', unsafe_allow_html=True)
                elif sentiment == "Negative":
                    impact_score = round(confidence * 10, 2)
                    st.markdown(f'<div style="text-align: center; margin: 20px 0;"><span style="background: rgba(255,107,107,0.2); padding: 15px 30px; border-radius: 15px; border: 2px solid #ff6b6b; display: inline-block;"><span style="font-size: 1.3em; font-weight: 700; color: #ff6b6b;">↓ Concern among investors</span><br><span style="font-size: 0.9em; color: #aaa; margin-top: 5px; display: block;">-{impact_score} sentiment points</span></span></div>', unsafe_allow_html=True)
                    st.markdown(f'<p style="text-align: center; color: #aaa; font-size: 0.95em;">The tariff developments indicate potential supply chain disruption, increased costs, or trade tensions, triggering investor caution{qualifier}.</p>', unsafe_allow_html=True)
                else:
                    impact_score = round((1 - confidence) * 5, 2)
                    st.markdown(f'<div style="text-align: center; margin: 20px 0;"><span style="background: rgba(255,217,61,0.2); padding: 15px 30px; border-radius: 15px; border: 2px solid #ffd93d; display: inline-block;"><span style="font-size: 1.3em; font-weight: 700; color: #ffd93d;">~ Stable market reaction</span><br><span style="font-size: 0.9em; color: #aaa; margin-top: 5px; display: block;">±{impact_score} sentiment points</span></span></div>', unsafe_allow_html=True)
                    st.markdown(f'<p style="text-align: center; color: #aaa; font-size: 0.95em;">The tariff news shows mixed signals with offsetting positive and negative factors{qualifier}.</p>', unsafe_allow_html=True)
            else:
                # Standard sentiment display
                st.markdown('<h2>🎯 Predicted Sentiment</h2>', unsafe_allow_html=True)
            if sentiment == "Positive":
                st.markdown(f'<div style="text-align: center;"><span class="sentiment-badge positive-badge">📈 {sentiment}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center; color: #aaa; margin-top: 10px; font-size: 0.95em;">Market signals are <strong style="color: #00f2a9;">{strength}</strong> indicating {impact}{qualifier}.</p>', unsafe_allow_html=True)
            elif sentiment == "Negative":
                st.markdown(f'<div style="text-align: center;"><span class="sentiment-badge negative-badge">📉 {sentiment}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center; color: #aaa; margin-top: 10px; font-size: 0.95em;">Market signals are <strong style="color: #ff6b6b;">{strength}</strong> indicating {impact}{qualifier}.</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: center;"><span class="sentiment-badge neutral-badge">➖ {sentiment}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center; color: #aaa; margin-top: 10px; font-size: 0.95em;">Market signals are <strong style="color: #ffd93d;">{strength}</strong> indicating {impact}{qualifier}.</p>', unsafe_allow_html=True)
            
            # Movement prediction with context
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1em; color: #aaa; margin-bottom: 5px;">Estimated Short-Term Price Movement</div>', unsafe_allow_html=True)
            if movement > 0:
                st.markdown(f'<div class="stat-value">+{movement}%</div>', unsafe_allow_html=True)
                if movement > 5:
                    st.markdown('<div style="font-size: 0.85em; color: #00f2a9; margin-top: 5px;">Strong positive catalyst detected</div>', unsafe_allow_html=True)
                elif movement > 2:
                    st.markdown('<div style="font-size: 0.85em; color: #00f2a9; margin-top: 5px;">Moderate upward momentum</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size: 0.85em; color: #00f2a9; margin-top: 5px;">Slight positive drift expected</div>', unsafe_allow_html=True)
            elif movement < 0:
                st.markdown(f'<div class="stat-value">{movement}%</div>', unsafe_allow_html=True)
                if movement < -5:
                    st.markdown('<div style="font-size: 0.85em; color: #ff6b6b; margin-top: 5px;">Strong negative catalyst detected</div>', unsafe_allow_html=True)
                elif movement < -2:
                    st.markdown('<div style="font-size: 0.85em; color: #ff6b6b; margin-top: 5px;">Moderate downward pressure</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size: 0.85em; color: #ff6b6b; margin-top: 5px;">Slight negative drift expected</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stat-value">{movement}%</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size: 0.85em; color: #ffd93d; margin-top: 5px;">Minimal price action anticipated</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability bar chart
            st.markdown('<h3>📊 Sentiment Probabilities</h3>', unsafe_allow_html=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sentiment_labels,
                    y=probs,
                    marker=dict(
                        color=['#00f2a9', '#ffd93d', '#ff6b6b'],
                        line=dict(color='rgba(255,255,255,0.3)', width=2)
                    ),
                    text=[f'{p:.2%}' for p in probs],
                    textposition='outside',
                    textfont=dict(size=14, color='white', family='Inter')
                )
            ])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title='Sentiment',
                yaxis_title='Probability'
            )
            
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', title_font=dict(size=14, color='#00f2a9'))
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', tickformat='.0%', title_font=dict(size=14, color='#00f2a9'))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gauge chart for confidence
            confidence = float(probs[sentiment_idx])
            
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level", 'font': {'size': 20, 'color': '#00f2a9'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
                    'bar': {'color': "#00f2a9"},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.3)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255,107,107,0.3)'},
                        {'range': [50, 75], 'color': 'rgba(255,217,61,0.3)'},
                        {'range': [75, 100], 'color': 'rgba(0,242,169,0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                height=300,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Disclaimer
            st.markdown(
                """
                <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 30px 0 20px 0;">
                <p style="font-size: 0.85em; color: #666; text-align: center; line-height: 1.5;">
                    <em>⚠️ Disclaimer: This sentiment analysis is powered by FinBERT AI and is for educational purposes only. 
                    Not financial advice. Always conduct your own research before making investment decisions.</em>
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze!")
