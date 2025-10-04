import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import base64
import plotly.graph_objects as go

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

/* Center everything */
.block-container {
    max-width: 900px !important;
    padding: 2rem !important;
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

/* Demo section */
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
</style>
""", unsafe_allow_html=True)

# --- App layout ---
st.markdown('<div class="title-bubble">📈 Finance News Sentiment & Stock Movement Predictor</div>', unsafe_allow_html=True)

st.markdown('<div class="instruction-box">💭 Paste your stock news, tweets, or finance text below. We\'ll analyze sentiment, predict stock movement, and visualize the results with beautiful charts!</div>', unsafe_allow_html=True)

# Demo mode toggle
demo_mode = st.checkbox("🎬 Show Demo Example", value=False)

if demo_mode:
    st.markdown('<div class="demo-tag">🌟 DEMO MODE ACTIVE</div>', unsafe_allow_html=True)
    text = "Apple Inc. announces record-breaking Q4 earnings, exceeding analyst expectations by 15%. iPhone sales surge with new product lineup, pushing stock to all-time highs."
    st.text_area("📝 Enter your finance text here:", text, key="finance_text", height=150)
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

            # Stock movement calculation
            movement = 0.0
            if sentiment == "Positive":
                movement = min(10, round(float(probs[sentiment_idx]) * 10, 2))
            elif sentiment == "Negative":
                movement = -min(10, round(float(probs[sentiment_idx]) * 10, 2))
            
            # Results display
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            
            # Sentiment badges
            st.markdown('<h2>🎯 Predicted Sentiment</h2>', unsafe_allow_html=True)
            if sentiment == "Positive":
                st.markdown(f'<div style="text-align: center;"><span class="sentiment-badge positive-badge">📈 {sentiment}</span></div>', unsafe_allow_html=True)
            elif sentiment == "Negative":
                st.markdown(f'<div style="text-align: center;"><span class="sentiment-badge negative-badge">📉 {sentiment}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: center;"><span class="sentiment-badge neutral-badge">➖ {sentiment}</span></div>', unsafe_allow_html=True)
            
            # Movement prediction
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1em; color: #aaa; margin-bottom: 5px;">Predicted Stock Movement</div>', unsafe_allow_html=True)
            if movement > 0:
                st.markdown(f'<div class="stat-value">+{movement}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stat-value">{movement}%</div>', unsafe_allow_html=True)
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
