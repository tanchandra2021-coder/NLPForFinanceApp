# app.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

st.title("📈 Finance News Sentiment & Stock Movement Predictor")

# User input
user_input = st.text_area("Enter stock news, tweets, or finance text:")

if user_input:
    # Run sentiment analysis
    results = sentiment_pipeline(user_input, return_all_scores=True)[0]
    
    # Convert results into dictionary {label: score}
    sentiment_scores = {res['label']: res['score'] for res in results}
    
    # Determine sentiment
    sentiment = max(sentiment_scores, key=sentiment_scores.get)
    
    # Predict stock movement (capped at ±10%)
    if sentiment == "POSITIVE":
        predicted_movement = min(10.0, sentiment_scores['POSITIVE'] * 10)  # up to +10%
    elif sentiment == "NEGATIVE":
        predicted_movement = max(-10.0, -sentiment_scores['NEGATIVE'] * 10)  # down to -10%
    else:
        predicted_movement = 0.0  # Neutral sentiment = ~0% movement

    # Display results
    st.subheader("Sentiment Probabilities:")
    st.write({
        "Negative": round(sentiment_scores.get("NEGATIVE", 0.0), 4),
        "Neutral": round(sentiment_scores.get("NEUTRAL", 0.0), 4),
        "Positive": round(sentiment_scores.get("POSITIVE", 0.0), 4),
    })

    st.subheader("Predicted Sentiment & Stock Movement:")
    st.write(f"**Sentiment:** {sentiment.capitalize()}")
    st.write(f"**Predicted Stock Movement:** {predicted_movement:.2f}%")


