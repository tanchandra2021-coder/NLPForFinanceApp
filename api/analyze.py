from http.server import BaseHTTPRequestHandler
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load model once (will be cached by Vercel)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Handle CORS
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Get request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        text = data.get('text', '')
        
        if not text.strip():
            response = {'error': 'No text provided'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Model prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        
        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_idx = np.argmax(probs)
        sentiment = sentiment_labels[sentiment_idx]
        
        # Calculate movement
        confidence = float(probs[sentiment_idx])
        pos_prob = float(probs[0])
        neg_prob = float(probs[2])
        
        sentiment_spread = abs(pos_prob - neg_prob)
        sentiment_certainty = max(probs) - sorted(probs)[-2]
        
        movement = 0.0
        if sentiment == "Positive":
            base_movement = confidence * sentiment_spread * 8
            certainty_boost = sentiment_certainty * 2
            movement = min(7.5, round(base_movement + certainty_boost, 2))
        elif sentiment == "Negative":
            base_movement = confidence * sentiment_spread * 9
            certainty_boost = sentiment_certainty * 2.5
            movement = -min(8.5, round(base_movement + certainty_boost, 2))
        else:
            movement = round((pos_prob - neg_prob) * 2, 2)
        
        # Response
        response = {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': pos_prob,
                'neutral': float(probs[1]),
                'negative': neg_prob
            },
            'movement': movement,
            'sentiment_spread': sentiment_spread,
            'sentiment_certainty': sentiment_certainty
        }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
