
from http.server import BaseHTTPRequestHandler
import json
import re

class handler(BaseHTTPRequestHandler):
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests for testing"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {'status': 'ok', 'message': 'Sentiment Analysis API is running'}
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """Handle sentiment analysis requests"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error('No data received')
                return
                
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            try:
                data = json.loads(post_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_error(f'Invalid JSON: {str(e)}')
                return
            
            # Get text
            text = data.get('text', '').strip()
            if not text:
                self._send_error('No text provided')
                return
            
            # Analyze sentiment
            result = self._analyze_sentiment(text)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self._send_error(f'Server error: {str(e)}')
    
    def _send_error(self, message):
        """Send error response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment using keyword matching"""
        text_lower = text.lower()
        
        # Positive keywords with weights
        positive = {
            'surge': 3, 'surged': 3, 'surging': 3, 'soar': 3, 'soared': 3, 'soaring': 3,
            'rally': 2.5, 'rallied': 2.5, 'jump': 2.5, 'jumped': 2.5, 'spike': 2.5,
            'gain': 2, 'gains': 2, 'gained': 2, 'up': 1.5, 'rise': 2, 'rose': 2,
            'increase': 2, 'increased': 2, 'growth': 2, 'profit': 2.5, 'profits': 2.5,
            'beat': 2.5, 'exceed': 2.5, 'exceeded': 2.5, 'strong': 2, 'robust': 2.5,
            'bullish': 2.5, 'positive': 2, 'optimistic': 2, 'outperform': 2.5,
            'success': 2, 'breakthrough': 3, 'record': 2.5, 'high': 1.5, 'boom': 2.5,
            'stellar': 3, 'impressive': 2, 'excellent': 2.5, 'outstanding': 2.5,
            'upgrade': 2, 'accelerate': 2, 'momentum': 1.5
        }
        
        # Negative keywords with weights
        negative = {
            'plunge': 3, 'plunged': 3, 'crash': 3, 'crashed': 3, 'collapse': 3,
            'tumble': 2.5, 'tumbled': 2.5, 'drop': 2, 'dropped': 2, 'fall': 2,
            'fell': 2, 'decline': 2, 'declined': 2, 'decrease': 2, 'loss': 2.5,
            'losses': 2.5, 'lose': 2.5, 'lost': 2.5, 'down': 1.5, 'lower': 1.5,
            'weak': 2, 'weakness': 2, 'bearish': 2.5, 'negative': 2, 'pessimistic': 2,
            'miss': 2.5, 'missed': 2.5, 'underperform': 2.5, 'concern': 2, 'worry': 2,
            'risk': 1.5, 'trouble': 2, 'struggle': 2, 'struggled': 2, 'fail': 2.5,
            'disappoint': 2.5, 'disappointed': 2.5, 'slump': 2.5, 'downgrade': 2,
            'cut': 1.5, 'slash': 2, 'reduce': 1.5, 'low': 1.5, 'worst': 3, 'bad': 2,
            'poor': 2
        }
        
        # Calculate scores
        pos_score = sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) * wt 
                       for w, wt in positive.items())
        neg_score = sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) * wt 
                       for w, wt in negative.items())
        
        # Handle percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for pct_str in percentages:
            pct = float(pct_str)
            if any(w in text_lower for w in ['up', 'gain', 'increase', 'rise', 'surge']):
                pos_score += min(pct / 2, 5)
            elif any(w in text_lower for w in ['down', 'loss', 'decrease', 'fall', 'drop']):
                neg_score += min(pct / 2, 5)
        
        # Calculate probabilities
        total = pos_score + neg_score + 1
        pos_prob = pos_score / total
        neg_prob = neg_score / total
        neu_prob = 1 / total
        
        # Normalize
        total_prob = pos_prob + neg_prob + neu_prob
        pos_prob /= total_prob
        neg_prob /= total_prob
        neu_prob /= total_prob
        
        # Determine sentiment
        if pos_prob > neg_prob and pos_prob > neu_prob:
            sentiment = "Positive"
            confidence = pos_prob
        elif neg_prob > pos_prob and neg_prob > neu_prob:
            sentiment = "Negative"
            confidence = neg_prob
        else:
            sentiment = "Neutral"
            confidence = neu_prob
        
        # Calculate movement
        spread = abs(pos_prob - neg_prob)
        certainty = max(pos_prob, neg_prob, neu_prob) - sorted([pos_prob, neg_prob, neu_prob])[1]
        
        if sentiment == "Positive":
            movement = min(7.5, round(confidence * spread * 8 + certainty * 2, 2))
        elif sentiment == "Negative":
            movement = -min(8.5, round(confidence * spread * 9 + certainty * 2.5, 2))
        else:
            movement = round((pos_prob - neg_prob) * 2, 2)
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'positive': round(pos_prob, 4),
                'neutral': round(neu_prob, 4),
                'negative': round(neg_prob, 4)
            },
            'movement': movement,
            'sentiment_spread': round(spread, 4),
            'sentiment_certainty': round(certainty, 4)
        }
