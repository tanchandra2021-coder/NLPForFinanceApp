from http.server import BaseHTTPRequestHandler
import json
import re

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            # Get request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                self._send_error_response(f"Invalid JSON: {str(e)}")
                return
            
            text = data.get('text', '').strip()
            
            if not text:
                self._send_error_response('No text provided')
                return
            
            # Analyze sentiment using rule-based approach
            sentiment_result = self._analyze_sentiment(text)
            
            self._send_success_response(sentiment_result)
            
        except Exception as e:
            self._send_error_response(f'Analysis failed: {str(e)}')
    
    def _analyze_sentiment(self, text):
        """Rule-based sentiment analysis for financial text"""
        text_lower = text.lower()
        
        # Financial positive keywords with weights
        positive_keywords = {
            'surge': 3, 'surged': 3, 'surging': 3, 'soar': 3, 'soared': 3, 'soaring': 3,
            'rally': 2.5, 'rallied': 2.5, 'rallying': 2.5, 'jump': 2.5, 'jumped': 2.5,
            'spike': 2.5, 'spiked': 2.5, 'gain': 2, 'gains': 2, 'gained': 2,
            'up': 1.5, 'rise': 2, 'rose': 2, 'rising': 2, 'increase': 2, 'increased': 2,
            'growth': 2, 'growing': 2, 'profit': 2.5, 'profits': 2.5, 'profitable': 2.5,
            'beat': 2.5, 'beats': 2.5, 'exceed': 2.5, 'exceeded': 2.5, 'exceeding': 2.5,
            'strong': 2, 'stronger': 2, 'strength': 2, 'robust': 2.5, 'bullish': 2.5,
            'positive': 2, 'optimistic': 2, 'outperform': 2.5, 'outperformed': 2.5,
            'success': 2, 'successful': 2, 'breakthrough': 3, 'record': 2.5,
            'high': 1.5, 'higher': 1.5, 'boom': 2.5, 'booming': 2.5, 'stellar': 3,
            'impressive': 2, 'excellent': 2.5, 'outstanding': 2.5, 'upgrade': 2,
            'upgraded': 2, 'accelerate': 2, 'accelerated': 2, 'momentum': 1.5
        }
        
        # Financial negative keywords with weights
        negative_keywords = {
            'plunge': 3, 'plunged': 3, 'plunging': 3, 'crash': 3, 'crashed': 3,
            'collapse': 3, 'collapsed': 3, 'tumble': 2.5, 'tumbled': 2.5,
            'drop': 2, 'dropped': 2, 'dropping': 2, 'fall': 2, 'fell': 2, 'falling': 2,
            'decline': 2, 'declined': 2, 'declining': 2, 'decrease': 2, 'decreased': 2,
            'loss': 2.5, 'losses': 2.5, 'lose': 2.5, 'lost': 2.5, 'losing': 2.5,
            'down': 1.5, 'lower': 1.5, 'weak': 2, 'weaker': 2, 'weakness': 2,
            'bearish': 2.5, 'negative': 2, 'pessimistic': 2, 'miss': 2.5, 'missed': 2.5,
            'underperform': 2.5, 'underperformed': 2.5, 'concern': 2, 'concerns': 2,
            'worried': 2, 'worry': 2, 'risk': 1.5, 'risks': 1.5, 'trouble': 2,
            'struggle': 2, 'struggled': 2, 'struggling': 2, 'fail': 2.5, 'failed': 2.5,
            'disappoint': 2.5, 'disappointed': 2.5, 'disappointing': 2.5,
            'slump': 2.5, 'slumped': 2.5, 'downgrade': 2, 'downgraded': 2,
            'cut': 1.5, 'cuts': 1.5, 'slash': 2, 'slashed': 2, 'reduce': 1.5,
            'reduction': 1.5, 'low': 1.5, 'worst': 3, 'bad': 2, 'poor': 2
        }
        
        # Neutral/balancing words
        neutral_keywords = {
            'stable': 1, 'steady': 1, 'unchanged': 1, 'flat': 1, 'mixed': 1,
            'moderate': 1, 'neutral': 1, 'hold': 0.5, 'maintain': 0.5, 'expect': 0.5
        }
        
        # Calculate scores
        positive_score = 0
        negative_score = 0
        neutral_score = 0
        
        # Count keyword occurrences
        for word, weight in positive_keywords.items():
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
            positive_score += count * weight
        
        for word, weight in negative_keywords.items():
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
            negative_score += count * weight
        
        for word, weight in neutral_keywords.items():
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
            neutral_score += count * weight
        
        # Check for percentage increases/decreases
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for pct in percentages:
            value = float(pct)
            if value > 0:
                # Look for context around the percentage
                if any(word in text_lower for word in ['up', 'gain', 'increase', 'rise', 'surge', 'jump']):
                    positive_score += min(value / 2, 5)
                elif any(word in text_lower for word in ['down', 'loss', 'decrease', 'fall', 'drop', 'decline']):
                    negative_score += min(value / 2, 5)
        
        # Normalize scores
        total_score = positive_score + negative_score + neutral_score
        
        if total_score == 0:
            # No clear sentiment indicators
            pos_prob = 0.33
            neg_prob = 0.33
            neu_prob = 0.34
        else:
            pos_prob = positive_score / total_score
            neg_prob = negative_score / total_score
            neu_prob = neutral_score / total_score
            
            # Redistribute if neutral is too high
            if neu_prob > 0.5:
                excess = neu_prob - 0.3
                pos_prob += excess / 2
                neg_prob += excess / 2
                neu_prob = 0.3
        
        # Ensure probabilities sum to 1
        total = pos_prob + neg_prob + neu_prob
        pos_prob /= total
        neg_prob /= total
        neu_prob /= total
        
        # Determine overall sentiment
        max_prob = max(pos_prob, neg_prob, neu_prob)
        if max_prob == pos_prob:
            sentiment = "Positive"
            confidence = pos_prob
        elif max_prob == neg_prob:
            sentiment = "Negative"
            confidence = neg_prob
        else:
            sentiment = "Neutral"
            confidence = neu_prob
        
        # Calculate movement
        sentiment_spread = abs(pos_prob - neg_prob)
        probs_sorted = sorted([pos_prob, neu_prob, neg_prob], reverse=True)
        sentiment_certainty = probs_sorted[0] - probs_sorted[1]
        
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
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'positive': round(pos_prob, 4),
                'neutral': round(neu_prob, 4),
                'negative': round(neg_prob, 4)
            },
            'movement': movement,
            'sentiment_spread': round(sentiment_spread, 4),
            'sentiment_certainty': round(sentiment_certainty, 4)
        }
    
    def _send_success_response(self, data):
        """Send a successful JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error_response(self, error_message):
        """Send an error JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        response = {'error': error_message}
        self.wfile.write(json.dumps(response).encode())
