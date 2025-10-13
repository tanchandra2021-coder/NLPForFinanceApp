from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
import sys

# Use Hugging Face Inference API (free, no API key needed for public models)
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

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
            
            # Call Hugging Face Inference API
            try:
                payload = json.dumps({"inputs": text}).encode('utf-8')
                req = urllib.request.Request(
                    HF_API_URL, 
                    data=payload, 
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=30) as response_data:
                    result = json.loads(response_data.read().decode('utf-8'))
                
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8') if e.fp else ''
                if e.code == 503:
                    self._send_error_response("Model is loading. Please wait 20-30 seconds and try again.")
                else:
                    self._send_error_response(f"API error ({e.code}): {error_body}")
                return
            except urllib.error.URLError as e:
                self._send_error_response(f"Network error: {str(e)}")
                return
            except Exception as e:
                self._send_error_response(f"Request failed: {str(e)}")
                return
            
            # Parse HF response - format: [[{"label": "positive", "score": 0.99}, ...]]
            if not isinstance(result, list) or len(result) == 0:
                self._send_error_response(f"Unexpected API response format: {result}")
                return
            
            predictions = result[0]
            
            # Extract probabilities
            probs_dict = {item['label'].lower(): item['score'] for item in predictions}
            pos_prob = probs_dict.get('positive', 0.0)
            neu_prob = probs_dict.get('neutral', 0.0)
            neg_prob = probs_dict.get('negative', 0.0)
            
            # Determine sentiment
            max_prob = max(pos_prob, neu_prob, neg_prob)
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
            
            # Send success response
            response_data = {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'positive': pos_prob,
                    'neutral': neu_prob,
                    'negative': neg_prob
                },
                'movement': movement,
                'sentiment_spread': sentiment_spread,
                'sentiment_certainty': sentiment_certainty
            }
            
            self._send_success_response(response_data)
            
        except Exception as e:
            # Catch-all for any unexpected errors
            self._send_error_response(f'Unexpected error: {str(e)}')
            print(f"Error in handler: {str(e)}", file=sys.stderr)
    
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
        self.send_response(200)  # Still send 200 for CORS compatibility
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        response = {'error': error_message}
        self.wfile.write(json.dumps(response).encode())
        print(f"Error response: {error_message}", file=sys.stderr)
