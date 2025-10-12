from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error

# Use Hugging Face Inference API (free, no API key needed for public models)
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

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
        
        try:
            # Call Hugging Face Inference API
            payload = json.dumps({"inputs": text}).encode('utf-8')
            req = urllib.request.Request(HF_API_URL, data=payload, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=30) as response_data:
                result = json.loads(response_data.read().decode('utf-8'))
            
            # Parse HF response - format: [[{"label": "positive", "score": 0.99}, ...]]
            if isinstance(result, list) and len(result) > 0:
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
                
                # Response
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
                
                self.wfile.write(json.dumps(response_data).encode())
            else:
                raise Exception("Unexpected response format from Hugging Face API")
                
        except urllib.error.HTTPError as e:
            error_msg = f"Hugging Face API error: {e.code}"
            if e.code == 503:
                error_msg = "Model is loading. Please try again in 20 seconds."
            response_data = {'error': error_msg}
            self.wfile.write(json.dumps(response_data).encode())
        except Exception as e:
            response_data = {'error': f'Analysis failed: {str(e)}'}
            self.wfile.write(json.dumps(response_data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
