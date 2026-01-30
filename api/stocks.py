from http.server import BaseHTTPRequestHandler
import json
import urllib.request

class handler(BaseHTTPRequestHandler):
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        API_KEY = 'YOUR_FINNHUB_API_KEY_HERE'  # Replace this!
        
        tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX',
                  'AMD', 'INTC', 'PYPL', 'COIN', 'SQ', 'SHOP', 'UBER', 'ABNB']
        
        stocks_data = []
        
        for ticker in tickers:
            try:
                url = f'https://finnhub.io/api/v1/quote?symbol={ticker}&token={API_KEY}'
                with urllib.request.urlopen(url, timeout=3) as response:
                    data = json.loads(response.read().decode())
                    
                    current_price = data.get('c', 0)  # Current price
                    prev_close = data.get('pc', current_price)  # Previous close
                    
                    if current_price > 0 and prev_close > 0:
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        stocks_data.append({
                            'symbol': ticker,
                            'price': round(current_price, 2),
                            'change': round(change_pct, 2)
                        })
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=300')  # Cache 5 minutes
        self.end_headers()
        self.wfile.write(json.dumps({'stocks': stocks_data}).encode())
