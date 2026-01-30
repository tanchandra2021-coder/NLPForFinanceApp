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
        API_KEY = 'P5NB85EKOHDBOHD8'
        
        tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX',
                  'AMD', 'INTC', 'PYPL', 'COIN', 'SQ', 'SHOP', 'UBER', 'ABNB']
        
        stocks_data = []
        
        # Batch request for quotes
        symbols = ','.join(tickers)
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbols}&apikey={API_KEY}'
        
        # For demo, get just a few to avoid rate limits
        for ticker in tickers[:6]:  # Get first 6 stocks
            try:
                quote_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}'
                with urllib.request.urlopen(quote_url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    
                    if 'Global Quote' in data:
                        quote = data['Global Quote']
                        price = float(quote.get('05. price', 0))
                        change_pct = float(quote.get('10. change percent', '0').replace('%', ''))
                        
                        stocks_data.append({
                            'symbol': ticker,
                            'price': round(price, 2),
                            'change': round(change_pct, 2)
                        })
            except:
                pass
        
        # Fill rest with static data to avoid rate limits
        static_data = [
            {'symbol': 'AMD', 'price': 138.67, 'change': 0.98},
            {'symbol': 'INTC', 'price': 21.33, 'change': -0.76},
            {'symbol': 'PYPL', 'price': 89.22, 'change': 1.45},
            {'symbol': 'COIN', 'price': 268.91, 'change': 4.12},
            {'symbol': 'SQ', 'price': 95.44, 'change': -1.89},
            {'symbol': 'SHOP', 'price': 112.76, 'change': 2.34},
            {'symbol': 'UBER', 'price': 78.55, 'change': -0.45},
            {'symbol': 'ABNB', 'price': 162.88, 'change': 1.67},
            {'symbol': 'NFLX', 'price': 712.45, 'change': -1.23},
            {'symbol': 'META', 'price': 638.12, 'change': 2.15}
        ]
        
        stocks_data.extend(static_data)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'stocks': stocks_data}).encode())
