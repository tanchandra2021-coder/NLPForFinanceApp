from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error

class handler(BaseHTTPRequestHandler):
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Get stock data using Yahoo Finance quote API"""
        try:
            tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX',
                      'AMD', 'INTC', 'PYPL', 'COIN', 'SQ', 'SHOP', 'UBER', 'ABNB']
            
            stocks_data = []
            
            # Use Yahoo Finance query API (no auth required)
            symbols = ','.join(tickers)
            url = f'https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}'
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())
                
                if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                    for quote in data['quoteResponse']['result']:
                        symbol = quote.get('symbol', '')
                        current_price = quote.get('regularMarketPrice', 0)
                        prev_close = quote.get('regularMarketPreviousClose', current_price)
                        
                        if prev_close and prev_close > 0:
                            change_percent = ((current_price - prev_close) / prev_close) * 100
                        else:
                            change_percent = 0
                        
                        stocks_data.append({
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'change': round(change_percent, 2)
                        })
            
            # If we got no data, return empty array so frontend uses fallback
            if not stocks_data:
                stocks_data = []
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'stocks': stocks_data}).encode())
            
        except Exception as e:
            print(f"Error: {e}")
            # Return empty array on error so frontend uses fallback
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'stocks': []}).encode())
