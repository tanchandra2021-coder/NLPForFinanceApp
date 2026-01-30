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
        """Get stock data using Yahoo Finance query API"""
        try:
            tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX',
                      'AMD', 'INTC', 'PYPL', 'COIN', 'SQ', 'SHOP', 'UBER', 'ABNB']
            
            stocks_data = []
            
            # Try Yahoo Finance API
            symbols = ','.join(tickers)
            url = f'https://query2.finance.yahoo.com/v7/finance/quote?symbols={symbols}&fields=regularMarketPrice,regularMarketChange,regularMarketChangePercent'
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            try:
                with urllib.request.urlopen(req, timeout=8) as response:
                    data = json.loads(response.read().decode())
                    
                    if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                        results = data['quoteResponse']['result']
                        for quote in results:
                            symbol = quote.get('symbol', '')
                            price = quote.get('regularMarketPrice', 0)
                            change = quote.get('regularMarketChangePercent', 0)
                            
                            if price > 0:  # Only add if we got valid data
                                stocks_data.append({
                                    'symbol': symbol,
                                    'price': round(price, 2),
                                    'change': round(change, 2)
                                })
            except Exception as fetch_error:
                print(f"Fetch error: {fetch_error}")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'public, max-age=60')
            self.end_headers()
            self.wfile.write(json.dumps({'stocks': stocks_data}).encode())
            
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'stocks': []}).encode())
