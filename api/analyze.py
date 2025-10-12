from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Handle CORS
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX']
        stock_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                prev_close = info.get('previousClose', current_price)
                
                if prev_close and prev_close > 0:
                    change_percent = ((current_price - prev_close) / prev_close) * 100
                else:
                    change_percent = 0
                
                stock_data.append({
                    'ticker': ticker,
                    'change': round(change_percent, 2)
                })
            except:
                # Fallback if API fails
                stock_data.append({
                    'ticker': ticker,
                    'change': 0
                })
        
        response = {'stocks': stock_data}
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
