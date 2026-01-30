from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf

class handler(BaseHTTPRequestHandler):
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Get real-time stock data using yfinance"""
        try:
            tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX',
                      'AMD', 'INTC', 'PYPL', 'COIN', 'SQ', 'SHOP', 'UBER', 'ABNB']
            
            stocks_data = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Get current price and change
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    prev_close = info.get('previousClose', current_price)
                    
                    # Calculate percent change
                    if prev_close and prev_close > 0:
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                    else:
                        change_percent = 0
                    
                    stocks_data.append({
                        'symbol': ticker,
                        'price': round(current_price, 2),
                        'change': round(change_percent, 2)
                    })
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    # Add placeholder data if fetch fails
                    stocks_data.append({
                        'symbol': ticker,
                        'price': 0,
                        'change': 0
                    })
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'stocks': stocks_data}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Error fetching stocks: {str(e)}'}).encode())
