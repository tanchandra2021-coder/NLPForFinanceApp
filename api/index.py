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
        """Analyze sentiment using comprehensive keyword matching and context"""
        text_lower = text.lower()
        
        # EXTREMELY POSITIVE (4.0-5.0 weight) - Major breakthroughs
        extreme_positive = {
            'skyrocket': 5, 'skyrocketed': 5, 'skyrocketing': 5, 'skyrockets': 5,
            'explode': 4.5, 'exploded': 4.5, 'explodes': 4.5, 'exploding': 4.5, 'explosion': 4.5,
            'breakthrough': 4.5, 'blockbuster': 4.5, 'bonanza': 4.5,
            'stellar': 4, 'spectacular': 4, 'phenomenal': 4, 'exceptional': 4,
            'outstanding': 4, 'remarkable': 4, 'unprecedented': 4,
            'revolutionary': 4, 'game-changing': 4.5, 'record-breaking': 4.5,
            'triumph': 4, 'triumphant': 4, 'dominate': 4, 'dominates': 4, 'dominated': 4
        }
        
        # VERY POSITIVE (2.5-3.9 weight)
        very_positive = {
            'surge': 3.5, 'surged': 3.5, 'surges': 3.5, 'surging': 3.5,
            'soar': 3.5, 'soared': 3.5, 'soars': 3.5, 'soaring': 3.5,
            'rocket': 3.5, 'rocketed': 3.5, 'rockets': 3.5, 'rocketing': 3.5,
            'rally': 3, 'rallied': 3, 'rallies': 3, 'rallying': 3,
            'spike': 3, 'spiked': 3, 'spikes': 3, 'spiking': 3,
            'jump': 3, 'jumped': 3, 'jumps': 3, 'jumping': 3, 'leap': 3, 'leaped': 3,
            'boom': 3, 'booming': 3, 'boomed': 3, 'booms': 3,
            'thrive': 3, 'thrived': 3, 'thrives': 3, 'thriving': 3,
            'excel': 3, 'excels': 3, 'excelled': 3, 'excellent': 3, 'excellence': 3,
            'beat': 3, 'beats': 3, 'beating': 3, 'crushed': 3, 'crush': 3,
            'exceed': 3, 'exceeds': 3, 'exceeded': 3, 'exceeding': 3,
            'outperform': 3, 'outperforms': 3, 'outperformed': 3, 'outperforming': 3,
            'bullish': 3, 'robust': 3, 'strong': 2.5, 'stronger': 2.5, 'strongest': 3,
            'impressive': 2.5, 'record': 3, 'record-high': 3.5, 'all-time high': 3.5
        }
        
        # POSITIVE (1.5-2.4 weight)
        positive = {
            'gain': 2, 'gains': 2, 'gained': 2, 'gaining': 2,
            'rise': 2, 'rises': 2, 'rose': 2, 'rising': 2, 'risen': 2,
            'increase': 2, 'increases': 2, 'increased': 2, 'increasing': 2,
            'grow': 2, 'grows': 2, 'grew': 2, 'growth': 2, 'growing': 2,
            'profit': 2.5, 'profits': 2.5, 'profitable': 2.5, 'profitability': 2.5,
            'revenue': 1.5, 'revenues': 1.5, 'earning': 2, 'earnings': 2,
            'up': 1.5, 'upward': 1.8, 'upside': 2, 'upturn': 2, 'uptick': 2,
            'high': 1.5, 'higher': 1.8, 'highest': 2,
            'improve': 2, 'improves': 2, 'improved': 2, 'improving': 2, 'improvement': 2,
            'advance': 2, 'advances': 2, 'advanced': 2, 'advancing': 2,
            'success': 2, 'successful': 2, 'succeed': 2, 'succeeds': 2, 'succeeded': 2,
            'positive': 2, 'optimistic': 2, 'optimism': 2, 'confident': 2, 'confidence': 2,
            'win': 2, 'wins': 2, 'won': 2, 'winning': 2, 'winner': 2,
            'upgrade': 2, 'upgrades': 2, 'upgraded': 2, 'upgrading': 2,
            'accelerate': 2, 'accelerated': 2, 'accelerates': 2, 'accelerating': 2,
            'momentum': 1.8, 'expansion': 2, 'expand': 2, 'expands': 2, 'expanded': 2,
            'recovery': 2, 'recover': 2, 'recovered': 2, 'recovering': 2,
            'rebound': 2, 'rebounded': 2, 'rebounds': 2, 'rebounding': 2,
            'approve': 2, 'approves': 2, 'approved': 2, 'approval': 2,
            'benefit': 1.8, 'benefits': 1.8, 'beneficial': 1.8,
            'opportunity': 1.5, 'opportunities': 1.5, 'promising': 2, 'promise': 1.5
        }
        
        # EXTREMELY NEGATIVE (4.0-5.0 weight)
        extreme_negative = {
            'plunge': 4.5, 'plunged': 4.5, 'plunges': 4.5, 'plunging': 4.5,
            'crash': 5, 'crashed': 5, 'crashes': 5, 'crashing': 5,
            'collapse': 5, 'collapsed': 5, 'collapses': 5, 'collapsing': 5,
            'devastate': 4.5, 'devastated': 4.5, 'devastating': 4.5, 'devastation': 4.5,
            'disaster': 4.5, 'disastrous': 4.5, 'catastrophe': 4.5, 'catastrophic': 4.5,
            'crisis': 4, 'crises': 4, 'critical': 3.5,
            'plummet': 4.5, 'plummeted': 4.5, 'plummets': 4.5, 'plummeting': 4.5,
            'bankruptcy': 4.5, 'bankrupt': 4.5, 'insolvent': 4.5, 'insolvency': 4.5,
            'fraud': 4, 'fraudulent': 4, 'scandal': 4, 'scandalous': 4,
            'criminal': 4, 'illegal': 3.5, 'unlawful': 3.5,
            'recession': 4, 'recessionary': 4, 'depression': 4.5
        }
        
        # VERY NEGATIVE (2.5-3.9 weight)
        very_negative = {
            'tumble': 3, 'tumbled': 3, 'tumbles': 3, 'tumbling': 3,
            'slump': 3, 'slumped': 3, 'slumps': 3, 'slumping': 3,
            'sink': 3, 'sinks': 3, 'sank': 3, 'sinking': 3, 'sunk': 3,
            'dive': 3, 'dived': 3, 'dives': 3, 'diving': 3,
            'nosedive': 3.5, 'nosedived': 3.5, 'nosedives': 3.5,
            'tank': 3, 'tanked': 3, 'tanks': 3, 'tanking': 3,
            'plummet': 3.5, 'fail': 3, 'fails': 3, 'failed': 3, 'failing': 3, 'failure': 3,
            'loss': 3, 'losses': 3, 'lose': 3, 'loses': 3, 'lost': 3, 'losing': 3,
            'disappoint': 3, 'disappoints': 3, 'disappointed': 3, 'disappointing': 3, 'disappointment': 3,
            'miss': 3, 'missed': 3, 'misses': 3, 'missing': 3,
            'underperform': 3, 'underperforms': 3, 'underperformed': 3, 'underperforming': 3,
            'struggle': 3, 'struggles': 3, 'struggled': 3, 'struggling': 3,
            'suffer': 3, 'suffers': 3, 'suffered': 3, 'suffering': 3,
            'bearish': 3, 'pessimistic': 3, 'pessimism': 3,
            'lawsuit': 3, 'lawsuits': 3, 'sue': 3, 'sued': 3, 'suing': 3, 'litigation': 3,
            'allege': 2.5, 'alleges': 2.5, 'alleged': 2.5, 'alleging': 2.5, 'allegation': 2.5, 'allegations': 2.5,
            'mislead': 3, 'misleads': 3, 'misleading': 3, 'misled': 3,
            'violate': 3, 'violates': 3, 'violated': 3, 'violation': 3, 'violations': 3,
            'penalty': 2.5, 'penalties': 2.5, 'fine': 2.5, 'fined': 2.5, 'fines': 2.5, 'fining': 2.5,
            'tariff': 3, 'tariffs': 3, 'sanction': 3, 'sanctions': 3, 'sanctioned': 3,
            'hostile': 3, 'hostility': 3, 'threat': 3, 'threatens': 3, 'threatened': 3, 'threatening': 3,
            'layoff': 3, 'layoffs': 3, 'downsize': 2.5, 'downsizing': 2.5, 'downsized': 2.5,
            'severance': 2.5, 'termination': 2.5, 'terminate': 2.5, 'terminated': 2.5,
            'investigate': 2.5, 'investigation': 2.5, 'investigating': 2.5, 'investigated': 2.5,
            'probe': 2.5, 'probing': 2.5, 'probed': 2.5,
            'worst': 3.5, 'worse': 3, 'worsen': 3, 'worsening': 3, 'worsened': 3
        }
        
        # NEGATIVE (1.5-2.4 weight)
        negative = {
            'drop': 2, 'dropped': 2, 'drops': 2, 'dropping': 2,
            'fall': 2, 'falls': 2, 'fell': 2, 'falling': 2, 'fallen': 2,
            'decline': 2, 'declined': 2, 'declines': 2, 'declining': 2,
            'decrease': 2, 'decreased': 2, 'decreases': 2, 'decreasing': 2,
            'down': 1.5, 'lower': 1.5, 'lowest': 2, 'downward': 2, 'downturn': 2,
            'low': 1.5, 'weaken': 2, 'weakens': 2, 'weakened': 2, 'weakening': 2,
            'weak': 2, 'weaker': 2, 'weakness': 2,
            'concern': 2, 'concerns': 2, 'concerned': 2, 'concerning': 2,
            'worry': 2, 'worried': 2, 'worries': 2, 'worrying': 2, 'worrisome': 2,
            'risk': 1.5, 'risks': 1.5, 'risky': 2, 'riskier': 2,
            'trouble': 2, 'troubled': 2, 'troubles': 2, 'troubling': 2,
            'bad': 2, 'poor': 2, 'poorer': 2, 'poorly': 2,
            'negative': 2, 'unfavorable': 2, 'adverse': 2, 'adversely': 2,
            'downgrade': 2, 'downgrades': 2, 'downgraded': 2, 'downgrading': 2,
            'cut': 1.8, 'cuts': 1.8, 'cutting': 1.8, 'slash': 2, 'slashed': 2, 'slashes': 2, 'slashing': 2,
            'reduce': 1.8, 'reduces': 1.8, 'reduced': 1.8, 'reducing': 1.8, 'reduction': 1.8,
            'slow': 1.8, 'slower': 1.8, 'slowing': 1.8, 'slowed': 1.8, 'slowdown': 2,
            'hurt': 2, 'hurts': 2, 'hurting': 2,
            'damage': 2, 'damages': 2, 'damaged': 2, 'damaging': 2,
            'dispute': 2, 'disputes': 2, 'disputed': 2, 'disputing': 2,
            'conflict': 2, 'conflicts': 2, 'tension': 2, 'tensions': 2,
            'volatile': 2, 'volatility': 2, 'uncertain': 2, 'uncertainty': 2,
            'debt': 1.8, 'debts': 1.8, 'default': 3, 'defaults': 3, 'defaulted': 3,
            'charge': 1.8, 'charges': 1.8, 'settlement': 2, 'settlements': 2,
            'cost': 1.5, 'costs': 1.5, 'costly': 2, 'expense': 1.5, 'expenses': 1.5,
            'challenge': 1.8, 'challenges': 1.8, 'challenging': 1.8, 'challenged': 1.8,
            'problem': 2, 'problems': 2, 'problematic': 2, 'issue': 1.5, 'issues': 1.5
        }
        
        # Merge all dictionaries
        all_positive = {**extreme_positive, **very_positive, **positive}
        all_negative = {**extreme_negative, **very_negative, **negative}
        
        # NEGATION HANDLING - Detect when negative words are negated
        # Words that negate/reverse sentiment
        negation_words = r'\b(no|not|without|eliminate|eliminates|eliminated|eliminating|remove|removes|removed|removing|end|ends|ended|ending|drop|drops|dropped|dropping|cut|cuts|cutting|reduce|reduces|reduced|reducing|decrease|decreases|decreased|decreasing|cancel|cancels|canceled|canceling|cancelled|cancelling|avoid|avoids|avoided|avoiding|prevent|prevents|prevented|preventing|reverse|reverses|reversed|reversing|lift|lifts|lifted|lifting|ease|eases|eased|easing|repeal|repeals|repealed|repealing)\b'
        
        # Calculate base scores with word boundary matching and negation detection
        pos_score = 0
        neg_score = 0
        
        for word, weight in all_positive.items():
            # Find all matches of this positive word
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                match_start = match.start()
                # Check 30 characters before the word for negation
                context_start = max(0, match_start - 30)
                context = text_lower[context_start:match_start]
                
                # If negated, this positive word becomes negative
                if re.search(negation_words, context):
                    neg_score += weight * 0.8  # Negated positive = negative (slightly reduced)
                else:
                    pos_score += weight
        
        for word, weight in all_negative.items():
            # Find all matches of this negative word
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                match_start = match.start()
                # Check 30 characters before the word for negation
                context_start = max(0, match_start - 30)
                context = text_lower[context_start:match_start]
                
                # If negated, this negative word becomes positive
                if re.search(negation_words, context):
                    pos_score += weight * 0.8  # Negated negative = positive (slightly reduced)
                else:
                    neg_score += weight
        
        # CONTEXT ANALYSIS - Handle negations and modifiers
        negation_pattern = r'\b(not|no|never|neither|without|lack|lacks|lacking)\b\s+\w+\s+(\w+)'
        negations = re.findall(negation_pattern, text_lower)
        
        # FINANCIAL METRICS ANALYSIS
        # Extract dollar amounts and context
        dollar_pattern = r'\$(\d+(?:\.\d+)?)\s*([BbMmKk]?)(?:\s+(billion|million|thousand))?'
        dollar_matches = re.findall(dollar_pattern, text)
        
        financial_boost = 0
        for amount_str, suffix, word_suffix in dollar_matches:
            amount = float(amount_str)
            
            # Convert to billions
            if suffix.upper() == 'B' or word_suffix == 'billion':
                amount_in_billions = amount
            elif suffix.upper() == 'M' or word_suffix == 'million':
                amount_in_billions = amount / 1000
            elif suffix.upper() == 'K' or word_suffix == 'thousand':
                amount_in_billions = amount / 1000000
            else:
                amount_in_billions = amount  # Assume billions if large number
            
            # Look at context around the dollar amount
            dollar_str = f'${amount_str}{suffix}'
            dollar_index = text.find(dollar_str)
            context_start = max(0, dollar_index - 50)
            context_end = min(len(text), dollar_index + 80)
            context = text[context_start:context_end].lower()
            
            # Determine if it's positive or negative based on context
            if any(word in context for word in ['gain', 'gains', 'profit', 'profits', 'income', 'revenue', 'earnings', 'beat', 'growth', 'increase', 'rose', 'up']):
                # Positive financial metric - scale by amount
                if amount_in_billions >= 5:
                    financial_boost += 3.5
                elif amount_in_billions >= 1:
                    financial_boost += 2.5
                elif amount_in_billions >= 0.1:
                    financial_boost += 1.5
                else:
                    financial_boost += 0.5
            elif any(word in context for word in ['loss', 'losses', 'charge', 'settlement', 'fine', 'penalty', 'severance', 'layoff', 'cost', 'expense', 'debt', 'write']):
                # Negative financial metric - scale by amount
                if amount_in_billions >= 5:
                    financial_boost -= 3.5
                elif amount_in_billions >= 1:
                    financial_boost -= 2.5
                elif amount_in_billions >= 0.1:
                    financial_boost -= 1.5
                else:
                    financial_boost -= 0.5
        
        # Apply financial boost
        if financial_boost > 0:
            pos_score += financial_boost
        else:
            neg_score += abs(financial_boost)
        
        # PERCENTAGE ANALYSIS - Enhanced
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for pct_str in percentages:
            pct = float(pct_str)
            
            # Find context around percentage
            pct_index = text.lower().find(pct_str)
            context_start = max(0, pct_index - 40)
            context_end = min(len(text_lower), pct_index + 40)
            pct_context = text_lower[context_start:context_end]
            
            # Scale percentage impact
            pct_weight = min(pct / 5, 8)  # Cap at 8 for very large percentages
            
            if any(w in pct_context for w in ['up', 'gain', 'increase', 'rise', 'surge', 'jump', 'grow', 'growth', 'higher', 'beat', 'exceed']):
                pos_score += pct_weight
            elif any(w in pct_context for w in ['down', 'loss', 'decrease', 'fall', 'drop', 'decline', 'lower', 'miss', 'cut', 'reduce']):
                neg_score += pct_weight
        
        # SPECIAL PATTERNS
        # "but" clause analysis - what comes after "but" is usually more important
        but_pattern = r'\bbut\b(.+?)(?:\.|$)'
        but_clauses = re.findall(but_pattern, text_lower, re.IGNORECASE)
        for clause in but_clauses:
            # Weight the "but" clause more heavily (1.5x)
            for word, weight in all_positive.items():
                matches = len(re.findall(r'\b' + re.escape(word) + r'\b', clause))
                pos_score += matches * weight * 0.5
            for word, weight in all_negative.items():
                matches = len(re.findall(r'\b' + re.escape(word) + r'\b', clause))
                neg_score += matches * weight * 0.5
        
        # "includes" or "despite" analysis - secondary information
        secondary_pattern = r'\b(includes?|despite|although|however|while)\b(.+?)(?:\.|$)'
        secondary_clauses = re.findall(secondary_pattern, text_lower, re.IGNORECASE)
        for marker, clause in secondary_clauses:
            # Weight secondary clauses slightly less (0.7x) unless it's "despite" (1.2x for what's achieved despite challenges)
            if marker in ['despite', 'although']:
                multiplier = 1.2
            else:
                multiplier = 0.7
            
            for word, weight in all_positive.items():
                matches = len(re.findall(r'\b' + re.escape(word) + r'\b', clause))
                if marker in ['despite', 'although']:
                    pos_score += matches * weight * multiplier
                else:
                    pos_score += matches * weight * multiplier
            for word, weight in all_negative.items():
                matches = len(re.findall(r'\b' + re.escape(word) + r'\b', clause))
                neg_score += matches * weight * multiplier
        
        # Calculate probabilities with adjusted baseline
        total = pos_score + neg_score + 0.5  # Reduced baseline for stronger signals
        pos_prob = pos_score / total
        neg_prob = neg_score / total
        neu_prob = 0.5 / total
        
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
        
        # Calculate movement with enhanced scaling
        spread = abs(pos_prob - neg_prob)
        certainty = max(pos_prob, neg_prob, neu_prob) - sorted([pos_prob, neg_prob, neu_prob])[1]
        
        # Enhanced movement calculation
        base_movement = confidence * spread * 10
        certainty_boost = certainty * 5
        
        if sentiment == "Positive":
            movement = min(15, round(base_movement + certainty_boost, 2))
        elif sentiment == "Negative":
            movement = -min(15, round(base_movement + certainty_boost, 2))
        else:
            movement = round((pos_prob - neg_prob) * 3, 2)
        
        # Debug info for score breakdown
        score_breakdown = {
            'pos_score': round(pos_score, 2),
            'neg_score': round(neg_score, 2),
            'financial_boost': round(financial_boost, 2)
        }
        
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
            'sentiment_certainty': round(certainty, 4),
            'score_breakdown': score_breakdown
        }
