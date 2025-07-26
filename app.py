from flask import Flask, render_template, request, redirect, url_for
import joblib
import re
import spacy
import spacy.cli
import os
from textblob import TextBlob
from collections import defaultdict

# os.environ['SPACY_ALLOW_DOWNLOAD'] = 'true'

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

# Load NLP models
# nlp = spacy.load('en_core_web_sm')
models = {
    'toxic': joblib.load('model/toxic_model.pkl'),
    'severe_toxic': joblib.load('model/severe_toxic_model.pkl'),
    'obscene': joblib.load('model/obscene_model.pkl'),
    'threat': joblib.load('model/threat_model.pkl'),
    'insult': joblib.load('model/insult_model.pkl'),
    'identity_hate': joblib.load('model/identity_hate_model.pkl')
}

# Abuse patterns with corresponding categories
ABUSE_PATTERNS = [
    (r'\b(you\'?re|you|u)\s+(too|so|very)?\s*(stupid|idiot|dumb|worthless)\b', 'insult'),
    (r'\b(you|u)\s+(always|never)\s+\w+', 'toxic'),
    (r'\b(nobody|no one)\s+(likes|love|care).*you\b', 'toxic'),
    (r'\b(you|u)\s+(fail|failed)\s+as\s+a\s+(person|human)\b', 'insult'),
    (r'\b(remember|recall)\s+wrong\b', 'gaslighting'),
    (r'\b(you\'?re|you|u)\s+(imagining|making up)\b', 'gaslighting'),
    (r'\b(you\'?re|you|u)\s+(crazy|psycho|mental)\b', 'insult'),
    (r'\b(overreacting|being dramatic)\b', 'manipulation'),
    (r'\b(you|u)\s+(disgust|sicken|repulse)\s+me\b', 'obscene'),
    (r'\b(kill|hurt|harm)\s+(yourself|you)\b', 'severe_toxic'),
    (r'\b(don\'?t\s+talk\s+to\s+them|stay\s+away\s+from\s+them)\b', 'isolation'),
    (r'\b(you\'?re|you)\s+not\s+good\s+enough\b', 'bullying'),
    (r'\b(nobody\s+else\s+will\s+love\s+you)\b', 'manipulation')
    

]

# Abuse type definitions
ABUSE_DEFINITIONS = {
    'toxic': {'name': 'Toxic', 'color': 'toxic'},
    'severe_toxic': {'name': 'Severe Toxic', 'color': 'severe_toxic'},
    'obscene': {'name': 'Obscene', 'color': 'obscene'},
    'threat': {'name': 'Threat', 'color': 'threat'},
    'insult': {'name': 'Insult', 'color': 'insult'},
    'identity_hate': {'name': 'Identity Hate', 'color': 'identity_hate'},
    'gaslighting': {'name': 'Gaslighting', 'color': 'gaslighting'},
    'manipulation': {'name': 'Manipulation', 'color': 'manipulation'},
    'isolation': {'name': 'Isolation', 'color': 'isolation'},
    'bullying': {'name': 'Bullying', 'color': 'bullying'}
}

def analyze_conversation(text):
    """Analyze the conversation for abusive content"""
    results = {
        'categories': set(),
        'abusive_phrases': [],
        'abusive_words': set(),
        'is_abusive': False
    }
    
    if not text.strip():
        return results
    
    # Analyze with ML models
    for category, model in models.items():
        if model.predict([text])[0] == 1:
            results['categories'].add(category)
    
    # Check for patterns with context
    doc = nlp(text.lower())
    sentences = [sent.text for sent in doc.sents]
    
    for sentence in sentences:
        for pattern, category in ABUSE_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                results['categories'].add(category)
                match = re.search(pattern, sentence, re.IGNORECASE)
                abusive_part = match.group(0) if match else sentence
                
                # Extract abusive words
                abusive_words = []
                if match:
                    for group in match.groups():
                        if group:
                            abusive_words.append(group.lower())
                            results['abusive_words'].add(group.lower())
                
                results['abusive_phrases'].append({
                    'text': sentence.strip(),
                    'category': category,
                    'abusive_part': abusive_part,
                    'abusive_words': abusive_words
                })
    
    # Analyze sentiment for context
    sentiment = TextBlob(text).sentiment
    if sentiment.polarity < -0.3 and any(word in text.lower() for word in ['you', 'u', 'your']):
        results['categories'].add('toxic')
    
    results['abusive_words'] = list(results['abusive_words'])
    results['is_abusive'] = len(results['categories']) > 0 or len(results['abusive_phrases']) > 0
    
    return results

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detection')
def detection():
    return render_template('detection.html', 
                         analyzed=False,
                         user_input='',
                         is_abusive=False,
                         show_details=False)

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    
    if not user_input.strip():
        return redirect(url_for('detection'))
    
    analysis_results = analyze_conversation(user_input)
    
    if not analysis_results['is_abusive']:
        return render_template('detection.html',
                            analyzed=True,
                            user_input=user_input,
                            is_abusive=False,
                            show_details=False)
    
    categorized = []
    for category in analysis_results.get('categories', set()):
        if category in ABUSE_DEFINITIONS:
            categorized.append(ABUSE_DEFINITIONS[category])
    
    return render_template('detection.html',
                         analyzed=True,
                         user_input=user_input,
                         is_abusive=True,
                         show_details=True,
                         categories=categorized,
                         abusive_phrases=analysis_results['abusive_phrases'],
                         abusive_words=analysis_results['abusive_words'],
                         ABUSE_DEFINITIONS=ABUSE_DEFINITIONS)

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
