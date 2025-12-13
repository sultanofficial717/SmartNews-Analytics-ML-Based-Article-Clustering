"""Simple News Clustering Web App"""
import os
import pickle
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'news_clustering_model.pkl')
model_data = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    print("Model loaded!")
except:
    print("Model not found")

def predict(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = ' '.join(text.split())
    vec = model_data['vectorizer'].transform([text])
    cluster = int(model_data['model'].predict(vec)[0])
    keywords = model_data.get('cluster_keywords', {}).get(cluster, [])[:8]
    return cluster, keywords

@app.route('/')
def home():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>News Clustering</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; padding: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #4facfe; text-align: center; }
        textarea { width: 100%; height: 200px; padding: 15px; font-size: 16px; 
                   background: #16213e; color: white; border: 2px solid #4facfe; 
                   border-radius: 10px; margin: 20px 0; }
        button { background: linear-gradient(135deg, #4facfe, #00f2fe); color: black; 
                 padding: 15px 40px; font-size: 18px; border: none; border-radius: 10px; 
                 cursor: pointer; display: block; margin: 20px auto; }
        button:hover { transform: scale(1.05); }
        #result { background: #16213e; padding: 30px; border-radius: 15px; 
                  margin-top: 30px; display: none; text-align: center; }
        .cluster-num { font-size: 60px; color: #4facfe; font-weight: bold; }
        .keywords { margin-top: 20px; }
        .keyword { background: #4facfe33; color: #4facfe; padding: 8px 16px; 
                   border-radius: 20px; display: inline-block; margin: 5px; }
        .samples { background: #0f3460; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .sample-btn { background: #16213e; color: #4facfe; border: 1px solid #4facfe; 
                      padding: 10px 20px; margin: 5px; border-radius: 8px; cursor: pointer; }
        .sample-btn:hover { background: #4facfe33; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 News Article Clustering</h1>
        <p style="text-align:center; color:#888;">Enter text to predict its cluster category</p>
        
        <div class="samples">
            <p style="color:#888; margin-bottom:10px;">📝 Try sample texts:</p>
            <button class="sample-btn" onclick="loadSample('tech')">Technology</button>
            <button class="sample-btn" onclick="loadSample('sports')">Sports</button>
            <button class="sample-btn" onclick="loadSample('science')">Science</button>
            <button class="sample-btn" onclick="loadSample('space')">Space</button>
        </div>
        
        <textarea id="text" placeholder="Paste your news article text here..."></textarea>
        
        <button onclick="predict()">🔮 Predict Cluster</button>
        
        <div id="result">
            <p>Predicted Cluster:</p>
            <div class="cluster-num" id="cluster">0</div>
            <div class="keywords" id="keywords"></div>
        </div>
    </div>
    
    <script>
        const samples = {
            tech: "The new graphics card features advanced ray tracing technology. Computer hardware companies are pushing processing power limits. Software developers optimize applications for better performance.",
            sports: "The hockey team won the championship in overtime. Baseball players prepare for the season with training camps. The motorsports event attracted thousands of racing fans.",
            science: "Scientists discovered a treatment that could revolutionize medicine. The research team published findings in a journal. Electronic devices use miniaturized circuits.",
            space: "NASA launched a spacecraft to explore distant planets. The space program advances our understanding of the universe. Astronomers discovered new celestial objects."
        };
        
        function loadSample(type) {
            document.getElementById('text').value = samples[type];
        }
        
        async function predict() {
            const text = document.getElementById('text').value;
            if (text.length < 10) {
                alert('Please enter at least 10 characters');
                return;
            }
            
            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                const data = await res.json();
                
                if (data.cluster !== undefined) {
                    document.getElementById('cluster').textContent = data.cluster;
                    document.getElementById('keywords').innerHTML = 
                        data.keywords.map(k => '<span class="keyword">' + k + '</span>').join('');
                    document.getElementById('result').style.display = 'block';
                } else {
                    alert(data.error || 'Prediction failed');
                }
            } catch(e) {
                alert('Error: ' + e.message);
            }
        }
    </script>
</body>
</html>'''

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model_data:
        return jsonify({'error': 'Model not loaded'}), 400
    try:
        text = request.json.get('text', '')
        if len(text) < 10:
            return jsonify({'error': 'Text too short'}), 400
        cluster, keywords = predict(text)
        return jsonify({'cluster': cluster, 'keywords': keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Open browser: http://127.0.0.1:5001")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
