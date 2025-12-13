"""
Complete News Article Clustering Flask Application
Run this file to start the web server
"""

import os
import pickle
import re
from flask import Flask, render_template_string, request, jsonify

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'news_clustering_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Global variables
predictor_data = None
model_loaded = False


def load_model():
    """Load the trained model"""
    global predictor_data, model_loaded
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                predictor_data = pickle.load(f)
            model_loaded = True
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"✗ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def predict_cluster(text):
    """Predict cluster for given text"""
    if not model_loaded or not predictor_data:
        return None
    
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Get model components
        model = predictor_data['model']
        vectorizer = predictor_data['vectorizer']
        cluster_keywords = predictor_data.get('cluster_keywords', {})
        
        # Transform text using the vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Predict cluster
        cluster = model.predict(text_vector)[0]
        
        # Get keywords for this cluster
        keywords = cluster_keywords.get(cluster, [])[:10]
        
        return {
            'cluster': int(cluster),
            'keywords': keywords,
            'processed_text': processed_text[:200] + '...' if len(processed_text) > 200 else processed_text
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


# HTML Template with all functionality embedded
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Article Clustering - ML Project</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.1));
            border-radius: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .header p {
            color: #a0a0a0;
            font-size: 1.1rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.9rem;
            margin-top: 20px;
        }

        .status-badge.ready {
            background: linear-gradient(135deg, #00b09b, #96c93d);
            color: white;
        }

        .status-badge.error {
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color: white;
        }

        /* Main Card */
        .main-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 1.4rem;
            color: #4facfe;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Textarea */
        .input-group {
            margin-bottom: 20px;
        }

        .input-group label {
            display: block;
            margin-bottom: 10px;
            color: #ccc;
            font-size: 0.95rem;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            font-size: 1rem;
            resize: vertical;
            min-height: 150px;
            transition: border-color 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: #4facfe;
        }

        textarea::placeholder {
            color: #666;
        }

        .char-count {
            text-align: right;
            font-size: 0.85rem;
            color: #666;
            margin-top: 5px;
        }

        /* Buttons */
        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 12px 25px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s;
            font-weight: 500;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-primary {
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: #000;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .btn-secondary:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.2);
        }

        .btn-tertiary {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: #fff;
        }

        .btn-tertiary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(240, 147, 251, 0.4);
        }

        /* Result Display */
        .result-container {
            margin-top: 30px;
            display: none;
        }

        .result-container.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.1));
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(79, 172, 254, 0.3);
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .result-header h3 {
            color: #4facfe;
            font-size: 1.2rem;
        }

        .close-btn {
            background: none;
            border: none;
            color: #999;
            cursor: pointer;
            font-size: 1.2rem;
            transition: color 0.3s;
        }

        .close-btn:hover {
            color: #fff;
        }

        .cluster-display {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 25px;
        }

        .cluster-badge {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: bold;
            color: #000;
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
        }

        .cluster-info h4 {
            color: #fff;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }

        .cluster-info p {
            color: #a0a0a0;
        }

        .keywords-section {
            margin-top: 20px;
        }

        .keywords-section h4 {
            color: #ccc;
            font-size: 0.95rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .keywords-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .keyword-tag {
            background: rgba(79, 172, 254, 0.2);
            color: #4facfe;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            border: 1px solid rgba(79, 172, 254, 0.3);
        }

        .preview-section {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }

        .preview-section h4 {
            color: #ccc;
            font-size: 0.9rem;
            margin-bottom: 10px;
        }

        .preview-section p {
            color: #999;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Toast Notification */
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }

        .toast.success {
            background: linear-gradient(135deg, #00b09b, #96c93d);
        }

        .toast.error {
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
        }

        /* Loading Spinner */
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(79, 172, 254, 0.2);
            border-top-color: #4facfe;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Sample Texts */
        .sample-section {
            margin-top: 20px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
        }

        .sample-section h4 {
            color: #ccc;
            margin-bottom: 15px;
            font-size: 0.95rem;
        }

        .sample-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .sample-btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.05);
            color: #ccc;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.3s;
        }

        .sample-btn:hover {
            background: rgba(79, 172, 254, 0.2);
            border-color: #4facfe;
            color: #4facfe;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9rem;
        }

        .footer a {
            color: #4facfe;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1><i class="fas fa-brain"></i> News Article Clustering</h1>
            <p>Intelligent text classification using K-Means Machine Learning</p>
            <div id="status-badge" class="status-badge {{ 'ready' if model_loaded else 'error' }}">
                {% if model_loaded %}
                <i class="fas fa-check-circle"></i> Model Ready
                {% else %}
                <i class="fas fa-exclamation-circle"></i> Model Not Loaded
                {% endif %}
            </div>
        </div>

        <!-- Main Prediction Card -->
        <div class="main-card">
            <h2 class="section-title"><i class="fas fa-magic"></i> Predict Article Cluster</h2>
            
            <div class="input-group">
                <label for="article-input">Enter News Article Text</label>
                <textarea 
                    id="article-input" 
                    placeholder="Paste your news article here... (minimum 10 characters)"
                    rows="6"></textarea>
                <div class="char-count"><span id="char-counter">0</span> / 5000 characters</div>
            </div>

            <div class="button-group">
                <button id="predict-btn" class="btn btn-primary" {{ 'disabled' if not model_loaded }}>
                    <i class="fas fa-paper-plane"></i> Predict Cluster
                </button>
                <button id="clear-btn" class="btn btn-secondary">
                    <i class="fas fa-times"></i> Clear
                </button>
            </div>

            <!-- Sample Texts -->
            <div class="sample-section">
                <h4><i class="fas fa-star"></i> Try Sample Texts</h4>
                <div class="sample-buttons">
                    <button class="sample-btn" data-sample="tech">Technology</button>
                    <button class="sample-btn" data-sample="sports">Sports</button>
                    <button class="sample-btn" data-sample="science">Science</button>
                    <button class="sample-btn" data-sample="politics">Politics</button>
                    <button class="sample-btn" data-sample="space">Space</button>
                </div>
            </div>

            <!-- Loading Spinner -->
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analyzing text...</p>
            </div>

            <!-- Result Display -->
            <div id="result-container" class="result-container">
                <div class="result-card">
                    <div class="result-header">
                        <h3><i class="fas fa-check-circle"></i> Prediction Result</h3>
                        <button id="close-result" class="close-btn"><i class="fas fa-times"></i></button>
                    </div>
                    
                    <div class="cluster-display">
                        <div class="cluster-badge">
                            <span id="cluster-number">0</span>
                        </div>
                        <div class="cluster-info">
                            <h4>Predicted Cluster</h4>
                            <p id="cluster-label">Category identified</p>
                        </div>
                    </div>

                    <div class="keywords-section">
                        <h4><i class="fas fa-tags"></i> Cluster Keywords</h4>
                        <div id="keywords-list" class="keywords-list"></div>
                    </div>

                    <div class="preview-section">
                        <h4><i class="fas fa-eye"></i> Processed Text Preview</h4>
                        <p id="text-preview"></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>News Article Clustering ML Project | Built with Flask & scikit-learn</p>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast" class="toast"></div>

    <script>
        // Sample texts
        const sampleTexts = {
            tech: "The new graphics card features advanced ray tracing technology with improved performance. Computer hardware companies are pushing the boundaries of processing power and memory bandwidth. Software developers are optimizing their applications to take advantage of these technological improvements.",
            sports: "The hockey team won the championship in an exciting overtime match. Baseball players are preparing for the upcoming season with intensive training camps. The motorsports racing event attracted thousands of fans to watch their favorite auto racers compete for the trophy.",
            science: "Scientists have discovered a new treatment that could revolutionize medicine and healthcare. The medical research team published their findings in a peer-reviewed journal. Electronic devices are becoming more sophisticated with miniaturized circuits and improved battery technology.",
            politics: "The government announced new policies regarding gun control and firearm regulations. Political discussions continue about security measures and law enforcement. Representatives from different parties debated the proposed legislation in congress.",
            space: "NASA launched a new spacecraft to explore distant planets and gather scientific data. The space program continues to advance our understanding of the universe. Astronomers discovered new celestial objects using advanced telescope technology."
        };

        // DOM Elements
        const articleInput = document.getElementById('article-input');
        const charCounter = document.getElementById('char-counter');
        const predictBtn = document.getElementById('predict-btn');
        const clearBtn = document.getElementById('clear-btn');
        const resultContainer = document.getElementById('result-container');
        const closeResult = document.getElementById('close-result');
        const loading = document.getElementById('loading');
        const toast = document.getElementById('toast');

        // Character counter
        articleInput.addEventListener('input', () => {
            const count = articleInput.value.length;
            charCounter.textContent = count;
            predictBtn.disabled = count < 10 || !{{ 'true' if model_loaded else 'false' }};
        });

        // Clear button
        clearBtn.addEventListener('click', () => {
            articleInput.value = '';
            charCounter.textContent = '0';
            resultContainer.classList.remove('show');
            predictBtn.disabled = true;
        });

        // Close result
        closeResult.addEventListener('click', () => {
            resultContainer.classList.remove('show');
        });

        // Sample text buttons
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const sampleType = btn.dataset.sample;
                articleInput.value = sampleTexts[sampleType];
                charCounter.textContent = articleInput.value.length;
                predictBtn.disabled = false;
            });
        });

        // Show toast notification
        function showToast(message, type = 'success') {
            toast.textContent = message;
            toast.className = `toast ${type} show`;
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        // Predict button
        predictBtn.addEventListener('click', async () => {
            const text = articleInput.value.trim();
            
            if (text.length < 10) {
                showToast('Text is too short (minimum 10 characters)', 'error');
                return;
            }

            // Show loading
            loading.classList.add('show');
            resultContainer.classList.remove('show');
            predictBtn.disabled = true;

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                if (response.ok && data.cluster !== undefined) {
                    // Display result
                    document.getElementById('cluster-number').textContent = data.cluster;
                    document.getElementById('cluster-label').textContent = `Cluster ${data.cluster} - Topic Category`;
                    document.getElementById('text-preview').textContent = data.processed_text || text.substring(0, 200) + '...';
                    
                    // Display keywords
                    const keywordsList = document.getElementById('keywords-list');
                    keywordsList.innerHTML = '';
                    
                    if (data.keywords && data.keywords.length > 0) {
                        data.keywords.forEach(keyword => {
                            const tag = document.createElement('span');
                            tag.className = 'keyword-tag';
                            tag.textContent = keyword;
                            keywordsList.appendChild(tag);
                        });
                    } else {
                        keywordsList.innerHTML = '<span class="keyword-tag">No keywords available</span>';
                    }

                    resultContainer.classList.add('show');
                    showToast('Prediction successful!', 'success');
                } else {
                    showToast(data.error || 'Prediction failed', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showToast('Connection error. Please try again.', 'error');
            } finally {
                loading.classList.remove('show');
                predictBtn.disabled = articleInput.value.length < 10;
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    """Home page"""
    return render_template_string(HTML_TEMPLATE, model_loaded=model_loaded)


@app.route('/api/status')
def status():
    """API endpoint to check model status"""
    return jsonify({
        'model_loaded': model_loaded,
        'message': 'Model is ready!' if model_loaded else 'Model not loaded.'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text is too short (minimum 10 characters)'}), 400
        
        # Get prediction
        result = predict_cluster(text)
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if not model_loaded or not predictor_data:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        cluster_keywords = predictor_data.get('cluster_keywords', {})
        n_clusters = len(cluster_keywords) if cluster_keywords else 5
        
        return jsonify({
            'n_clusters': n_clusters,
            'clusters': {str(k): v[:5] for k, v in cluster_keywords.items()}
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  NEWS ARTICLE CLUSTERING - ML PROJECT")
    print("="*60)
    
    # Load model on startup
    load_model()
    
    if model_loaded:
        print("\n✓ Model loaded successfully!")
        print("✓ Application is ready to accept predictions")
    else:
        print("\n⚠ Model not found. Please train the model first.")
        print("  Run: python train.py")
    
    print("\n" + "-"*60)
    print("  Starting Flask server...")
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("-"*60 + "\n")
    
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
