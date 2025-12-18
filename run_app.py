"""
Complete News Article Clustering Flask Application
Run this file to start the web server
"""

import os
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from ml_utils import OpenRouterEmbeddingGenerator, NewsClusterPredictor

# Load environment variables
load_dotenv()

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
            print(f"[OK] Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"[ERROR] Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
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


def predict_cluster(text, model_type='kmeans'):
    """Predict cluster for given text"""
    if not model_loaded or not predictor_data:
        return None
    
    try:
        # Preprocess the text
        if 'preprocessor' in predictor_data:
            processed_text = predictor_data['preprocessor'].preprocess(text)
        else:
            # Fallback if preprocessor not found
            processed_text = text.lower()
            processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
            processed_text = ' '.join(processed_text.split())
        
        # Get common components
        vectorizer = predictor_data['vectorizer']
        lsa_model = predictor_data.get('lsa_model')
        
        # Transform text to vector
        text_vector = vectorizer.transform([processed_text]).toarray()
        
        # Apply LSA if available
        if lsa_model:
            text_vector = lsa_model.transform(text_vector)
        
        # Get models data
        models_data = predictor_data.get('models', {})
        
        # Handle legacy model format
        if not models_data and 'model' in predictor_data:
             models_data = {
                 'kmeans': {
                     'model': predictor_data['model'],
                     'keywords': predictor_data.get('cluster_keywords', {})
                 }
             }
        
        if model_type not in models_data:
            model_type = 'kmeans'
            
        if model_type not in models_data:
             # If still not found (e.g. empty models_data), return error
             return {'error': 'No models available'}

        model_data = models_data[model_type]
        
        cluster = -1
        confidence = 0.0
        keywords = []

        if model_type == 'kmeans':
            model = model_data['model']
            # Predict cluster
            cluster = model.predict(text_vector)[0]
            # Calculate confidence based on distance to centroid
            distances = model.transform(text_vector)
            min_dist = distances[0][cluster]
            confidence = 1.0 / (1.0 + min_dist) # Simple heuristic
            
            keywords = model_data['keywords'].get(cluster, [])[:10]

        elif model_type == 'hierarchical':
            # For these, we use nearest centroid classification
            centroids_dict = model_data.get('centroids', {})
            
            if not centroids_dict:
                return {
                    'cluster': -1,
                    'confidence': 0.0,
                    'keywords': [],
                    'message': f"No clusters found for {model_type} (noise only or empty)"
                }

            # Convert centroids dict to array for distance calculation
            labels = list(centroids_dict.keys())
            centroids = np.array(list(centroids_dict.values()))
            
            # Calculate distances
            distances = euclidean_distances(text_vector, centroids)
            min_idx = np.argmin(distances)
            min_dist = distances[0][min_idx]
            
            cluster = labels[min_idx]
            confidence = 1.0 / (1.0 + min_dist)
            
            keywords = model_data['keywords'].get(cluster, [])[:10]

        return {
            'cluster': int(cluster),
            'confidence': float(confidence),
            'keywords': keywords,
            'processed_text': processed_text[:200] + '...' if len(processed_text) > 200 else processed_text,
            'model_used': model_type
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
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
    <title>SmartNews Analytics | ML Clustering</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --background: #0f172a;
            --surface: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --error: #ef4444;
            --border: #334155;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--background);
            color: var(--text);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Navigation Bar */
        .navbar {
            height: 60px;
            background-color: var(--surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            padding: 0 20px;
            justify-content: space-between;
            z-index: 100;
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 1.2rem;
            color: var(--text);
        }

        .navbar-brand i {
            color: var(--primary);
            font-size: 1.4rem;
        }

        .navbar-actions {
            display: flex;
            gap: 15px;
        }

        .nav-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 1.1rem;
            transition: color 0.2s;
        }

        .nav-btn:hover {
            color: var(--text);
        }

        /* Main Layout */
        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 300px;
            background-color: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease;
            z-index: 90;
        }

        .sidebar.collapsed {
            transform: translateX(-100%);
            width: 0;
            border: none;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }

        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid var(--border);
        }

        .metric-title {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--accent);
        }

        .metric-subtitle {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 5px;
        }

        .btn-viz {
            flex: 1;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
        }

        .btn-viz:hover {
            background: var(--border);
            color: var(--text);
        }

        .btn-viz.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .plot-container {
            margin-top: 10px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: transform 0.2s;
        }

        .plot-container:hover {
            border-color: var(--accent);
        }

        .plot-container img {
            width: 100%;
            height: auto;
            display: block;
        }

        /* Content Area */
        .content {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            position: relative;
        }

        .toggle-sidebar-btn {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 50;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }

        .toggle-sidebar-btn:hover {
            background: var(--border);
        }

        .app-card {
            max-width: 800px;
            margin: 0 auto;
            background: var(--surface);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }

        .card-header {
            padding: 25px;
            border-bottom: 1px solid var(--border);
            background: linear-gradient(to right, rgba(67, 97, 238, 0.1), transparent);
        }

        .card-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .card-header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .card-body {
            padding: 25px;
        }

        /* Form Elements */
        .form-group {
            margin-bottom: 20px;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .form-select, .form-textarea {
            width: 100%;
            background: var(--background);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            color: var(--text);
            font-family: inherit;
            font-size: 0.95rem;
            transition: border-color 0.2s;
        }

        .form-select:focus, .form-textarea:focus {
            outline: none;
            border-color: var(--primary);
        }

        .form-textarea {
            min-height: 150px;
            resize: vertical;
        }

        .btn {
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: var(--secondary);
        }

        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }

        .btn-outline:hover {
            background: var(--border);
        }

        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        /* Results */
        .result-section {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid var(--border);
            display: none;
        }

        .result-section.show {
            display: block;
            animation: slideDown 0.4s ease;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            margin-bottom: 15px;
        }

        .keywords-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .keyword-chip {
            background: rgba(67, 97, 238, 0.1);
            color: var(--accent);
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            border: 1px solid rgba(67, 97, 238, 0.2);
        }

        /* Sample Pills */
        .sample-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .sample-pill {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-pill:hover {
            background: var(--border);
            color: var(--text);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--background);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        /* Dendrogram Modal */
        .dendrogram-modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(5px);
            overflow: auto;
            align-items: center;
            justify-content: center;
        }

        .dendrogram-modal.show {
            display: flex;
            animation: fadeIn 0.3s ease;
        }

        .dendrogram-content {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            position: relative;
            padding: 20px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }

        .dendrogram-content img {
            max-width: 100%;
            height: auto;
            display: block;
            border-radius: 8px;
        }

        .close-dendrogram {
            position: absolute;
            top: 15px;
            right: 20px;
            color: var(--text-muted);
            font-size: 2rem;
            cursor: pointer;
            z-index: 10;
            transition: color 0.2s;
        }

        .close-dendrogram:hover {
            color: var(--text);
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar">
        <div class="navbar-brand">
            <i class="fas fa-layer-group"></i>
            <span>SmartNews Analytics</span>
        </div>
        <div class="navbar-actions">
            <button class="nav-btn" title="Settings"><i class="fas fa-cog"></i></button>
            <button class="nav-btn" title="Help"><i class="fas fa-question-circle"></i></button>
            <a href="https://github.com/sultanofficial717/SmartNews-Analytics-ML-Based-Article-Clustering.git" target="_blank" class="nav-btn" title="GitHub"><i class="fab fa-github"></i></a>
        </div>
    </nav>

    <div class="main-container">
        <!-- Sidebar -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <i class="fas fa-chart-line"></i> Model Performance
            </div>
            <div class="sidebar-content">
                {% if scores %}
                    {% for model_name, model_scores in scores.items() %}
                    <div class="metric-card">
                        <div class="metric-title">{{ model_name|upper }} Model</div>
                        {% if model_scores.silhouette %}
                        <div class="metric-value">{{ "%.3f"|format(model_scores.silhouette) }}</div>
                        <div class="metric-subtitle">Silhouette Score</div>
                        {% endif %}
                        {% if model_scores.davies_bouldin %}
                        <div style="margin-top: 10px;">
                            <div class="metric-value" style="color: #f59e0b;">{{ "%.3f"|format(model_scores.davies_bouldin) }}</div>
                            <div class="metric-subtitle">Davies-Bouldin Score</div>
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="metric-card">
                        <div class="metric-subtitle">No performance data available. Please train the model.</div>
                    </div>
                {% endif %}

                <div class="sidebar-header" style="margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--border);">
                    <i class="fas fa-project-diagram"></i> Cluster Visualization
                </div>
                <div class="form-group">
                    <label class="form-label">Feature Representation</label>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn-viz active" id="btn-tfidf" onclick="selectRep('lsa_tfidf')">LSA+TF-IDF</button>
                        {% if has_embedding %}
                        <button class="btn-viz" id="btn-emb" onclick="selectRep('embedding')">Embedding</button>
                        {% else %}
                        <button class="btn-viz" id="btn-emb" disabled style="opacity: 0.5; cursor: not-allowed;" title="Embedding model not available">Embedding</button>
                        {% endif %}
                    </div>
                </div>
                <div class="viz-controls" style="display: flex; gap: 5px; margin-bottom: 10px; flex-wrap: wrap;">
                    <button class="btn-viz active" onclick="changePlot('kmeans')" title="K-Means">KM</button>
                    <button class="btn-viz" onclick="changePlot('hierarchical')" title="Hierarchical">HC</button>
                    <button class="btn-viz" onclick="changePlot('dendrogram')" title="Dendrogram">Tree</button>
                </div>
                <div class="plot-container">
                    <img id="viz-image" src="{{ url_for('static', filename='kmeans_plot_tfidf.png') }}" alt="Cluster Plot" onclick="openModal(this.src)">
                </div>
                <div style="text-align: center; font-size: 0.75rem; color: var(--text-muted); margin-top: 5px;">
                    Click image to enlarge
                </div>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="content">
            <button class="toggle-sidebar-btn" id="toggleSidebar">
                <i class="fas fa-bars"></i>
            </button>

            <div class="app-card">
                <div class="card-header">
                    <h2>Article Classifier</h2>
                    <p>Paste a news article below to automatically categorize it using our ML models.</p>
                </div>
                <div class="card-body">
                    <div class="form-group">
                        <label class="form-label">Select Model</label>
                        <select id="model-select" class="form-select">
                            <option value="kmeans">K-Means Clustering (Recommended)</option>
                            <option value="hierarchical">Hierarchical Clustering</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Article Text</label>
                        <textarea id="article-input" class="form-textarea" placeholder="Paste the full text of the news article here..."></textarea>
                        <div class="sample-pills">
                            <span class="sample-pill" data-type="tech">Technology</span>
                            <span class="sample-pill" data-type="sports">Sports</span>
                            <span class="sample-pill" data-type="politics">Politics</span>
                            <span class="sample-pill" data-type="medical">Medical</span>
                        </div>
                    </div>

                    <div class="btn-group">
                        <button id="predict-btn" class="btn btn-primary">
                            <i class="fas fa-magic"></i> Analyze Article
                        </button>
                        <button id="clear-btn" class="btn btn-outline">
                            <i class="fas fa-eraser"></i> Clear
                        </button>
                    </div>

                    <!-- Results -->
                    <div id="result-section" class="result-section">
                        <div class="result-badge">
                            <i class="fas fa-check-circle"></i>
                            <span id="cluster-result">Cluster 0</span>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Confidence Score</label>
                            <div style="background: var(--background); height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="confidence-bar" style="width: 0%; height: 100%; background: var(--success); transition: width 1s ease;"></div>
                            </div>
                            <div id="confidence-text" style="text-align: right; font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">0%</div>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Top Keywords</label>
                            <div id="keywords-container" class="keywords-container">
                                <!-- Keywords will be injected here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="dendrogram-modal">
        <div class="dendrogram-content">
            <span class="close-dendrogram" onclick="closeImageModal()">&times;</span>
            <h3 id="modalTitle" style="margin-bottom: 20px; color: var(--primary); text-align: center;">Visualization</h3>
            <img id="modalImage" src="" alt="Visualization">
        </div>
    </div>

    <script>
        // Representation selection
        let currentRep = 'lsa_tfidf';
        function selectRep(rep) {
            currentRep = rep;
            document.getElementById('btn-tfidf').classList.toggle('active', rep === 'lsa_tfidf');
            document.getElementById('btn-emb').classList.toggle('active', rep === 'embedding');
            // Reset plot to kmeans for new rep
            changePlot('kmeans');
        }

        // Visualization Logic
        function changePlot(type) {
            const img = document.getElementById('viz-image');
            const plotMap = {
                'lsa_tfidf': {
                    'kmeans': 'kmeans_plot_tfidf.png',
                    'hierarchical': 'hierarchical_plot_tfidf.png',
                    'dendrogram': 'dendrogram_tfidf.png'
                },
                'embedding': {
                    'kmeans': 'kmeans_plot_emb.png',
                    'hierarchical': 'hierarchical_plot_emb.png',
                    'dendrogram': 'dendrogram_emb.png'
                }
            };
            let filename = plotMap[currentRep][type];
            let title = '';
            switch(type) {
                case 'kmeans':
                    title = (currentRep === 'embedding') ? 'K-Means Clustering (Embedding)' : 'K-Means Clustering (LSA+TF-IDF)';
                    break;
                case 'hierarchical':
                    title = (currentRep === 'embedding') ? 'Hierarchical Clustering (Embedding)' : 'Hierarchical Clustering (LSA+TF-IDF)';
                    break;
                case 'dendrogram':
                    title = (currentRep === 'embedding') ? 'Hierarchical Dendrogram (Embedding)' : 'Hierarchical Dendrogram (LSA+TF-IDF)';
                    break;
            }
            img.src = "{{ url_for('static', filename='') }}" + filename;
            img.dataset.title = title;
        }

        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const modalTitle = document.getElementById('modalTitle');
            const currentImg = document.getElementById('viz-image');
            
            modalImg.src = currentImg.src;
            modalTitle.textContent = currentImg.dataset.title || 'Visualization';
            modal.classList.add('show');
        }

        function closeImageModal() {
            document.getElementById('imageModal').classList.remove('show');
        }

        // Close on outside click
        document.getElementById('imageModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('imageModal')) {
                closeImageModal();
            }
        });

        // Sidebar Toggle
        const sidebar = document.getElementById('sidebar');
        const toggleBtn = document.getElementById('toggleSidebar');
        
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });

        // Sample Texts
        const samples = {
            tech: "The new quantum processor achieves breakthrough speeds in computing tasks. Researchers demonstrated quantum supremacy by solving complex problems in seconds.",
            sports: "The championship game ended in a dramatic overtime victory. The team's defense was impenetrable, leading them to lift the trophy after a grueling season.",
            politics: "The senate passed the new bill with a majority vote today. Lawmakers debated for hours regarding the implications of the new tax reform legislation.",
            medical: "Clinical trials show promising results for the new vaccine. Doctors are optimistic about the treatment's efficacy in preventing the spread of the virus."
        };

        document.querySelectorAll('.sample-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                document.getElementById('article-input').value = samples[pill.dataset.type];
            });
        });

        // Clear
        document.getElementById('clear-btn').addEventListener('click', () => {
            document.getElementById('article-input').value = '';
            document.getElementById('result-section').classList.remove('show');
        });

        // Predict
        document.getElementById('predict-btn').addEventListener('click', async () => {
            const text = document.getElementById('article-input').value;
            const model = document.getElementById('model-select').value;
            const btn = document.getElementById('predict-btn');

            if (text.length < 10) {
                alert('Please enter at least 10 characters.');
                return;
            }

            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text: text, model_type: model })
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                } else {
                    // Update UI
                    document.getElementById('cluster-result').textContent = `Cluster ${data.cluster}`;
                    
                    const confidence = (data.confidence * 100).toFixed(1);
                    document.getElementById('confidence-bar').style.width = `${confidence}%`;
                    document.getElementById('confidence-text').textContent = `${confidence}%`;

                    const keywordsContainer = document.getElementById('keywords-container');
                    keywordsContainer.innerHTML = '';
                    data.keywords.forEach(kw => {
                        const chip = document.createElement('span');
                        chip.className = 'keyword-chip';
                        chip.textContent = kw;
                        keywordsContainer.appendChild(chip);
                    });

                    document.getElementById('result-section').classList.add('show');
                }
            } catch (e) {
                console.error(e);
                alert('An error occurred.');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-magic"></i> Analyze Article';
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    """Home page"""
    scores = {}
    has_embedding = False
    if predictor_data and 'models' in predictor_data:
        for model_name, model_info in predictor_data['models'].items():
            if model_name == 'embedding':
                has_embedding = True
            if 'kmeans' in model_info and 'scores' in model_info['kmeans']:
                scores[model_name] = model_info['kmeans']['scores']
    
    return render_template_string(HTML_TEMPLATE, model_loaded=model_loaded, scores=scores, has_embedding=has_embedding)


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
        model_type = data.get('model_type', 'kmeans')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text is too short (minimum 10 characters)'}), 400
        
        # Get prediction
        result = predict_cluster(text, model_type)
        
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
        print("\n[OK] Model loaded successfully!")
        print("[OK] Application is ready to accept predictions")
    else:
        print("\n[WARN] Model not found. Please train the model first.")
        print("  Run: python train.py")
    
    print("\n" + "-"*60)
    print("  Starting Flask server...")
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("-"*60 + "\n")
    
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
