"""Flask application for News Article Clustering"""

import os
import pickle
import json
from flask import Flask, render_template, request, jsonify
from ml_utils import TextPreprocessor, NewsClusterPredictor
import pandas as pd
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'news_clustering_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, '..', 'uploads')

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global predictor object
predictor = None
model_loaded = False


def load_model():
    """Load the trained model"""
    global predictor, model_loaded
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            preprocessor = model_data['preprocessor']
            cluster_keywords = model_data['cluster_keywords']
            
            predictor = NewsClusterPredictor(model, vectorizer, preprocessor, cluster_keywords)
            model_loaded = True
            print(f"✓ Model loaded from {MODEL_PATH}")
            return True
        else:
            print(f"✗ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/api/status')
def status():
    """API endpoint to check model status"""
    return jsonify({
        'model_loaded': model_loaded,
        'message': 'Model is ready!' if model_loaded else 'Model not loaded. Please train the model first.'
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
        result = predictor.predict_with_details(text)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """API endpoint for batch prediction"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid texts format'}), 400
        
        # Filter empty texts
        texts = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= 10]
        
        if not texts:
            return jsonify({'error': 'No valid texts provided'}), 400
        
        # Get predictions
        results = predictor.predict_batch(texts)
        
        return jsonify({'predictions': results, 'count': len(results)}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if not model_loaded or not predictor:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        info = {
            'n_clusters': len(predictor.cluster_keywords),
            'clusters': {}
        }
        
        for cluster_id, keywords in predictor.cluster_keywords.items():
            info['clusters'][str(cluster_id)] = {
                'id': cluster_id,
                'top_keywords': keywords[:5]
            }
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-predict', methods=['GET'])
def sample_predict():
    """Get sample predictions"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        samples = [
            "The stock market showed strong gains today as technology companies reported better-than-expected earnings.",
            "Scientists have discovered a new species of deep-sea fish in the Mariana Trench with unique adaptations.",
            "The championship game went into overtime with an exciting finish and incredible plays.",
            "New healthcare policies were announced during the press conference addressing patient care.",
            "Machine learning algorithms are revolutionizing data analysis and artificial intelligence applications."
        ]
        
        results = predictor.predict_batch(samples)
        
        return jsonify({'predictions': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
