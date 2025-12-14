"""Predictor module for clustering predictions"""

import numpy as np
from .embeddings import OpenRouterEmbeddingGenerator

class NewsClusterPredictor:
    """Class to predict cluster for new news articles"""

    def __init__(self, model, vectorizer, preprocessor, cluster_keywords, lsa_model=None, embedding_config=None, centroids=None):
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
        self.cluster_keywords = cluster_keywords
        self.lsa_model = lsa_model
        self.embedding_config = embedding_config
        self.embedding_generator = None
        self.centroids = centroids
        
        if self.embedding_config:
            try:
                self.embedding_generator = OpenRouterEmbeddingGenerator(
                    api_key=self.embedding_config['api_key'],
                    model=self.embedding_config['model']
                )
            except Exception as e:
                print(f"Error initializing embedding generator: {e}")

    def _get_vector(self, text):
        """Internal method to get vector representation"""
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Try embeddings first if configured
        if self.embedding_generator:
            emb = self.embedding_generator.get_embedding(processed_text)
            if emb:
                return np.array([emb])
            print("Warning: Embedding generation failed, falling back to TF-IDF")

        # TF-IDF path (fallback or primary)
        if self.vectorizer:
            X_new = self.vectorizer.transform([processed_text]).toarray()
            
            if self.lsa_model:
                X_new = self.lsa_model.transform(X_new)
            
            return X_new
            
        return None

    def predict_single(self, text):
        """Predict cluster for a single news article"""
        X_new = self._get_vector(text)
        
        if X_new is None:
            return -1

        # Predict cluster
        if hasattr(self.model, 'predict'):
            cluster = self.model.predict(X_new)[0]
        elif self.centroids is not None:
            # Nearest centroid classification for models without predict()
            # centroids should be a dict {label: centroid} or array
            if isinstance(self.centroids, dict):
                labels = list(self.centroids.keys())
                centers = np.array(list(self.centroids.values()))
                distances = np.linalg.norm(centers - X_new, axis=1)
                cluster = labels[np.argmin(distances)]
            else:
                distances = np.linalg.norm(self.centroids - X_new, axis=1)
                cluster = np.argmin(distances)
        else:
            cluster = -1

        return cluster

    def predict_with_details(self, text):
        """Predict cluster and return detailed information"""
        cluster = self.predict_single(text)
        
        if cluster == -1:
            return {
                'cluster': -1,
                'keywords': [],
                'confidence': 0.0,
                'preview': text[:200]
            }

        # Get cluster keywords
        keywords = self.cluster_keywords.get(cluster, [])

        # Calculate confidence
        X_new = self._get_vector(text)
        confidence = None
        
        if X_new is not None:
            if hasattr(self.model, 'cluster_centers_'):
                # For K-Means
                distances = np.linalg.norm(self.model.cluster_centers_ - X_new, axis=1)
                confidence = 1 / (1 + distances[cluster])
            elif self.centroids is not None:
                # For others
                if isinstance(self.centroids, dict):
                    centers = np.array(list(self.centroids.values()))
                    distances = np.linalg.norm(centers - X_new, axis=1)
                    confidence = 1 / (1 + np.min(distances))
                else:
                    distances = np.linalg.norm(self.centroids - X_new, axis=1)
                    confidence = 1 / (1 + distances[cluster])

        result = {
            'cluster': int(cluster),
            'keywords': [str(k) for k in keywords],
            'confidence': float(confidence) if confidence else None,
            'preview': text[:200] + '...' if len(text) > 200 else text
        }

        return result

    def predict_batch(self, texts):
        """Predict clusters for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_with_details(text)
            results.append(result)
        return results
