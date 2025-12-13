"""Predictor module for clustering predictions"""

import numpy as np


class NewsClusterPredictor:
    """Class to predict cluster for new news articles"""

    def __init__(self, model, vectorizer, preprocessor, cluster_keywords):
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
        self.cluster_keywords = cluster_keywords

    def predict_single(self, text):
        """Predict cluster for a single news article"""
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)

        # Convert to TF-IDF vector
        X_new = self.vectorizer.transform([processed_text]).toarray()

        # Predict cluster
        cluster = self.model.predict(X_new)[0]

        return cluster

    def predict_with_details(self, text):
        """Predict cluster and return detailed information"""
        cluster = self.predict_single(text)

        # Get cluster keywords
        keywords = self.cluster_keywords.get(cluster, [])

        # Calculate confidence (distance to cluster center for K-Means)
        processed_text = self.preprocessor.preprocess(text)
        X_new = self.vectorizer.transform([processed_text]).toarray()

        confidence = None
        if hasattr(self.model, 'cluster_centers_'):
            # For K-Means, calculate distance to cluster center
            distances = np.linalg.norm(self.model.cluster_centers_ - X_new, axis=1)
            confidence = 1 / (1 + distances[cluster])  # Convert distance to confidence score

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
