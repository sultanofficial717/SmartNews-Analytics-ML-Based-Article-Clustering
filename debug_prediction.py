import sys
import os
import pickle
import numpy as np
from run_app import preprocess_text

# Add current directory to path
sys.path.append(os.getcwd())

MODEL_PATH = os.path.join('models', 'news_clustering_model.pkl')

def debug_prediction(text):
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        predictor_data = pickle.load(f)

    vectorizer = predictor_data['vectorizer']
    models_data = predictor_data['models']
    kmeans_data = models_data['kmeans']
    model = kmeans_data['model']
    keywords = kmeans_data['keywords']

    print(f"\nInput Text: '{text}'")
    processed_text = preprocess_text(text)
    print(f"Processed Text: '{processed_text}'")

    text_vector = vectorizer.transform([processed_text]).toarray()
    
    # Predict
    cluster = model.predict(text_vector)[0]
    print(f"\nPredicted Cluster: {cluster}")
    print(f"Cluster Keywords: {keywords.get(cluster, [])[:5]}")

    # Distances to centroids
    distances = model.transform(text_vector)[0]
    print("\nDistances to Centroids:")
    for i, dist in enumerate(distances):
        kw = keywords.get(i, [])[:3]
        print(f"Cluster {i} ({kw}): {dist:.4f}")

if __name__ == "__main__":
    # Ambiguous sports text
    text1 = "The player suffered a serious injury during the match and was taken to the hospital."
    print("--- Test 1 ---")
    debug_prediction(text1)

    text2 = "The team lost the game terribly. It was a disaster for the coach."
    print("\n--- Test 2 ---")
    debug_prediction(text2)

