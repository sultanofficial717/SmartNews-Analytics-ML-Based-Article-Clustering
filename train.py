"""
Model training script for news clustering
Run this to train the model before starting the Flask app
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_utils import (
    TextPreprocessor,
    extract_tfidf_features,
    apply_kmeans,
    apply_hierarchical,
    apply_dbscan,
    extract_top_keywords,
    compute_centroids
)

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'news_clustering_model.pkl'

# API Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("⚠ Warning: OPENROUTER_API_KEY not found in environment variables.")
    print("  Embeddings will not be generated (falling back to TF-IDF).")
    
EMBEDDING_MODEL = "mistralai/devstral-2512"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def load_news_data(csv_path='data/list.csv', text_folder='data/*.txt'):
    """Load news articles from CSV and text files"""
    
    # Check if csv_path exists, if not check if it exists in root
    if not os.path.exists(csv_path) and os.path.exists(os.path.basename(csv_path)):
        csv_path = os.path.basename(csv_path)
        print(f"ℹ Found CSV in root directory: {csv_path}")

    df = pd.DataFrame()
    documents = []

    try:
        # Try to load CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} entries from CSV")

            valid_documents = []
            valid_indices = []

            # Try to find matching text files
            for idx, row in df.iterrows():
                newsgroup = row.get('newsgroup', '')
                doc_id = row.get('document_id', '')

                # Try different path patterns
                possible_paths = [
                    f"data/{newsgroup}/{doc_id}.txt",
                    f"data/{doc_id}.txt",
                    f"{newsgroup}/{doc_id}.txt",
                    f"{doc_id}.txt",
                ]

                text = None
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                                text = f.read()
                            break
                        except:
                            pass

                if text and len(text.strip()) > 10:
                    valid_documents.append(text)
                    valid_indices.append(idx)

            if valid_documents:
                df = df.iloc[valid_indices].reset_index(drop=True)
                documents = valid_documents
                print(f"✓ Found {len(documents)} matching text files")
            else:
                print("⚠ CSV found but no matching text files. Using glob fallback...")
                df = pd.DataFrame()
                documents = []

    except FileNotFoundError:
        print(f"⚠ CSV file not found at '{csv_path}'. Using glob fallback...")
        df = pd.DataFrame()
        documents = []

    # Fallback: Load all .txt files using glob
    if not documents:
        print("Loading all .txt files...")
        filenames = []
        
        # Try data dir first
        files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
        
        # If no files in data dir, try root
        if not files:
             files = glob.glob('*.txt')
             if files:
                 print(f"ℹ Found {len(files)} text files in root directory")

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
                    if len(text.strip()) > 10:
                        documents.append(text)
                        filenames.append(os.path.basename(filepath))
            except Exception as e:
                print(f"  ⚠ Error reading {filepath}: {e}")

        # Check if we loaded aggregated files as single documents
        if len(documents) < 10 and files:
            print("⚠ Few documents loaded. Checking if files are aggregated newsgroups...")
            all_split_docs = []
            all_filenames = []
            
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                        content = f.read()
                        # Check if it looks like aggregated file
                        if content.count('From: ') > 10:
                            print(f"  ℹ Splitting aggregated file: {os.path.basename(filepath)}")
                            # Split by "\nFrom: "
                            parts = content.split('\nFrom: ')
                            
                            for i, part in enumerate(parts):
                                if i > 0:
                                    part = "From: " + part
                                
                                if len(part.strip()) > 100:
                                    all_split_docs.append(part)
                                    all_filenames.append(f"{os.path.basename(filepath)}_{i}")
                except:
                    pass
            
            if len(all_split_docs) > len(documents):
                print(f"✓ Successfully split into {len(all_split_docs)} documents")
                documents = all_split_docs
                filenames = all_filenames

        if documents:
            # Create DataFrame from filenames
            doc_ids = [fn.replace('.txt', '') for fn in filenames]
            newsgroups = []
            for fn in filenames:
                parts = fn.replace('.txt', '').split('.')
                newsgroups.append(parts[0] if len(parts) > 1 else 'unknown')

            df = pd.DataFrame({
                'document_id': doc_ids,
                'newsgroup': newsgroups,
                'filename': filenames
            })
            print(f"✓ Loaded {len(documents)} documents from glob search")

    if not documents:
        print("✗ No documents loaded! Please ensure CSV and/or text files are in the data folder.")
        return pd.DataFrame(), []

    if 'cluster' not in df.columns:
        df['cluster'] = -1

    return df, documents


def save_model(models_data, vectorizer, preprocessor):
    """Save trained models"""
    final_data = {
        'models': models_data,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"✓ Models saved to {MODEL_PATH}")


def train_clustering_pipeline(n_clusters=8):
    """Complete training pipeline"""
    print("\n" + "="*70)
    print("NEWS ARTICLE CLUSTERING - TRAINING PIPELINE")
    print("="*70 + "\n")

    # Step 1: Load data
    print("📂 Loading data...")
    df, documents = load_news_data()

    if not documents:
        print("✗ Failed to load documents. Exiting.")
        return False

    print(f"   Loaded {len(documents)} documents")

    # Step 2: Preprocess
    print("\n📝 Preprocessing documents...")
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)

    # Step 3: Extract features
    print("\n🔍 Extracting TF-IDF features...")
    X, feature_names, vectorizer = extract_tfidf_features(processed_docs, max_features=500)

    models_data = {}

    # Step 4a: Apply K-Means
    print(f"\n🎯 Applying K-Means clustering with {n_clusters} clusters...")
    kmeans_labels, kmeans_model = apply_kmeans(X, n_clusters=n_clusters)
    kmeans_keywords = extract_top_keywords(X, feature_names, kmeans_labels, n_keywords=10)
    
    models_data['kmeans'] = {
        'model': kmeans_model,
        'keywords': kmeans_keywords,
        'labels': kmeans_labels
    }

    # Step 4b: Apply Hierarchical
    print(f"\n🎯 Applying Hierarchical clustering with {n_clusters} clusters...")
    hier_labels, hier_model = apply_hierarchical(X, n_clusters=n_clusters)
    hier_keywords = extract_top_keywords(X, feature_names, hier_labels, n_keywords=10)
    hier_centroids = compute_centroids(X, hier_labels)

    models_data['hierarchical'] = {
        'model': hier_model,
        'keywords': hier_keywords,
        'centroids': hier_centroids,
        'labels': hier_labels
    }

    # Step 4c: Apply DBSCAN
    # Heuristic for eps: usually around 0.5-0.8 for cosine distance (TF-IDF)
    print(f"\n🎯 Applying DBSCAN clustering...")
    dbscan_labels, dbscan_model = apply_dbscan(X, eps=0.5, min_samples=2) # min_samples=2 for small dataset
    dbscan_keywords = extract_top_keywords(X, feature_names, dbscan_labels, n_keywords=10)
    dbscan_centroids = compute_centroids(X, dbscan_labels)

    models_data['dbscan'] = {
        'model': dbscan_model,
        'keywords': dbscan_keywords,
        'centroids': dbscan_centroids,
        'labels': dbscan_labels
    }

    # Step 6: Save model
    print("\n💾 Saving models...")
    save_model(models_data, vectorizer, preprocessor)

    # Add K-Means cluster labels to dataframe (default)
    df['cluster'] = kmeans_labels

    # Save results
    results_path = BASE_DIR / 'clustered_results.csv'
    df.to_csv(results_path, index=False)
    print(f"✓ Results saved to {results_path}")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel Statistics:")
    print(f"  • Total documents: {len(documents)}")
    print(f"  • Clusters (K-Means/Hierarchical): {n_clusters}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Model path: {MODEL_PATH}")
    print("\nYou can now start the Flask app with: python run_app.py")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train news clustering model')
    parser.add_argument('--clusters', type=int, default=8,
                        help='Number of clusters (default: 8)')
    args = parser.parse_args()

    success = train_clustering_pipeline(n_clusters=args.clusters)
    sys.exit(0 if success else 1)
