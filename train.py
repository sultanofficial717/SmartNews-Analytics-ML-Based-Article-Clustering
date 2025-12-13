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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_utils import (
    TextPreprocessor,
    extract_tfidf_features,
    apply_kmeans,
    extract_top_keywords
)

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'news_clustering_model.pkl'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def load_news_data(csv_path='data/list.csv', text_folder='data/*.txt'):
    """Load news articles from CSV and text files"""
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
        
        for filepath in glob.glob(os.path.join(DATA_DIR, '*.txt')):
            try:
                with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
                    if len(text.strip()) > 10:
                        documents.append(text)
                        filenames.append(os.path.basename(filepath))
            except Exception as e:
                print(f"  ⚠ Error reading {filepath}: {e}")

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


def save_model(model, vectorizer, preprocessor, cluster_keywords):
    """Save trained model"""
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'cluster_keywords': cluster_keywords
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to {MODEL_PATH}")


def train_clustering_pipeline(n_clusters=5):
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

    # Step 4: Apply clustering
    print(f"\n🎯 Applying K-Means clustering with {n_clusters} clusters...")
    labels, model = apply_kmeans(X, n_clusters=n_clusters)

    # Step 5: Extract keywords
    print("\n🏷️ Extracting cluster keywords...")
    cluster_keywords = extract_top_keywords(X, feature_names, labels, n_keywords=10)

    # Step 6: Save model
    print("\n💾 Saving model...")
    save_model(model, vectorizer, preprocessor, cluster_keywords)

    # Add cluster labels to dataframe
    df['cluster'] = labels

    # Save results
    results_path = BASE_DIR / 'clustered_results.csv'
    df.to_csv(results_path, index=False)
    print(f"✓ Results saved to {results_path}")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel Statistics:")
    print(f"  • Total documents: {len(documents)}")
    print(f"  • Clusters: {n_clusters}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Model path: {MODEL_PATH}")
    print("\nYou can now start the Flask app with: python -m app.app")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train news clustering model')
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of clusters (default: 5)')
    args = parser.parse_args()

    success = train_clustering_pipeline(n_clusters=args.clusters)
    sys.exit(0 if success else 1)
