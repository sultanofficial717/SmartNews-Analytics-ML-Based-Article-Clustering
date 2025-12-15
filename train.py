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
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    compute_centroids,
    reduce_dimensions,
    OpenRouterEmbeddingGenerator
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
STATIC_DIR = BASE_DIR / 'static'
STATIC_DIR.mkdir(exist_ok=True)


def generate_cluster_plot(X, labels, output_path):
    """Generate and save a 2D scatter plot of clusters"""
    print(f"Generating cluster visualization to {output_path}...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # Add legend
    plt.colorbar(scatter, label='Cluster ID')
    
    plt.title('News Article Clusters (2D Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization saved")


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


def save_model(models_data, vectorizer, preprocessor, lsa_model=None, embedding_config=None):
    """Save trained models"""
    final_data = {
        'models': models_data,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'lsa': lsa_model,
        'embedding_config': embedding_config
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"✓ Models saved to {MODEL_PATH}")


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
    print("\n🔍 Extracting TF-IDF features (for keywords)...")
    X_tfidf, feature_names, vectorizer = extract_tfidf_features(processed_docs, max_features=500)

    # Step 3b: Dimensionality Reduction (LSA) - Fallback
    print("\n📉 Reducing dimensions (LSA) - Fallback/Baseline...")
    X_lsa, lsa_model = reduce_dimensions(X_tfidf, n_components=100)

    # Step 3c: Sentence Embeddings (API)
    print("\n🧠 Fetching Sentence Embeddings (Mistral/OpenRouter)...")
    embedding_generator = OpenRouterEmbeddingGenerator(api_key=API_KEY, model=EMBEDDING_MODEL)
    X_embeddings = embedding_generator.get_embeddings_batch(processed_docs)

    embedding_config = None
    if X_embeddings is not None and len(X_embeddings) == len(documents):
        print(f"✓ Successfully fetched embeddings. Shape: {X_embeddings.shape}")
        X_features = X_embeddings
        embedding_config = {
            'api_key': API_KEY,
            'model': EMBEDDING_MODEL
        }
    else:
        print("⚠ Failed to fetch embeddings. Falling back to TF-IDF + LSA.")
        X_features = X_lsa

    models_data = {}

    # Step 4a: Apply K-Means
    print(f"\n🎯 Applying K-Means clustering with {n_clusters} clusters...")
    kmeans_labels, kmeans_model = apply_kmeans(X_features, n_clusters=n_clusters)
    # Use original TF-IDF for keywords
    kmeans_keywords = extract_top_keywords(X_tfidf, feature_names, kmeans_labels, n_keywords=10)
    
    models_data['kmeans'] = {
        'model': kmeans_model,
        'keywords': kmeans_keywords,
        'labels': kmeans_labels
    }
    
    # Generate Visualization
    plot_path = STATIC_DIR / 'cluster_plot.png'
    generate_cluster_plot(X_features, kmeans_labels, plot_path)

    # Step 4b: Apply Hierarchical
    print(f"\n🎯 Applying Hierarchical clustering with {n_clusters} clusters...")
    hier_labels, hier_model = apply_hierarchical(X_features, n_clusters=n_clusters)
    hier_keywords = extract_top_keywords(X_tfidf, feature_names, hier_labels, n_keywords=10)
    hier_centroids = compute_centroids(X_features, hier_labels)

    models_data['hierarchical'] = {
        'model': hier_model,
        'keywords': hier_keywords,
        'centroids': hier_centroids,
        'labels': hier_labels
    }

    # Step 4c: Apply DBSCAN
    print(f"\n🎯 Applying DBSCAN clustering...")
    # For embeddings (cosine similarity is better, but DBSCAN uses euclidean by default)
    # Embeddings are usually normalized, so euclidean distance is related to cosine similarity.
    # dist = sqrt(2(1-cos)). If cos=0.7 (similar), dist=sqrt(0.6)=0.77. If cos=0.3, dist=sqrt(1.4)=1.18.
    # So eps around 0.8-1.0 is reasonable.
    dbscan_labels, dbscan_model = apply_dbscan(X_features, eps=0.8, min_samples=2) 
    dbscan_keywords = extract_top_keywords(X_tfidf, feature_names, dbscan_labels, n_keywords=10)
    dbscan_centroids = compute_centroids(X_features, dbscan_labels)

    models_data['dbscan'] = {
        'model': dbscan_model,
        'keywords': dbscan_keywords,
        'centroids': dbscan_centroids,
        'labels': dbscan_labels
    }

    # Step 6: Save model
    print("\n💾 Saving models...")
    save_model(models_data, vectorizer, preprocessor, lsa_model, embedding_config)

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
    print(f"  • Features Used: {'Embeddings' if embedding_config else 'TF-IDF+LSA'}")
    print(f"  • Feature Dim: {X_features.shape[1]}")
    print(f"  • Model path: {MODEL_PATH}")
    print("\nYou can now start the Flask app with: python run_app.py")
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
