"""
Model training script for news clustering
Run this to train the model before starting the Flask app
"""
print("DEBUG: Script starting...")

import pandas as pd
import numpy as np
import glob
import os
import pickle
import sys
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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
    extract_top_keywords,
    compute_centroids,
    reduce_dimensions,
    tune_lsa_kmeans,
    analyze_silhouette_scores
)
from ml_utils.embeddings import OpenRouterEmbeddingGenerator

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'news_clustering_model.pkl'

# API Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("[WARN] Warning: OPENROUTER_API_KEY not found in environment variables.")
    print("  Embeddings will not be generated (falling back to TF-IDF).")
    
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / 'static'
STATIC_DIR.mkdir(exist_ok=True)


def generate_dendrogram_plot(X, output_path):
    """Generate and save a dendrogram plot"""
    print(f"Generating dendrogram to {output_path}...")
    
    plt.figure(figsize=(12, 8))
    
    # Compute linkage matrix
    Z = linkage(X, method='ward')
    
    # Plot dendrogram
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Dendrogram saved")


def generate_cluster_plot(X, labels, output_path, title='Cluster Visualization'):
    """Generate and save a 2D scatter plot of clusters with centroids"""
    print(f"Generating cluster visualization to {output_path}...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    
    # Get unique labels
    unique_labels = sorted(list(set(labels)))
    
    # Create scatter plot
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # Plot centroids
    for label in unique_labels:
        if label == -1:
            continue # Skip noise centroid
            
        mask = labels == label
        if np.any(mask):
            centroid = X_2d[mask].mean(axis=0)
            plt.plot(centroid[0], centroid[1], 'rX', markersize=12, markeredgewidth=2, label='Centroid' if label==unique_labels[0] else "")
            plt.annotate(f'C{label}', (centroid[0], centroid[1]), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold', color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.colorbar(scatter, label='Cluster ID')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Visualization saved")


def generate_dendrogram_plot(X, output_path):
    """Generate and save dendrogram for hierarchical clustering"""
    print(f"Generating dendrogram to {output_path}...")
    
    # Compute linkage matrix
    # Use a subset if dataset is too large for visualization
    if X.shape[0] > 1000:
        print("  Subsampling for dendrogram...")
        indices = np.random.choice(X.shape[0], 1000, replace=False)
        X_plot = X[indices]
    else:
        X_plot = X
        
    Z = linkage(X_plot, method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Size / Sample Index')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Dendrogram saved")


def load_news_data(csv_path='data/list.csv', text_folder='data/*.txt'):
    """Load news articles from CSV and text files"""
    
    # Check if csv_path exists, if not check if it exists in root
    if not os.path.exists(csv_path) and os.path.exists(os.path.basename(csv_path)):
        csv_path = os.path.basename(csv_path)
        print(f"[INFO] Found CSV in root directory: {csv_path}")

    df = pd.DataFrame()
    documents = []

    try:
        # Try to load CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[OK] Loaded {len(df)} entries from CSV")

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
                print(f"[OK] Found {len(documents)} matching text files")
            else:
                print("[WARN] CSV found but no matching text files. Using glob fallback...")
                df = pd.DataFrame()
                documents = []

    except FileNotFoundError:
        print(f"[WARN] CSV file not found at '{csv_path}'. Using glob fallback...")
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
                 print(f"[INFO] Found {len(files)} text files in root directory")

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
                    if len(text.strip()) > 10:
                        documents.append(text)
                        filenames.append(os.path.basename(filepath))
            except Exception as e:
                print(f"  [WARN] Error reading {filepath}: {e}")

        # Check if we loaded aggregated files as single documents
        if len(documents) < 10 and files:
            print("[WARN] Few documents loaded. Checking if files are aggregated newsgroups...")
            all_split_docs = []
            all_filenames = []
            
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                        content = f.read()
                        # Check if it looks like aggregated file
                        if content.count('From: ') > 10:
                            print(f"  [INFO] Splitting aggregated file: {os.path.basename(filepath)}")
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
                print(f"[OK] Successfully split into {len(all_split_docs)} documents")
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
            print(f"[OK] Loaded {len(documents)} documents from glob search")

    if not documents:
        print("[ERROR] No documents loaded! Please ensure CSV and/or text files are in the data folder.")
        return pd.DataFrame(), []

    if 'cluster' not in df.columns:
        df['cluster'] = -1

    return df, documents


def save_model(models_data, vectorizer, preprocessor, lsa_model=None):
    """Save trained models"""
    final_data = {
        'models': models_data,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'lsa_model': lsa_model
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"[OK] Models saved to {MODEL_PATH}")


def train_clustering_pipeline(n_clusters=5):
    """Complete training pipeline"""
    print("\n" + "="*70)
    print("NEWS ARTICLE CLUSTERING - TRAINING PIPELINE")
    print("="*70 + "\n")

    # Step 1: Load data
    print("[DATA] Loading data...")
    df, documents = load_news_data()

    if not documents:
        print("[ERROR] Failed to load documents. Exiting.")
        return False

    print(f"   Loaded {len(documents)} documents")

    # Step 2: Preprocess
    print("\n[PREP] Preprocessing documents...")
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)

    # Step 3: Extract TF-IDF features
    print("\n[TFIDF] Extracting TF-IDF features...")
    X_tfidf, feature_names, vectorizer = extract_tfidf_features(processed_docs)
    print(f"   TF-IDF matrix shape: {X_tfidf.shape}")

    # --- LSA+TF-IDF Model ---
    print("\n[LSA+TFIDF] Training pipeline...")
    best_params_tfidf = tune_lsa_kmeans(X_tfidf, max_components=50, max_clusters=10, fixed_n_clusters=n_clusters)
    n_components_tfidf = best_params_tfidf['n_components']
    n_clusters_tfidf = best_params_tfidf['n_clusters']
    print(f"[LSA+TFIDF] Using tuned parameters: n_components={n_components_tfidf}, n_clusters={n_clusters_tfidf}")
    X_lsa_tfidf, lsa_model_tfidf = reduce_dimensions(X_tfidf, n_components=n_components_tfidf)
    silhouette_path_tfidf = STATIC_DIR / 'silhouette_analysis_tfidf.png'
    analyze_silhouette_scores(X_lsa_tfidf, max_k=15, output_path=silhouette_path_tfidf)
    dendrogram_path_tfidf = STATIC_DIR / 'dendrogram_tfidf.png'
    generate_dendrogram_plot(X_lsa_tfidf, dendrogram_path_tfidf)
    models_data = {'lsa_tfidf': {}}
    # KMeans
    kmeans_labels_tfidf, kmeans_model_tfidf, kmeans_scores_tfidf = apply_kmeans(X_lsa_tfidf, n_clusters=n_clusters_tfidf)
    kmeans_keywords_tfidf = extract_top_keywords(X_tfidf, feature_names, kmeans_labels_tfidf, n_keywords=10)
    models_data['lsa_tfidf']['kmeans'] = {
        'model': kmeans_model_tfidf,
        'keywords': kmeans_keywords_tfidf,
        'labels': kmeans_labels_tfidf,
        'scores': kmeans_scores_tfidf
    }
    plot_path_tfidf = STATIC_DIR / 'kmeans_plot_tfidf.png'
    generate_cluster_plot(X_lsa_tfidf, kmeans_labels_tfidf, plot_path_tfidf, title='K-Means Clustering (LSA+TF-IDF)')
    # Hierarchical
    hier_labels_tfidf, hier_model_tfidf, hier_scores_tfidf = apply_hierarchical(X_lsa_tfidf, n_clusters=n_clusters_tfidf)
    hier_keywords_tfidf = extract_top_keywords(X_tfidf, feature_names, hier_labels_tfidf, n_keywords=10)
    hier_centroids_tfidf = compute_centroids(X_lsa_tfidf, hier_labels_tfidf)
    generate_cluster_plot(X_lsa_tfidf, hier_labels_tfidf, STATIC_DIR / 'hierarchical_plot_tfidf.png', title='Hierarchical Clustering (LSA+TF-IDF)')
    models_data['lsa_tfidf']['hierarchical'] = {
        'model': hier_model_tfidf,
        'keywords': hier_keywords_tfidf,
        'centroids': hier_centroids_tfidf,
        'labels': hier_labels_tfidf,
        'scores': hier_scores_tfidf
    }

    # --- Embedding Model ---
    if API_KEY:
        print("\n[EMBEDDING] Training pipeline...")
        try:
            embedder = OpenRouterEmbeddingGenerator(API_KEY, model=EMBEDDING_MODEL)
            X_embeddings = embedder.get_embeddings_batch(processed_docs)
            if X_embeddings is not None and len(X_embeddings) == len(processed_docs):
                best_params_emb = tune_lsa_kmeans(X_embeddings, max_components=50, max_clusters=10, fixed_n_clusters=n_clusters)
                n_components_emb = best_params_emb['n_components']
                n_clusters_emb = best_params_emb['n_clusters']
                print(f"[EMBEDDING] Using tuned parameters: n_components={n_components_emb}, n_clusters={n_clusters_emb}")
                X_lsa_emb, lsa_model_emb = reduce_dimensions(X_embeddings, n_components=n_components_emb)
                silhouette_path_emb = STATIC_DIR / 'silhouette_analysis_emb.png'
                analyze_silhouette_scores(X_lsa_emb, max_k=15, output_path=silhouette_path_emb)
                dendrogram_path_emb = STATIC_DIR / 'dendrogram_emb.png'
                generate_dendrogram_plot(X_lsa_emb, dendrogram_path_emb)
                models_data['embedding'] = {}
                # KMeans
                kmeans_labels_emb, kmeans_model_emb, kmeans_scores_emb = apply_kmeans(X_lsa_emb, n_clusters=n_clusters_emb)
                kmeans_keywords_emb = extract_top_keywords(X_tfidf, feature_names, kmeans_labels_emb, n_keywords=10)
                models_data['embedding']['kmeans'] = {
                    'model': kmeans_model_emb,
                    'keywords': kmeans_keywords_emb,
                    'labels': kmeans_labels_emb,
                    'scores': kmeans_scores_emb
                }
                plot_path_emb = STATIC_DIR / 'kmeans_plot_emb.png'
                generate_cluster_plot(X_lsa_emb, kmeans_labels_emb, plot_path_emb, title='K-Means Clustering (Embedding)')
                # Hierarchical
                hier_labels_emb, hier_model_emb, hier_scores_emb = apply_hierarchical(X_lsa_emb, n_clusters=n_clusters_emb)
                hier_keywords_emb = extract_top_keywords(X_tfidf, feature_names, hier_labels_emb, n_keywords=10)
                hier_centroids_emb = compute_centroids(X_lsa_emb, hier_labels_emb)
                generate_cluster_plot(X_lsa_emb, hier_labels_emb, STATIC_DIR / 'hierarchical_plot_emb.png', title='Hierarchical Clustering (Embedding)')
                models_data['embedding']['hierarchical'] = {
                    'model': hier_model_emb,
                    'keywords': hier_keywords_emb,
                    'centroids': hier_centroids_emb,
                    'labels': hier_labels_emb,
                    'scores': hier_scores_emb
                }
            else:
                print("[WARN] Embedding generation failed. Skipping embedding model.")
        except Exception as e:
            print(f"[WARN] Error generating embeddings: {e}. Skipping embedding model.")

    # Step 6: Save model
    print("\n[SAVE] Saving models...")
    save_model(models_data, vectorizer, preprocessor, None)

    # Add K-Means cluster labels to dataframe (default, from LSA+TFIDF)
    df['cluster'] = models_data['lsa_tfidf']['kmeans']['labels']

    # Save results
    results_path = BASE_DIR / 'clustered_results.csv'
    df.to_csv(results_path, index=False)
    print(f"[OK] Results saved to {results_path}")

    print("\n" + "="*70)
    print("[DONE] TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel Statistics:")
    print(f"  • Total documents: {len(documents)}")
    print(f"  • Clusters (K-Means/Hierarchical): {n_clusters}")
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
