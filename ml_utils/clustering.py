"""Clustering algorithms module"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def extract_tfidf_features(documents, max_features=1000, ngram_range=(1, 2)):
    """Convert text to TF-IDF vectors"""
    print(f"Extracting TF-IDF features (max_features={max_features})...")

    # Adjust parameters for small datasets
    n_docs = len(documents)
    min_df = 2 if n_docs >= 5 else 1
    # Lower max_df to remove more common words that might blur clusters
    max_df = 0.5 if n_docs >= 5 else 1.0

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Feature matrix shape: {X.shape}")
    return X.toarray(), feature_names, vectorizer


def compute_centroids(X, labels):
    """Compute centroids for clusters"""
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = X[mask].mean(axis=0)
    
    return centroids


def apply_kmeans(X, n_clusters=5):
    """Apply K-Means clustering"""
    print(f"\nApplying K-Means with {n_clusters} clusters...")
    # Increased n_init for better convergence
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    labels = kmeans.fit_predict(X)

    scores = {}
    # Calculate metrics
    if 2 <= n_clusters < X.shape[0]:
        try:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
            scores = {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}
        except Exception as e:
            print(f"Could not calculate metrics: {e}")
    else:
        print(f"Skipping metrics calculation (n_clusters={n_clusters}, n_samples={X.shape[0]})")

    return labels, kmeans, scores


def apply_hierarchical(X, n_clusters=5):
    """Apply Hierarchical clustering"""
    print(f"\nApplying Hierarchical clustering with {n_clusters} clusters...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X)

    scores = {}
    # Calculate metrics
    if 2 <= n_clusters < X.shape[0]:
        try:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
            scores = {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}
        except Exception as e:
            print(f"Could not calculate metrics: {e}")
    else:
        print(f"Skipping metrics calculation (n_clusters={n_clusters}, n_samples={X.shape[0]})")

    return labels, hierarchical, scores


def apply_dbscan(X, eps=0.5, min_samples=5):
    """Apply DBSCAN clustering"""
    print(f"\nApplying DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    scores = {}
    if n_clusters > 1 and n_clusters < X.shape[0]:
        try:
            silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
            print(f"Silhouette Score: {silhouette:.3f}")
            scores = {'silhouette': silhouette}
        except Exception as e:
            print(f"Could not calculate metrics: {e}")

    return labels, dbscan, scores


def extract_top_keywords(X, feature_names, labels, n_keywords=10):
    """Extract top keywords for each cluster"""
    print("\n" + "="*60)
    print("TOP KEYWORDS PER CLUSTER")
    print("="*60)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_keywords = {}

    for cluster_id in range(n_clusters):
        # Get documents in this cluster
        cluster_mask = labels == cluster_id
        cluster_docs = X[cluster_mask]

        # Calculate mean TF-IDF for this cluster
        mean_tfidf = cluster_docs.mean(axis=0)

        # Get top keywords
        top_indices = mean_tfidf.argsort()[-n_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]

        cluster_keywords[cluster_id] = top_keywords

        print(f"\nCluster {cluster_id} ({cluster_mask.sum()} documents):")
        print(f"Keywords: {', '.join(top_keywords)}")

    return cluster_keywords


def reduce_dimensions(X, n_components=100):
    """Reduce dimensions using LSA (TruncatedSVD)"""
    # Adjust n_components if dataset is small
    n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    if n_components < 2:
        n_components = 2
        
    print(f"Reducing dimensions to {n_components} components...")
    lsa = make_pipeline(TruncatedSVD(n_components=n_components, random_state=42), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X)
    
    explained_variance = lsa.steps[0][1].explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance:.2%}")
    
    return X_lsa, lsa


def tune_lsa_kmeans(X_tfidf, max_components=50, max_clusters=10, fixed_n_clusters=None):
    """Tune LSA components and K-Means clusters for best Silhouette Score
    
    Note: This function performs an exhaustive grid search over parameter combinations.
    For large datasets, consider using random search or Bayesian optimization for better performance.
    """
    print("\n[TUNING] Tuning LSA components and K-Means clusters...")
    best_score = -1
    best_params = {'n_components': 10, 'n_clusters': fixed_n_clusters if fixed_n_clusters else 5} 
    
    # Try different n_components
    # For small dataset (124 docs), components should be small
    component_options = [2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
    component_range = [n for n in component_options if n < X_tfidf.shape[0] and n < X_tfidf.shape[1]]
    
    if not component_range:
        component_range = [min(X_tfidf.shape[0]-1, X_tfidf.shape[1]-1, 5)]

    # Determine cluster range
    if fixed_n_clusters:
        cluster_range = [fixed_n_clusters]
        print(f"[TUNING] Fixed number of clusters: {fixed_n_clusters}")
    else:
        cluster_range = range(2, max_clusters + 1)

    for n_comp in component_range:
        # Apply LSA
        try:
            lsa = make_pipeline(TruncatedSVD(n_components=n_comp, random_state=42), Normalizer(copy=False))
            X_lsa = lsa.fit_transform(X_tfidf)
            
            # Try different n_clusters
            for k in cluster_range:
                if k >= X_lsa.shape[0]:
                    continue
                    
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
                labels = kmeans.fit_predict(X_lsa)
                
                try:
                    score = silhouette_score(X_lsa, labels)
                    db_score = davies_bouldin_score(X_lsa, labels)
                    
                    # Prefer fewer clusters if scores are similar (Occam's razor)
                    if score > best_score:
                        best_score = score
                        best_params = {'n_components': n_comp, 'n_clusters': k}
                        print(f"  New best: Sil={score:.3f}, DB={db_score:.3f} (Components={n_comp}, Clusters={k})")
                except ValueError as e:
                    # Silhouette score fails when all points are in one cluster or invalid clustering
                    pass
        except Exception as e:
            print(f"  Error during tuning with n_components={n_comp}: {e}")
                
    print(f"[TUNING] Best parameters found: {best_params} with Silhouette Score: {best_score:.3f}")
    return best_params
