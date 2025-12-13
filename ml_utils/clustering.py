"""Clustering algorithms module"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_features(documents, max_features=1000, ngram_range=(1, 2)):
    """Convert text to TF-IDF vectors"""
    print(f"Extracting TF-IDF features (max_features={max_features})...")

    # Adjust parameters for small datasets
    n_docs = len(documents)
    min_df = 2 if n_docs >= 5 else 1
    max_df = 0.8 if n_docs >= 5 else 1.0

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Calculate metrics
    if 2 <= n_clusters < X.shape[0]:
        try:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        except Exception as e:
            print(f"Could not calculate metrics: {e}")
    else:
        print(f"Skipping metrics calculation (n_clusters={n_clusters}, n_samples={X.shape[0]})")

    return labels, kmeans


def apply_hierarchical(X, n_clusters=5):
    """Apply Hierarchical clustering"""
    print(f"\nApplying Hierarchical clustering with {n_clusters} clusters...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X)

    # Calculate metrics
    if 2 <= n_clusters < X.shape[0]:
        try:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        except Exception as e:
            print(f"Could not calculate metrics: {e}")
    else:
        print(f"Skipping metrics calculation (n_clusters={n_clusters}, n_samples={X.shape[0]})")

    return labels, hierarchical


def apply_dbscan(X, eps=0.5, min_samples=5):
    """Apply DBSCAN clustering"""
    print(f"\nApplying DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    if n_clusters > 1 and n_clusters < X.shape[0]:
        try:
            silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
            print(f"Silhouette Score: {silhouette:.3f}")
        except Exception as e:
            print(f"Could not calculate metrics: {e}")

    return labels, dbscan


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
