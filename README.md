# 📰 SmartNews Analytics: ML-Based Article Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A comprehensive Machine Learning project for beginners and intermediate learners to understand Unsupervised Learning, NLP pipelines, and Model Deployment.**

---

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Machine Learning Concepts Explained](#-machine-learning-concepts-explained)
- [Project Architecture](#-project-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Docker Support](#-docker-support)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)

---

## 🔭 Project Overview

**SmartNews Analytics** is an intelligent system that automatically groups (clusters) news articles into categories like *Sports, Technology, Health, Politics, etc.* without needing labeled training data. 

It uses **Unsupervised Learning**, meaning the model learns patterns and relationships in the text on its own. This project is designed to be a perfect learning resource for:
- **NLP (Natural Language Processing)**: How to clean and convert text to numbers.
- **Clustering Algorithms**: Understanding K-Means, DBSCAN, and Hierarchical Clustering.
- **ML Pipelines**: Building a robust training and inference workflow.
- **Web Deployment**: Serving an ML model via a Flask API and UI.

> **Note on Data**: Unlike many tutorials that use pre-cleaned datasets (like Kaggle or 20 Newsgroups), this project uses **real-world, raw data** that was manually scraped from various news sources. This demonstrates the challenges and reality of working with custom datasets in a professional environment.

---

## 🌟 Key Features

*   **Custom Scraped Dataset**: Built on a unique collection of articles covering Business, Crime, Medical, Politics, Religion, Sports, and Technology.
*   **Multi-Algorithm Support**: Switch between K-Means, Hierarchical, and DBSCAN clustering.
*   **Advanced NLP Pipeline**: Includes text cleaning, stopword removal, stemming, and TF-IDF vectorization.
*   **Hybrid Feature Extraction**: Supports both traditional **TF-IDF** and modern **LLM Embeddings** (via OpenRouter/Mistral).
*   **Interactive Web Interface**: A beautiful, dark-themed UI to test the model in real-time.
*   **REST API**: Production-ready endpoints for integrating with other apps.
*   **Dockerized**: Ready for containerized deployment.

---

## 🧠 Machine Learning Concepts Explained

If you are new to ML, here is what's happening under the hood:

### 1. The Dataset (Real-World vs. Kaggle)
Most beginners start with "clean" datasets (CSV files with perfect columns). In the real world, data is messy.
- **Our Data**: We scraped raw text files (`business.txt`, `technology.txt`, `crimes.txt`, etc.) from the web.
- **The Challenge**: The model must figure out that `sports.txt` and `sports 1.txt` belong to the same category purely by reading the words inside them, without being told the file names.

### 2. Text Preprocessing (`ml_utils/preprocessor.py`)
Computers can't understand text, so we clean it first:
- **Lowercasing**: "Apple" and "apple" become the same.
- **Noise Removal**: Removing URLs, emails, and special characters.
- **Stopword Removal**: Removing common words like "the", "is", "and" that carry little meaning.
- **Stemming**: Reducing words to their root (e.g., "running" -> "run").

### 3. Feature Extraction (`ml_utils/clustering.py`)
We convert text into numbers (vectors):
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Gives weight to unique words in a document while downplaying common words across all documents.
- **Embeddings (Optional)**: Uses a Large Language Model (Mistral) to understand the *semantic meaning* of sentences, not just keyword matching.

### 4. Dimensionality Reduction (`ml_utils/clustering.py`)
- **LSA (Latent Semantic Analysis)**: Reduces the number of features (from thousands of words to ~100 components) to make clustering faster and more accurate by capturing "concepts" rather than just words.

### 5. Clustering Algorithms
- **K-Means**: The default. It tries to find `K` centers (centroids) and assigns every document to the nearest center. Good for general topics.
- **Hierarchical**: Builds a tree of clusters. Good for seeing how topics relate to sub-topics.
- **DBSCAN**: Groups dense regions of points. Good for finding outliers (noise) and handling irregular cluster shapes.

---

## 📂 Project Architecture

```bash
SmartNews-Analytics/
├── data/                   # 📂 Raw text files for training
├── models/                 # 💾 Saved models (.pkl)
├── ml_utils/               # 🧠 Core ML Logic
│   ├── clustering.py       #   - Clustering algorithms (KMeans, etc.)
│   ├── embeddings.py       #   - API client for LLM embeddings
│   ├── predictor.py        #   - Inference logic for new predictions
│   └── preprocessor.py     #   - Text cleaning pipeline
├── .env                    # 🔐 API Keys (Not committed to Git)
├── Dockerfile              # 🐳 Docker configuration
├── requirements.txt        # 📦 Python dependencies
├── run_app.py              # 🚀 Flask Web Application
└── train.py                # 🚂 Model Training Script
```

---

## ⚡ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/sultanofficial717/SmartNews-Analytics-ML-Based-Article-Clustering.git
cd SmartNews-Analytics-ML-Based-Article-Clustering
```

### 2. Create a Virtual Environment
It's best practice to isolate your dependencies.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory to secure your API keys.
```bash
# Copy the example file
cp .env.example .env
```
Open `.env` and add your OpenRouter API key (optional, only if using embeddings):
```
OPENROUTER_API_KEY=your_key_here
```

---

## 🚀 Usage Guide

### Step 1: Train the Model
Before you can predict, you must train the model on your data.
```bash
# Train with default settings (8 clusters)
python train.py

# Train with a specific number of clusters
python train.py --clusters 5
```
*This will save the trained model to `models/news_clustering_model.pkl`.*

### Step 2: Run the Web Application
Start the Flask server to interact with the model.
```bash
python run_app.py
```
Open your browser and go to: **http://127.0.0.1:5000**

---

## 🐳 Docker Support

Run the application in a container without installing Python locally.

**Build the image:**
```bash
docker build -t news-clustering .
```

**Run the container:**
```bash
docker run -p 5000:5000 news-clustering
```

---

## 🔌 API Documentation

The application exposes a REST API for predictions.

### `POST /api/predict`
Predicts the cluster for a given text.

**Request Body:**
```json
{
    "text": "The stock market crashed today due to inflation fears...",
    "model_type": "kmeans"  // Options: "kmeans", "hierarchical", "dbscan"
}
```

**Response:**
```json
{
    "cluster": 2,
    "confidence": 0.85,
    "keywords": ["market", "economy", "inflation", "stock"],
    "model_used": "kmeans",
    "processed_text": "stock market crash today..."
}
```

---

## 🤝 Contributing

Contributions are welcome! This is an open-source learning project.
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

**Happy Learning! 🚀**
