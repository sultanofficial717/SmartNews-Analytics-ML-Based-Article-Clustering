# News Article Clustering ML Project

An intelligent news article clustering system using Machine Learning with K-Means algorithm and TF-IDF feature extraction. Features a beautiful Flask web interface for interactive predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Features

- **K-Means Clustering**: Intelligent grouping of news articles
- **TF-IDF Feature Extraction**: Advanced text processing and vectorization
- **Text Preprocessing**: Tokenization, lemmatization, stopword removal
- **REST API**: Full-featured API for predictions
- **Elegant Web Interface**: Modern, responsive Flask frontend
- **Batch Processing**: Predict clusters for multiple articles at once
- **Model Persistence**: Save and load trained models
- **Cluster Analysis**: View top keywords and statistics per cluster

## 🏗️ Project Structure

```
ML Project Dataset/
├── app/                          # Flask application
│   ├── app.py                   # Main Flask app
│   ├── templates/
│   │   └── index.html           # Web interface
│   └── static/
│       ├── css/
│       │   └── style.css        # Styling
│       └── js/
│           └── main.js          # Frontend logic
├── ml_utils/                     # Machine learning modules
│   ├── __init__.py
│   ├── preprocessing.py         # Text preprocessing
│   ├── clustering.py            # Clustering algorithms
│   └── predictor.py             # Prediction class
├── data/                         # Data files
│   ├── list.csv                 # Document metadata
│   └── *.txt                    # Text files
├── models/                       # Trained models
│   └── news_clustering_model.pkl
├── train.py                      # Training script
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project:**
```bash
cd "ML Project Dataset"
```

2. **Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Training the Model

Before running the Flask app, train the clustering model:

```bash
# Train with default 5 clusters
python train.py

# Train with custom number of clusters
python train.py --clusters 7
```

The training script will:
- Load text files from the `data/` directory
- Preprocess all documents
- Extract TF-IDF features
- Apply K-Means clustering
- Save the trained model to `models/news_clustering_model.pkl`

### Running the Flask App

```bash
# Navigate to app directory
cd app

# Run Flask development server
python app.py
```

The application will be available at `http://localhost:5000`

## 📖 Usage

### Web Interface

1. **Single Prediction**: Enter a news article in the text area and click "Predict Cluster"
2. **Sample Predictions**: Click "Load Sample" to see pre-loaded examples
3. **Batch Prediction**: Enter multiple articles (one per line) and predict all at once
4. **View Results**: See cluster assignment, confidence score, and related keywords

### API Endpoints

#### Check Model Status
```bash
GET /api/status
```

Response:
```json
{
  "model_loaded": true,
  "message": "Model is ready!"
}
```

#### Single Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "text": "Your news article text here..."
}
```

Response:
```json
{
  "cluster": 2,
  "keywords": ["economy", "market", "trade", ...],
  "confidence": 0.87,
  "preview": "Your news article text here..."
}
```

#### Batch Prediction
```bash
POST /api/predict-batch
Content-Type: application/json

{
  "texts": [
    "Article 1 text...",
    "Article 2 text...",
    "Article 3 text..."
  ]
}
```

#### Model Information
```bash
GET /api/model-info
```

Response:
```json
{
  "n_clusters": 5,
  "clusters": {
    "0": {
      "id": 0,
      "top_keywords": ["politics", "government", "congress", ...]
    },
    ...
  }
}
```

## 🔧 Configuration

### Model Parameters

Edit `train.py` to modify:
- `n_clusters`: Number of clusters (default: 5)
- `max_features`: Maximum TF-IDF features (default: 500)
- `ngram_range`: N-gram range for TF-IDF (default: (1, 2))

### Flask Configuration

Edit `app/app.py` to modify:
- `DEBUG`: Debug mode (default: True)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)

## 📊 Model Details

### Algorithm: K-Means Clustering
- **Initialization**: k-means++ (n_init=10)
- **Distance Metric**: Euclidean
- **Convergence**: Tolerance = 1e-4

### Text Processing Pipeline
1. **Lowercase conversion**: Normalize text case
2. **URL removal**: Strip URLs and email addresses
3. **Number removal**: Remove all digits
4. **Punctuation removal**: Clean special characters
5. **Tokenization**: Split text into words
6. **Lemmatization**: Reduce words to base form
7. **Stopword removal**: Remove common words

### Feature Extraction: TF-IDF
- **max_features**: 500 (most important features)
- **ngram_range**: (1, 2) (unigrams and bigrams)
- **min_df**: 2 (appears in at least 2 documents)
- **max_df**: 0.8 (appears in at most 80% of documents)

## 📈 Performance Metrics

The model provides:
- **Silhouette Score**: Measures cluster cohesion (-1 to 1)
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Confidence Score**: Per-prediction confidence based on distance to cluster center

## 🐳 Docker Support (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5000

CMD ["python", "-m", "app.app"]
```

Build and run:
```bash
docker build -t news-clustering .
docker run -p 5000:5000 news-clustering
```

## 📝 Data Format

### CSV File (list.csv)
```csv
newsgroup,document_id
talk.religion.misc,82757
comp.graphics,84000
sci.med,72345
```

### Text Files
Place `.txt` files in the `data/` directory or organize by newsgroup:
```
data/
├── talk.religion.misc/
│   ├── 82757.txt
│   ├── 82758.txt
│   └── ...
├── comp.graphics/
│   ├── 84000.txt
│   └── ...
└── sci.med/
    ├── 72345.txt
    └── ...
```

## 🔍 Troubleshooting

### Model Not Loading
- Ensure `train.py` has been run successfully
- Check `models/` directory contains `news_clustering_model.pkl`
- Review console output for NLTK download errors

### No Documents Loaded
- Verify text files are in `data/` directory
- Check file encoding is compatible (UTF-8 or Latin-1)
- Ensure `list.csv` paths match actual file locations

### Performance Issues
- Reduce `max_features` in `train.py` (default: 500)
- Use fewer clusters (`--clusters` parameter)
- Process documents in batches rather than all at once

### Port Already in Use
```bash
# Change port in app/app.py
app.run(port=5001)
```

## 📦 Dependencies

- **flask**: Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Visualization

See `requirements.txt` for full list with versions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author

Created as an ML Project for news article clustering and categorization.

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review console logs for errors
3. Open an issue on GitHub

## 🎯 Future Enhancements

- [ ] Support for hierarchical and DBSCAN clustering
- [ ] Real-time model retraining
- [ ] User authentication and history
- [ ] Advanced visualization dashboard
- [ ] Multi-language support
- [ ] Document similarity search
- [ ] Clustering quality metrics UI
- [ ] Export results to PDF

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [NLTK Documentation](https://www.nltk.org/)
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

**Last Updated**: December 2024  
**Version**: 1.0.0
