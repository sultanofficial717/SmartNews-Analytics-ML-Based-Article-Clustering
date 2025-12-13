# News Article Clustering System

A machine learning project demonstrating unsupervised text clustering using K-Means, Hierarchical, and DBSCAN algorithms. This system groups news articles into distinct categories based on their content using TF-IDF vectorization and various clustering techniques.

## Project Overview

This project serves as an educational resource for understanding how unsupervised learning algorithms can organize unstructured text data. It implements a complete pipeline from raw text processing to a deployment-ready web application.

### Key Concepts

*   **Text Preprocessing**: Cleaning, tokenization, and stemming/lemmatization to prepare text for analysis.
*   **TF-IDF Vectorization**: Converting text into numerical vectors (Term Frequency-Inverse Document Frequency) to represent the importance of words.
*   **Clustering Algorithms**:
    *   **K-Means**: Partitions data into *k* distinct clusters based on distance to centroids.
    *   **Hierarchical Clustering**: Builds a tree of clusters to find natural groupings.
    *   **DBSCAN**: Density-based clustering that can find arbitrarily shaped clusters and identify noise.
*   **Model Persistence**: Saving trained models and vectorizers for real-time inference.

## Project Structure

```
.
├── data/                   # Training data (text files)
├── ml_utils/               # Python package for ML operations
│   ├── clustering.py       # Clustering algorithm implementations
│   ├── preprocessor.py     # Text cleaning and normalization
│   └── predictor.py        # Inference logic
├── models/                 # Directory for saved model artifacts
├── run_app.py              # Flask web application entry point
├── train.py                # Script to train and save the models
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1.  **Clone the repository** (or download the source code).

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data** (automatically handled by the scripts, but can be done manually):
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```

## Usage

### 1. Training the Model

The training script reads text files from the `data/` directory, processes them, and trains multiple clustering models.

```bash
python train.py --clusters 4
```

*   `--clusters`: Specify the number of expected clusters (default is 5). Set this to match the number of distinct topics in your dataset.

The trained models will be saved to `models/news_clustering_model.pkl`.

### 2. Running the Web Application

Launch the Flask web interface to interact with the trained models.

```bash
python run_app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

### 3. Using the Interface

*   **Select Model**: Choose between K-Means, Hierarchical, or DBSCAN clustering.
*   **Input Text**: Paste a news article or use one of the sample texts.
*   **Predict**: The system will classify the text into one of the trained clusters and provide a confidence score.

## Customization

To train on your own data:
1.  Place your text files (`.txt`) in the `data/` directory.
2.  Run `python train.py --clusters N`, where N is the number of topics/files you added.

## License

This project is open source and available under the MIT License.
