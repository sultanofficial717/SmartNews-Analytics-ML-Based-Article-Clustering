# API Documentation

## Base URL
```
http://localhost:5000/api
```

## Endpoints

### 1. Check Model Status

**Endpoint:** `GET /status`

**Description:** Check if the model is loaded and ready for predictions

**Response:**
```json
{
  "model_loaded": true,
  "message": "Model is ready!"
}
```

**Status Codes:**
- `200`: Success

**Example:**
```bash
curl http://localhost:5000/api/status
```

---

### 2. Single Prediction

**Endpoint:** `POST /predict`

**Description:** Predict the cluster for a single news article

**Request:**
```json
{
  "text": "The stock market showed strong gains today as technology companies reported better-than-expected earnings."
}
```

**Response:**
```json
{
  "cluster": 2,
  "keywords": [
    "market",
    "stock",
    "economic",
    "trading",
    "finance",
    "investment",
    "growth",
    "earnings",
    "company",
    "profit"
  ],
  "confidence": 0.8734,
  "preview": "The stock market showed strong gains today as technology companies reported better-than-expected earnings."
}
```

**Parameters:**
- `text` (string, required): News article text (minimum 10 characters, maximum 5000 characters)

**Response Fields:**
- `cluster` (integer): Predicted cluster ID (0 to n_clusters-1)
- `keywords` (array): Top keywords for the cluster
- `confidence` (float): Confidence score (0 to 1)
- `preview` (string): Preview of the input text

**Status Codes:**
- `200`: Success
- `400`: Bad request (missing or invalid text)
- `500`: Server error

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article here..."}'
```

---

### 3. Batch Prediction

**Endpoint:** `POST /predict-batch`

**Description:** Predict clusters for multiple articles at once

**Request:**
```json
{
  "texts": [
    "The president announced new healthcare policies.",
    "Scientists discovered a new species of deep-sea fish.",
    "The championship game went into overtime with an exciting finish."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "cluster": 0,
      "keywords": ["health", "policy", "government", "care", "disease", ...],
      "confidence": 0.8456,
      "preview": "The president announced new healthcare policies."
    },
    {
      "cluster": 3,
      "keywords": ["species", "science", "discovery", "research", "animal", ...],
      "confidence": 0.9123,
      "preview": "Scientists discovered a new species of deep-sea fish."
    },
    {
      "cluster": 4,
      "keywords": ["game", "sport", "team", "championship", "play", ...],
      "confidence": 0.8789,
      "preview": "The championship game went into overtime with an exciting finish."
    }
  ],
  "count": 3
}
```

**Parameters:**
- `texts` (array, required): Array of text strings (each minimum 10 characters)

**Response Fields:**
- `predictions` (array): Array of prediction objects
- `count` (integer): Number of predictions returned

**Constraints:**
- Maximum 50 articles per request
- Each article must be at least 10 characters
- Total payload size limited to 16MB

**Status Codes:**
- `200`: Success
- `400`: Bad request (invalid format or constraints violated)
- `500`: Server error

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Article 1 text...",
      "Article 2 text...",
      "Article 3 text..."
    ]
  }'
```

---

### 4. Model Information

**Endpoint:** `GET /model-info`

**Description:** Get information about the trained model

**Response:**
```json
{
  "n_clusters": 5,
  "clusters": {
    "0": {
      "id": 0,
      "top_keywords": [
        "government",
        "politics",
        "congress",
        "bill",
        "vote"
      ]
    },
    "1": {
      "id": 1,
      "top_keywords": [
        "technology",
        "software",
        "computer",
        "digital",
        "internet"
      ]
    },
    "2": {
      "id": 2,
      "top_keywords": [
        "sport",
        "team",
        "game",
        "player",
        "win"
      ]
    },
    "3": {
      "id": 3,
      "top_keywords": [
        "science",
        "research",
        "study",
        "scientist",
        "discovery"
      ]
    },
    "4": {
      "id": 4,
      "top_keywords": [
        "market",
        "stock",
        "economic",
        "trading",
        "finance"
      ]
    }
  }
}
```

**Status Codes:**
- `200`: Success
- `400`: Model not loaded
- `500`: Server error

**Example:**
```bash
curl http://localhost:5000/api/model-info
```

---

### 5. Sample Predictions

**Endpoint:** `GET /sample-predict`

**Description:** Get predictions for pre-defined sample articles

**Response:**
```json
{
  "predictions": [
    {
      "cluster": 2,
      "keywords": ["market", "stock", "economic", "trading", "finance"],
      "confidence": 0.8734,
      "preview": "The stock market showed strong gains today..."
    },
    {
      "cluster": 3,
      "keywords": ["science", "research", "study", "scientist", "discovery"],
      "confidence": 0.9123,
      "preview": "Scientists have discovered a new species..."
    },
    {
      "cluster": 4,
      "keywords": ["sport", "team", "game", "player", "win"],
      "confidence": 0.8789,
      "preview": "The championship game went into overtime..."
    },
    {
      "cluster": 0,
      "keywords": ["health", "policy", "government", "care", "disease"],
      "confidence": 0.8456,
      "preview": "New healthcare policies were announced..."
    },
    {
      "cluster": 1,
      "keywords": ["technology", "software", "computer", "digital", "internet"],
      "confidence": 0.9012,
      "preview": "Machine learning algorithms are revolutionizing..."
    }
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Model not loaded
- `500`: Server error

**Example:**
```bash
curl http://localhost:5000/api/sample-predict
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "No text provided"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

---

## Request Examples

### Python
```python
import requests

BASE_URL = "http://localhost:5000/api"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "Your news article here..."}
)
result = response.json()
print(f"Cluster: {result['cluster']}")
print(f"Keywords: {result['keywords']}")

# Batch prediction
texts = [
    "Article 1...",
    "Article 2...",
    "Article 3..."
]
response = requests.post(
    f"{BASE_URL}/predict-batch",
    json={"texts": texts}
)
results = response.json()
for pred in results['predictions']:
    print(f"Cluster: {pred['cluster']}")
```

### JavaScript
```javascript
const BASE_URL = "http://localhost:5000/api";

// Single prediction
async function predictSingle(text) {
    const response = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
    });
    return await response.json();
}

// Batch prediction
async function predictBatch(texts) {
    const response = await fetch(`${BASE_URL}/predict-batch`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({texts})
    });
    return await response.json();
}

// Usage
const result = await predictSingle("Your article text");
console.log(`Cluster: ${result.cluster}`);
```

### cURL
```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text..."}'

# Batch prediction
curl -X POST http://localhost:5000/api/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Article 1...",
      "Article 2...",
      "Article 3..."
    ]
  }'

# Model info
curl http://localhost:5000/api/model-info

# Sample predictions
curl http://localhost:5000/api/sample-predict
```

---

## Rate Limiting

- No hard rate limits currently enforced
- Recommended: Max 100 requests/minute
- Each prediction takes ~100-500ms depending on text length

---

## CORS

CORS is enabled for development. For production:

```python
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": ["your-domain.com"]}})
```

---

## Version History

- **v1.0.0** (2024-12-13): Initial release
  - Single and batch predictions
  - Model information endpoint
  - Sample predictions endpoint

---

## Support

For issues or questions about the API:
1. Check this documentation
2. Review console logs
3. Check README.md
4. Open an issue on GitHub

---

**Last Updated**: December 2024
