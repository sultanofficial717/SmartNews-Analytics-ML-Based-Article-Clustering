# QUICK START GUIDE

## 🎯 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
Ensure these files are in the `data/` folder:
- `list.csv` - CSV file with newsgroup and document_id columns
- `*.txt` - Text files with article content (named as document_id.txt)

### Step 3: Train the Model
```bash
python train.py
```
This will create `models/news_clustering_model.pkl`

### Step 4: Run the Web App
```bash
cd app
python app.py
```

### Step 5: Open Your Browser
Go to: `http://localhost:5000`

---

## 📚 Features You Can Use

### 1. Single Prediction
Paste any news article and get its cluster instantly

### 2. Batch Prediction
Enter multiple articles (one per line) for bulk prediction

### 3. API Access
Use REST endpoints for integration with other apps

### 4. Model Information
View cluster keywords and statistics

---

## 🔧 Customize Clusters

Train with different number of clusters:
```bash
python train.py --clusters 7
```

## 📊 View Results
After training, check `clustered_results.csv` for:
- Document IDs
- Assigned clusters
- Newsgroups

---

## 🆘 Troubleshooting

**Q: "Model not loaded" error**
A: Run `python train.py` first

**Q: No documents loaded**
A: Check `data/` folder contains CSV and TXT files

**Q: Port 5000 already in use**
A: Edit `app/app.py` line and change port to 5001

**Q: NLTK data error**
A: Manually run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```

---

## 📖 API Examples

### Python Requests
```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/api/predict', 
    json={'text': 'Your news article here'})
result = response.json()
print(f"Cluster: {result['cluster']}")
print(f"Keywords: {result['keywords']}")
```

### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article here"}'
```

### JavaScript Fetch
```javascript
fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'Your article'})
}).then(r => r.json()).then(data => console.log(data));
```

---

## 🚀 Deployment

### Local with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

### Docker
```bash
docker build -t news-clustering .
docker run -p 5000:5000 news-clustering
```

### Environment Variables
Create `.env` file:
```
FLASK_ENV=production
SECRET_KEY=your-secret-key
DEBUG=False
```

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Review console output for error messages
3. Verify data files are properly formatted
4. Check requirements.txt versions match

---

**Happy Clustering! 🎉**
