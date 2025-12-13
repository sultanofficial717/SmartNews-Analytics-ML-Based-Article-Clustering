# 🆘 TROUBLESHOOTING GUIDE

## Common Issues & Solutions

---

## 🔴 Installation Issues

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install Flask specifically
pip install flask==3.0.0
```

### Issue: Python version compatibility
**Solution:**
```bash
# Check Python version (need 3.8+)
python --version

# If wrong version, specify
python3.10 -m pip install -r requirements.txt
python3.10 train.py
python3.10 run.py run
```

### Issue: Virtual environment not activated
**Solution:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify (should show venv in prompt)
```

### Issue: Permission denied when installing
**Solution:**
```bash
# Windows (use administrator)
# Right-click Command Prompt → Run as Administrator

# macOS/Linux
pip install --user -r requirements.txt
```

---

## 🔴 Data Loading Issues

### Issue: "No documents loaded at all"
**Solution:**
1. Verify data folder structure:
```
data/
├── list.csv
├── alt.atheism.txt
├── comp.graphics.txt
├── rec.autos.txt
└── ... (other .txt files)
```

2. Check CSV format:
```csv
newsgroup,document_id
talk.religion.misc,82757
comp.graphics,84000
```

3. Ensure files are readable:
```bash
# List files in data folder
dir data/
```

### Issue: "File encoding error"
**Solution:**
```bash
# Files should be UTF-8 or Latin-1 encoded
# If issue persists, convert files:
# Open file and save as UTF-8 encoding
```

### Issue: CSV not found
**Solution:**
```bash
# Ensure list.csv is in data/ folder, not root
cp list.csv data/list.csv
```

---

## 🔴 Training Issues

### Issue: Training takes too long
**Solution:**
```bash
# Reduce features
python train.py --clusters 3

# Or modify train.py max_features:
# change: extract_tfidf_features(processed_docs, max_features=500)
# to:     extract_tfidf_features(processed_docs, max_features=100)
```

### Issue: NLTK data errors
**Solution:**
```bash
# Download NLTK data manually
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
"
```

### Issue: Memory error during training
**Solution:**
```bash
# Use fewer documents
# Edit train.py to limit documents:
df = df.head(100)  # Use first 100 documents

# Or reduce feature count
max_features=100  # Instead of 500
```

### Issue: "Model training failed"
**Solution:**
1. Check console for specific error
2. Verify data is not corrupted
3. Try with fewer clusters:
```bash
python train.py --clusters 3
```

---

## 🔴 Flask App Issues

### Issue: "Address already in use" (port 5000)
**Solution:**
```bash
# Option 1: Use different port
# Edit app/app.py:
app.run(port=5001)  # Change to 5001

# Option 2: Kill process using port
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti :5000 | xargs kill -9
```

### Issue: "Model not loaded" in web app
**Solution:**
```bash
# Train model first
python train.py

# Check if model file exists
dir models/
# Should show: news_clustering_model.pkl

# Then start app
python run.py run
```

### Issue: Flask app won't start
**Solution:**
```bash
# Check Python path is correct
cd app
python app.py  # Run directly

# Check imports are working
python -c "import flask; import ml_utils; print('OK')"
```

### Issue: "No module named 'ml_utils'"
**Solution:**
```bash
# Ensure you're running from root directory
cd ..  # Go to project root
python run.py run

# Or check __init__.py exists in ml_utils/
dir ml_utils/__init__.py
```

### Issue: Web page won't load
**Solution:**
1. Check Flask is running:
   - Console should show "Running on http://localhost:5000"
2. Check browser address: http://localhost:5000 (not https)
3. Clear browser cache: Ctrl+F5
4. Check browser console (F12) for JavaScript errors

### Issue: CSS/JS not loading
**Solution:**
```bash
# Check static files exist
dir app/static/css/style.css
dir app/static/js/main.js

# If missing, copy them:
cp static/css/style.css app/static/css/
cp static/js/main.js app/static/js/
```

---

## 🔴 Prediction Issues

### Issue: "Model not loaded" error in API
**Solution:**
1. Start Flask app: `python run.py run`
2. Ensure model exists: `models/news_clustering_model.pkl`
3. Check Flask console for errors
4. Wait 2-3 seconds for model to load

### Issue: Prediction taking too long
**Solution:**
- Normal: First prediction takes 1-2 seconds
- Subsequent predictions: 100-500ms
- For batch: Multiple articles take longer
- Check system resources (CPU, RAM)

### Issue: "Text is too short" error
**Solution:**
```bash
# Minimum text length is 10 characters
# Provide longer text: at least 15-20 characters

# Error message will show minimum requirement
```

### Issue: Cluster prediction seems wrong
**Solution:**
1. Check preprocessing is working
2. Verify training completed successfully
3. Check cluster keywords match expected topics
4. Review API documentation for expected output

---

## 🔴 API Issues

### Issue: "CORS error" in browser
**Solution:**
```bash
# For development, CORS is already enabled
# For production, add to app.py:
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

### Issue: "Invalid JSON" error
**Solution:**
```bash
# Ensure JSON is valid
# Correct:
{"text": "Your article here"}

# Wrong:
{"text": 'Your article here'}  # Single quotes

# Use proper tools
# cURL example:
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article"}'
```

### Issue: API returns 500 error
**Solution:**
1. Check Flask console for error message
2. Verify request format is correct
3. Check model is loaded: GET /api/status
4. Try sample endpoint: GET /api/sample-predict

---

## 🔴 Docker Issues

### Issue: "Docker image build fails"
**Solution:**
```bash
# Try with verbose output
docker build --progress=plain -t news-clustering .

# Check Dockerfile for errors
# Ensure all COPY paths exist

# Clean rebuild
docker build --no-cache -t news-clustering .
```

### Issue: "Container won't start"
**Solution:**
```bash
# Check logs
docker logs news-clustering

# Run interactively
docker run -it news-clustering /bin/bash

# Check if training completes
docker build -t test-train .
```

### Issue: "Port mapping error"
**Solution:**
```bash
# Check if port 5000 is in use
docker ps  # See running containers

# Use different port
docker run -p 5001:5000 news-clustering
```

---

## 🔴 GitHub Issues

### Issue: "fatal: not a git repository"
**Solution:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/news-clustering-ml
git push -u origin main
```

### Issue: "Permission denied (publickey)"
**Solution:**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Add public key to GitHub Settings → SSH Keys
```

### Issue: "Large files rejected"
**Solution:**
```bash
# Don't commit model files to Git
# Add to .gitignore:
models/*.pkl
data/*.csv
data/*.txt

# Use Git LFS for large files
git lfs install
git lfs track "*.pkl"
```

---

## 🔴 Performance Issues

### Issue: Slow predictions
**Causes & Solutions:**
1. **Too many features**: Reduce max_features from 500 to 100
2. **Large documents**: Limit text length to 1000 characters
3. **System resources**: Close other applications
4. **Python version**: Use Python 3.10+ (faster)

### Issue: High memory usage
**Solution:**
```bash
# Reduce training data
df = df.sample(n=500)  # Use 500 documents

# Reduce clusters
python train.py --clusters 3

# Reduce features
max_features=100
```

### Issue: Slow web interface
**Solution:**
1. Check network connection
2. Clear browser cache (Ctrl+Shift+Delete)
3. Check browser console (F12) for JavaScript errors
4. Try different browser
5. Check Flask is running smoothly

---

## 🔴 Deployment Issues

### Issue: App works locally but not on server
**Solutions:**
1. Check Python version matches
2. Install all dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Use absolute paths instead of relative
5. Check file permissions

### Issue: 502 Bad Gateway error
**Solution:**
```bash
# For Heroku/Render:
# Check Procfile content:
web: gunicorn -w 4 -b 0.0.0.0:$PORT app.app:app

# Ensure it's correct, then redeploy
```

### Issue: Database/file permission errors
**Solution:**
```bash
# Check file permissions
chmod 755 models/
chmod 644 models/news_clustering_model.pkl

# Ensure data folder is readable
chmod 755 data/
```

---

## ✅ Verification Checklist

### Before Training
- [ ] Python version is 3.8+
- [ ] Dependencies installed: `pip list | grep -E "flask|pandas|scikit"`
- [ ] Data files exist in `data/` folder
- [ ] CSV file has correct format
- [ ] Text files are readable

### Before Running App
- [ ] Model exists: `models/news_clustering_model.pkl`
- [ ] Flask can import: `python -c "import flask; print('OK')"`
- [ ] Port 5000 is free: `netstat -ano | findstr :5000`
- [ ] NLTK data is downloaded

### After Deployment
- [ ] App responds to HTTP requests
- [ ] API endpoint returns data
- [ ] Static files load correctly
- [ ] Error handling works
- [ ] Logs are generated

---

## 🐛 Debug Mode

### Enable detailed logging
```bash
# Edit app/app.py:
app.run(debug=True)  # Already enabled

# Or set environment:
export FLASK_ENV=development
export FLASK_DEBUG=1
```

### Check specific errors
```bash
# Python errors
python -c "
import sys
sys.path.insert(0, '.')
from ml_utils import TextPreprocessor
print('✓ Import successful')
"

# Flask errors
export FLASK_APP=app/app.py
flask shell
```

---

## 📞 Still Having Issues?

1. **Check Logs**: Look at console output carefully
2. **Review Docs**: See README.md and API_DOCUMENTATION.md
3. **Try Examples**: Use QUICKSTART.md examples
4. **Verify Setup**: Follow installation steps again
5. **Check Versions**: Ensure dependency versions match requirements.txt

---

## 🚀 Quick Recovery Steps

If everything is broken:

```bash
# 1. Fresh environment
deactivate
rmdir venv
python -m venv venv
venv\Scripts\activate

# 2. Reinstall
pip install -r requirements.txt

# 3. Retrain
python train.py

# 4. Test
python run.py run
```

---

**Need more help?** Check the complete documentation in README.md!
