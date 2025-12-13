/* ===========================
   NEWS CLUSTERING APP - MAIN JAVASCRIPT
   =========================== */

const API_BASE = '/api';

// DOM Elements
const articleInput = document.getElementById('article-input');
const charCounter = document.getElementById('char-counter');
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const sampleBtn = document.getElementById('sample-btn');
const resultContainer = document.getElementById('result-container');
const closeResultBtn = document.getElementById('close-result');

const batchInput = document.getElementById('batch-input');
const batchPredictBtn = document.getElementById('batch-predict-btn');
const batchResults = document.getElementById('batch-results');

const loadingSpinner = document.getElementById('loading-spinner');
const toast = document.getElementById('toast');
const statusBadge = document.getElementById('status-badge');

let clusterKeywords = {};

// ===========================
// INITIALIZATION
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    checkModelStatus();
    loadModelInfo();
});

function initializeApp() {
    console.log('🚀 News Clustering App initialized');
}

function setupEventListeners() {
    // Prediction form
    articleInput.addEventListener('input', updateCharCounter);
    predictBtn.addEventListener('click', handlePredict);
    clearBtn.addEventListener('click', handleClear);
    sampleBtn.addEventListener('click', handleLoadSample);
    closeResultBtn.addEventListener('click', () => resultContainer.classList.add('hidden'));

    // Batch prediction
    batchPredictBtn.addEventListener('click', handleBatchPredict);

    // Prevent form submission
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', e => e.preventDefault());
    });
}

// ===========================
// UTILITY FUNCTIONS
// ===========================

function showLoading(show = true) {
    if (show) {
        loadingSpinner.classList.remove('hidden');
    } else {
        loadingSpinner.classList.add('hidden');
    }
}

function showToast(message, type = 'success', duration = 3000) {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, duration);
}

function updateCharCounter() {
    const count = articleInput.value.length;
    charCounter.textContent = count;
    
    if (count < 10) {
        predictBtn.disabled = true;
    } else {
        predictBtn.disabled = false;
    }
}

// ===========================
// MODEL STATUS & INFO
// ===========================

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        if (data.model_loaded) {
            statusBadge.className = 'status-badge ready';
            statusBadge.innerHTML = '<i class="fas fa-check-circle"></i> Model Ready';
        } else {
            statusBadge.className = 'status-badge error';
            statusBadge.innerHTML = '<i class="fas fa-exclamation-circle"></i> Model Not Loaded';
            disableAllInputs();
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        statusBadge.className = 'status-badge error';
        statusBadge.innerHTML = '<i class="fas fa-times-circle"></i> Connection Error';
        disableAllInputs();
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model-info`);
        if (!response.ok) return;
        
        const data = await response.json();
        clusterKeywords = data.clusters;
        
        // Update UI with cluster info
        document.getElementById('cluster-count').textContent = data.n_clusters;
        
        const clustersInfoDiv = document.querySelector('.cluster-info-card');
        if (clustersInfoDiv) {
            let html = '<h3><i class="fas fa-cube"></i> Cluster Categories</h3>';
            html += '<div class="clusters-list">';
            
            for (let i = 0; i < data.n_clusters; i++) {
                const keywords = data.clusters[i]?.top_keywords || [];
                html += `
                    <div class="cluster-tag">
                        <div class="cluster-tag-id">C${i}</div>
                        <div class="cluster-tag-keywords">${keywords.slice(0, 2).join(', ')}</div>
                    </div>
                `;
            }
            
            html += '</div>';
            clustersInfoDiv.innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

function disableAllInputs() {
    articleInput.disabled = true;
    batchInput.disabled = true;
    predictBtn.disabled = true;
    batchPredictBtn.disabled = true;
    sampleBtn.disabled = true;
    showToast('Model not loaded. Please train the model first.', 'error', 5000);
}

// ===========================
// PREDICTION HANDLERS
// ===========================

async function handlePredict() {
    const text = articleInput.value.trim();
    
    if (!text) {
        showToast('Please enter some text', 'error');
        return;
    }
    
    if (text.length < 10) {
        showToast('Text is too short (minimum 10 characters)', 'error');
        return;
    }
    
    if (text.length > 5000) {
        showToast('Text is too long (maximum 5000 characters)', 'error');
        return;
    }
    
    showLoading(true);
    predictBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayPredictionResult(result);
        showToast('Prediction successful!', 'success');
        
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Error during prediction: ' + error.message, 'error');
    } finally {
        showLoading(false);
        predictBtn.disabled = false;
    }
}

function displayPredictionResult(result) {
    const clusterId = result.cluster;
    const keywords = result.keywords || [];
    const confidence = result.confidence || 0;
    const preview = result.preview || '';
    
    // Update result display
    document.getElementById('predicted-cluster').textContent = clusterId;
    document.getElementById('cluster-label').textContent = `Article Category ${clusterId}`;
    
    // Update confidence bar
    const confidencePercent = Math.round(confidence * 100);
    document.getElementById('confidence-text').textContent = `${confidencePercent}%`;
    document.getElementById('confidence-fill').style.width = `${Math.min(confidencePercent, 100)}%`;
    
    // Update keywords
    const keywordsList = document.getElementById('keywords-list');
    keywordsList.innerHTML = keywords
        .map(keyword => `<span class="keyword-tag">${keyword}</span>`)
        .join('');
    
    // Update preview
    document.getElementById('text-preview').textContent = preview;
    
    // Show result container
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function handleBatchPredict() {
    const text = batchInput.value.trim();
    
    if (!text) {
        showToast('Please enter some articles', 'error');
        return;
    }
    
    // Split by newlines and filter
    const texts = text.split('\n')
        .map(t => t.trim())
        .filter(t => t.length > 0);
    
    if (texts.length === 0) {
        showToast('No valid articles entered', 'error');
        return;
    }
    
    if (texts.length > 50) {
        showToast('Maximum 50 articles at a time', 'error');
        return;
    }
    
    showLoading(true);
    batchPredictBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/predict-batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ texts })
        });
        
        if (!response.ok) {
            throw new Error('Batch prediction failed');
        }
        
        const data = await response.json();
        displayBatchResults(data.predictions);
        showToast(`Predicted ${data.count} articles successfully!`, 'success');
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        showToast('Error during batch prediction: ' + error.message, 'error');
    } finally {
        showLoading(false);
        batchPredictBtn.disabled = false;
    }
}

function displayBatchResults(predictions) {
    const batchItems = document.getElementById('batch-items');
    
    batchItems.innerHTML = predictions
        .map((pred, index) => `
            <div class="batch-item">
                <div class="batch-item-header">
                    <div class="batch-item-cluster">${pred.cluster}</div>
                    <div>
                        <strong>Article ${index + 1}</strong>
                        <div style="font-size: 0.85rem; color: var(--text-muted);">
                            Confidence: ${Math.round((pred.confidence || 0) * 100)}%
                        </div>
                    </div>
                </div>
                <div class="batch-item-text">${pred.preview}</div>
                <div style="margin-top: 12px; font-size: 0.8rem;">
                    <strong>Keywords:</strong> ${pred.keywords.slice(0, 3).join(', ')}
                </div>
            </div>
        `)
        .join('');
    
    batchResults.classList.remove('hidden');
    batchResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===========================
// ACTION HANDLERS
// ===========================

function handleClear() {
    articleInput.value = '';
    charCounter.textContent = '0';
    resultContainer.classList.add('hidden');
    predictBtn.disabled = true;
}

async function handleLoadSample() {
    showLoading(true);
    sampleBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/sample-predict`);
        
        if (!response.ok) {
            throw new Error('Failed to load samples');
        }
        
        const data = await response.json();
        
        if (data.predictions && data.predictions.length > 0) {
            // Display first sample in main input
            articleInput.value = data.predictions[0].preview;
            updateCharCounter();
            
            // Display all predictions
            displayBatchResults(data.predictions);
            
            showToast('Sample predictions loaded!', 'success');
        }
    } catch (error) {
        console.error('Error loading samples:', error);
        showToast('Error loading samples: ' + error.message, 'error');
    } finally {
        showLoading(false);
        sampleBtn.disabled = false;
    }
}

// ===========================
// PERIODIC STATUS CHECK
// ===========================

// Check model status every 30 seconds
setInterval(checkModelStatus, 30000);

console.log('✅ JavaScript loaded successfully');
