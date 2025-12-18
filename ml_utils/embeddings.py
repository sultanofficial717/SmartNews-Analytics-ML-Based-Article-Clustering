import requests
import numpy as np
import time

class OpenRouterEmbeddingGenerator:
    def __init__(self, api_key, model="openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/embeddings"

    def get_embedding(self, text):
        """Get embedding for a single text"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SmartNews-Analytics", # Required by OpenRouter
            "X-Title": "SmartNews Analytics" # Required by OpenRouter
        }
        
        # Truncate text if too long (approx 8000 tokens is usually the limit, but let's be safe with 4000 chars)
        if len(text) > 4000:
            text = text[:4000]
            
        payload = {
            "model": self.model,
            "input": text
        }
        
        print(f"    Sending request (len={len(text)})...")
        try:
            # Create a session for connection pooling if not already existing
            if not hasattr(self, 'session'):
                self.session = requests.Session()
                self.session.trust_env = False # Disable proxies
            
            response = self.session.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                return data['data'][0]['embedding']
            else:
                print(f"Error: No embedding data found in response: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching embedding (timeout/connection): {e}")
            return None
        except Exception as e:
            print(f"Error fetching embedding: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            return None

    def get_embeddings_batch(self, texts):
        """Get embeddings for a list of texts"""
        embeddings = []
        print(f"Fetching embeddings for {len(texts)} documents using {self.model}...")
        
        for i, text in enumerate(texts):
            print(f"  Processing document {i+1}/{len(texts)}...")
            emb = self.get_embedding(text)
            
            if emb is None:
                print(f"  Failed to get embedding for doc {i+1}. Aborting API method.")
                return None
                
            embeddings.append(emb)
            # Rate limiting niceness
            time.sleep(0.2)
            
        return np.array(embeddings)
