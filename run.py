#!/usr/bin/env python
"""
Main entry point for the News Clustering application
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_flask_app():
    """Run the Flask application"""
    os.chdir(str(PROJECT_ROOT / 'app'))
    from app import app
    
    print("\n" + "="*70)
    print("🚀 NEWS ARTICLE CLUSTERING SYSTEM")
    print("="*70)
    print("\n📱 Server running on: http://localhost:5000")
    print("💡 Press CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

def train_model():
    """Train the clustering model"""
    os.chdir(str(PROJECT_ROOT))
    from train import train_clustering_pipeline
    train_clustering_pipeline()

def show_help():
    """Show help message"""
    print("\n" + "="*70)
    print("NEWS CLUSTERING ML PROJECT - COMMANDS")
    print("="*70 + "\n")
    print("Usage: python run.py [COMMAND]\n")
    print("Commands:")
    print("  run       - Start the Flask web application (default)")
    print("  train     - Train the clustering model")
    print("  help      - Show this help message\n")
    print("Examples:")
    print("  python run.py run           # Start web app")
    print("  python run.py train         # Train model with 5 clusters")
    print("  python run.py train --clusters 7  # Train with 7 clusters\n")
    print("="*70 + "\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'train':
            # Forward remaining arguments to train script
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            train_model()
        elif command == 'run':
            run_flask_app()
        elif command in ['help', '-h', '--help']:
            show_help()
        else:
            print(f"\n✗ Unknown command: {command}\n")
            show_help()
            sys.exit(1)
    else:
        # Default: run Flask app
        run_flask_app()
