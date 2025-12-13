"""
Setup script to initialize the project
Run this after cloning the repository
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        BASE_DIR / 'data',
        BASE_DIR / 'models',
        BASE_DIR / 'uploads',
        BASE_DIR / 'logs'
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"   ✓ {directory}")


def create_env_file():
    """Create .env file from .env.example"""
    print("\n📝 Setting up environment file...")
    env_example = BASE_DIR / '.env.example'
    env_file = BASE_DIR / '.env'
    
    if env_example.exists() and not env_file.exists():
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print(f"   ✓ Created .env file")
    elif env_file.exists():
        print(f"   ✓ .env file already exists")


def install_requirements():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    requirements_file = BASE_DIR / 'requirements.txt'
    
    if requirements_file.exists():
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                cwd=str(BASE_DIR)
            )
            print("   ✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Error installing dependencies: {e}")
            return False
    else:
        print(f"   ✗ requirements.txt not found")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    print("\n📥 Downloading NLTK data...")
    
    try:
        import nltk
        
        required_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'punkt_tab'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}') if data in ['punkt', 'punkt_tab'] else nltk.data.find(f'corpora/{data}')
            except LookupError:
                print(f"   Downloading {data}...")
                nltk.download(data, quiet=True)
        
        print("   ✓ NLTK data ready")
        return True
    except Exception as e:
        print(f"   ⚠ Error downloading NLTK data: {e}")
        return False


def print_completion_message():
    """Print setup completion message"""
    print("\n" + "="*70)
    print("✅ PROJECT SETUP COMPLETE!")
    print("="*70)
    print("\n📋 Next steps:\n")
    print("1. Place your data files in the 'data/' directory:")
    print("   - CSV file: data/list.csv")
    print("   - Text files: data/*.txt")
    print()
    print("2. Train the model:")
    print("   python train.py")
    print()
    print("3. Start the Flask app:")
    print("   cd app")
    print("   python app.py")
    print()
    print("4. Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n" + "="*70 + "\n")


def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("🚀 NEWS CLUSTERING ML PROJECT - SETUP")
    print("="*70 + "\n")
    
    create_directories()
    create_env_file()
    
    # Optional: Install requirements (can be done manually)
    print("\n💡 To complete setup, you can run:")
    print("   pip install -r requirements.txt")
    print("   python train.py")
    
    print_completion_message()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Setup error: {e}")
        sys.exit(1)
