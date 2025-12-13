#!/usr/bin/env python3
"""
Installation and setup script for the News Clustering ML Project
Run this script after downloading the project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class ProjectSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.os_type = platform.system()
        
    def print_header(self):
        """Print welcome header"""
        print("\n" + "="*70)
        print("📰 NEWS ARTICLE CLUSTERING ML PROJECT")
        print("Installation & Setup Script")
        print("="*70 + "\n")
    
    def print_status(self, message, status="ℹ"):
        """Print status message"""
        print(f"{status} {message}")
    
    def check_python_version(self):
        """Check if Python version is 3.8+"""
        self.print_status("Checking Python version...")
        version = sys.version_info
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_status(f"Python {version.major}.{version.minor} detected", "✗")
            self.print_status("Python 3.8 or higher is required!", "⚠")
            return False
        
        self.print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "✓")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        self.print_status("Creating directories...")
        
        directories = [
            'data',
            'models',
            'uploads',
            'logs',
            'app/templates',
            'app/static/css',
            'app/static/js',
            'app/static/images',
            'ml_utils'
        ]
        
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
        
        self.print_status("Directories created ✓", "✓")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_status("Installing dependencies...")
        self.print_status("This may take 2-3 minutes...", "⏳")
        
        try:
            requirements = self.project_root / 'requirements.txt'
            if requirements.exists():
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                    cwd=str(self.project_root)
                )
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-r', str(requirements)],
                    cwd=str(self.project_root)
                )
                self.print_status("Dependencies installed ✓", "✓")
                return True
            else:
                self.print_status("requirements.txt not found", "✗")
                return False
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error installing dependencies: {e}", "✗")
            return False
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        self.print_status("Downloading NLTK data...")
        
        try:
            import nltk
            
            required_data = [
                ('tokenizers', 'punkt'),
                ('corpora', 'stopwords'),
                ('corpora', 'wordnet'),
                ('taggers', 'averaged_perceptron_tagger'),
            ]
            
            for data_type, data_name in required_data:
                try:
                    if data_type == 'tokenizers':
                        nltk.data.find(f'{data_type}/{data_name}')
                    else:
                        nltk.data.find(f'{data_type}/{data_name}')
                except LookupError:
                    self.print_status(f"  Downloading {data_name}...", "⏳")
                    nltk.download(data_name, quiet=True)
            
            # Download punkt_tab
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            
            self.print_status("NLTK data ready ✓", "✓")
            return True
        except Exception as e:
            self.print_status(f"Warning: Could not download all NLTK data: {e}", "⚠")
            self.print_status("You can manually run: python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger punkt_tab", "⚠")
            return True  # Non-critical error
    
    def copy_data_files(self):
        """Copy data files to data folder"""
        self.print_status("Organizing data files...")
        
        try:
            # Copy CSV
            csv_file = self.project_root / 'list.csv'
            data_dir = self.project_root / 'data'
            
            if csv_file.exists():
                import shutil
                shutil.copy(csv_file, data_dir / 'list.csv')
            
            # Copy TXT files
            for txt_file in self.project_root.glob('*.txt'):
                if txt_file.is_file():
                    import shutil
                    shutil.copy(txt_file, data_dir / txt_file.name)
            
            self.print_status("Data files organized ✓", "✓")
            return True
        except Exception as e:
            self.print_status(f"Warning: Could not organize data files: {e}", "⚠")
            return True  # Non-critical
    
    def verify_structure(self):
        """Verify project structure"""
        self.print_status("Verifying project structure...")
        
        required_files = [
            'train.py',
            'run.py',
            'requirements.txt',
            'README.md',
            'app/app.py',
            'app/templates/index.html',
            'app/static/css/style.css',
            'app/static/js/main.js',
            'ml_utils/__init__.py',
            'ml_utils/preprocessor.py',
            'ml_utils/clustering.py',
            'ml_utils/predictor.py',
        ]
        
        missing = []
        for file in required_files:
            path = self.project_root / file
            if not path.exists():
                missing.append(file)
        
        if missing:
            self.print_status(f"Missing files: {', '.join(missing)}", "✗")
            return False
        
        self.print_status("Project structure verified ✓", "✓")
        return True
    
    def print_next_steps(self):
        """Print next steps"""
        print("\n" + "="*70)
        print("🎉 SETUP COMPLETE!")
        print("="*70 + "\n")
        
        print("📋 NEXT STEPS:\n")
        print("1. TRAIN THE MODEL:")
        print("   python train.py")
        print()
        print("2. START THE WEB APP:")
        print("   python run.py run")
        print("   or")
        print("   cd app && python app.py")
        print()
        print("3. OPEN YOUR BROWSER:")
        print("   http://localhost:5000")
        print()
        print("═"*70)
        print("📚 DOCUMENTATION:")
        print("   • README.md - Full documentation")
        print("   • QUICKSTART.md - 5-minute quick start")
        print("   • API_DOCUMENTATION.md - API reference")
        print("   • TROUBLESHOOTING.md - Common issues")
        print("   • GITHUB_SETUP.md - GitHub repository setup")
        print()
        print("💡 TIPS:")
        print("   • Ensure data files are in data/ folder")
        print("   • Training takes 1-2 minutes for first time")
        print("   • Use --clusters flag to train with different cluster count")
        print("   • Check console output for any errors")
        print()
        print("🚀 READY TO DEPLOY:")
        print("   • Docker: docker-compose up")
        print("   • GitHub: See GITHUB_SETUP.md")
        print("   • Cloud: Check README.md for deployment options")
        print()
        print("="*70 + "\n")
    
    def run(self):
        """Run complete setup"""
        self.print_header()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        print()
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        print()
        
        # Download NLTK data
        self.download_nltk_data()
        
        print()
        
        # Copy data files
        self.copy_data_files()
        
        print()
        
        # Verify structure
        if not self.verify_structure():
            return False
        
        print()
        
        # Print next steps
        self.print_next_steps()
        
        return True


def main():
    """Main entry point"""
    setup = ProjectSetup()
    
    try:
        success = setup.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
