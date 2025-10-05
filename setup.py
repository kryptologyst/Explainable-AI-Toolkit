#!/usr/bin/env python3
"""
Setup script for Modern XAI Toolkit
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["examples", "data", "outputs", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import shap
        import lime
        import sklearn
        import xgboost as xgb
        import lightgbm as lgb
        import torch
        import gradio as gr
        import streamlit as st
        
        print("✅ All packages imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Modern XAI Toolkit")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run 'python modern_xai.py' for command-line demo")
    print("2. Run 'python xai_web_app.py' for Gradio web interface")
    print("3. Run 'streamlit run streamlit_app.py' for Streamlit dashboard")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
