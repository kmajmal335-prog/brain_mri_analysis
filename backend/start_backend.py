#!/usr/bin/env python3
"""
Backend startup script for Brain MRI Analysis API
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

def main():
    """Start the FastAPI backend server"""
    
    # Set environment variables
    os.environ.setdefault('PYTHONPATH', str(parent_dir))
    
    print("🚀 Starting Brain MRI Analysis Backend...")
    print(f"📁 Working directory: {current_dir}")
    print(f"🐍 Python path: {sys.path}")
    
    # Check for model path
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        print(f"🧠 Model path set: {model_path}")
    else:
        print("⚠️  No MODEL_PATH environment variable set")
        print("   You can load a model later using the /load-model endpoint")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir), str(parent_dir)],
        log_level="info"
    )

if __name__ == "__main__":
    main()