#!/usr/bin/env python3
"""
Complete startup script for Brain MRI Analysis Application
This script starts both the FastAPI backend and the React frontend
"""
import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🧠 BRAIN MRI ANALYSIS APPLICATION")
    print("=" * 60)
    print("🚀 Starting Full Application Stack...")
    print("📊 Backend: FastAPI (Port 8000)")
    print("🌐 Frontend: React + Vite (Port 12000)")
    print("=" * 60)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import torch
        import fastapi
        import uvicorn
        print("✅ Python dependencies found")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found")
        print("💡 Please install Node.js from https://nodejs.org/")
        return False
    
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("\n🔧 Starting Backend...")
    
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)
    
    try:
        # Start the backend server
        subprocess.run([
            sys.executable, "start_backend.py"
        ], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
        return False

def start_frontend():
    """Start the React frontend"""
    print("\n🔧 Starting Frontend...")
    
    frontend_dir = Path(__file__).parent / "project"
    os.chdir(frontend_dir)
    
    try:
        # Install dependencies if needed
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start the development server
        subprocess.run([
            "npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "12000"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
        return False

def start_backend_thread():
    """Start backend in a separate thread"""
    try:
        start_backend()
    except Exception as e:
        print(f"❌ Backend thread error: {e}")

def main():
    """Main function to start the application"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        sys.exit(1)
    
    print("\n🎯 Starting application components...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend_thread, daemon=True)
    backend_thread.start()
    
    # Wait a bit for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Start frontend (this will block)
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        print("\n👋 Shutting down application...")

if __name__ == "__main__":
    main()