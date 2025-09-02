#!/usr/bin/env python3
"""
Create a demo model for testing the ML integration
This creates a simple trained model that can be used to test the API
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
from train_model import ImprovedModel, AdvancedEnsemble

def create_demo_model():
    """Create a demo model with random weights for testing"""
    print("🔧 Creating demo model for testing...")
    
    # Create models directory
    models_dir = Path("/workspace/project/brain_mri_analysis/models")
    models_dir.mkdir(exist_ok=True)
    
    # Create a simple model
    model = ImprovedModel(num_classes=4)
    
    # Create demo checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'single',
        'epoch': 50,
        'accuracy': 0.95,
        'loss': 0.15,
        'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'created_for': 'demo_testing'
    }
    
    # Save demo model
    model_path = models_dir / "demo_model.pth"
    torch.save(checkpoint, model_path)
    
    print(f"✅ Demo model created at: {model_path}")
    print(f"   Model type: {checkpoint['model_type']}")
    print(f"   Classes: {checkpoint['classes']}")
    print(f"   Demo accuracy: {checkpoint['accuracy']}")
    
    return str(model_path)

def create_demo_ensemble_model():
    """Create a demo ensemble model for testing"""
    print("🔧 Creating demo ensemble model for testing...")
    
    # Create models directory
    models_dir = Path("/workspace/project/brain_mri_analysis/models")
    models_dir.mkdir(exist_ok=True)
    
    # Create ensemble model
    model = AdvancedEnsemble(num_classes=4)
    
    # Create demo checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'ensemble',
        'epoch': 100,
        'accuracy': 0.98,
        'loss': 0.08,
        'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'created_for': 'demo_testing'
    }
    
    # Save demo ensemble model
    model_path = models_dir / "demo_ensemble_model.pth"
    torch.save(checkpoint, model_path)
    
    print(f"✅ Demo ensemble model created at: {model_path}")
    print(f"   Model type: {checkpoint['model_type']}")
    print(f"   Classes: {checkpoint['classes']}")
    print(f"   Demo accuracy: {checkpoint['accuracy']}")
    
    return str(model_path)

if __name__ == "__main__":
    print("🧠 CREATING DEMO MODELS FOR TESTING")
    print("=" * 50)
    
    # Create both types of demo models
    single_model_path = create_demo_model()
    ensemble_model_path = create_demo_ensemble_model()
    
    print("\n✅ Demo models created successfully!")
    print("\n📋 Next steps:")
    print("1. Test single model:")
    print(f"   curl -X POST http://localhost:8000/load-model -d 'model_path={single_model_path}'")
    print("\n2. Test ensemble model:")
    print(f"   curl -X POST http://localhost:8000/load-model -d 'model_path={ensemble_model_path}'")
    print("\n3. Run full integration test:")
    print("   python test_ml_integration.py")
    print("\n4. Upload MRI images via the web interface!")