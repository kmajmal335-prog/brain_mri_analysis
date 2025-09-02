#!/usr/bin/env python3
"""
Test script for ML Model Integration with FastAPI Backend
Tests the complete pipeline: model loading -> image upload -> prediction
"""

import requests
import json
import os
from pathlib import Path
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_MODEL_PATH = "/workspace/project/brain_mri_analysis/models/demo_model.pth"  # Update this path
TEST_IMAGE_PATH = "/workspace/project/brain_mri_analysis/data/test_image.jpg"  # Update this path

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
            print(f"   Classes: {data['classes']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🔄 Testing Model Loading...")
    
    # Check if model file exists
    if not os.path.exists(TEST_MODEL_PATH):
        print(f"⚠️  Model file not found at: {TEST_MODEL_PATH}")
        print("   Please train a model first using: python train_model.py")
        return False
    
    try:
        # Load model
        data = {"model_path": TEST_MODEL_PATH}
        response = requests.post(f"{API_BASE_URL}/load-model", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model loaded successfully")
            print(f"   Model Type: {result['model_info']['model_type']}")
            print(f"   Device: {result['model_info']['device']}")
            print(f"   Classes: {result['model_info']['classes']}")
            if 'accuracy' in result['model_info']:
                print(f"   Training Accuracy: {result['model_info']['accuracy']:.4f}")
            return True
        else:
            print(f"❌ Model loading failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n📊 Testing Model Info...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model Type: {data['model_info']['model_type']}")
            print(f"   Device: {data['device']}")
            print(f"   Loaded At: {data['model_info']['loaded_at']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def create_test_image():
    """Create a test MRI image if none exists"""
    print("\n🖼️  Creating test MRI image...")
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a synthetic MRI-like image
        img = Image.new('L', (224, 224), color=50)  # Dark background
        draw = ImageDraw.Draw(img)
        
        # Draw brain-like structure
        draw.ellipse([30, 30, 194, 194], fill=120, outline=150)  # Brain outline
        draw.ellipse([60, 60, 164, 164], fill=80, outline=100)   # Inner structure
        draw.ellipse([90, 90, 134, 134], fill=160, outline=180)  # Bright region (potential tumor)
        
        # Add some noise for realism
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save test image
        os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
        img.save(TEST_IMAGE_PATH)
        print(f"✅ Test image created at: {TEST_IMAGE_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return False

def test_image_prediction():
    """Test image prediction functionality"""
    print("\n🧠 Testing MRI Image Prediction...")
    
    # Check if test image exists, create if not
    if not os.path.exists(TEST_IMAGE_PATH):
        if not create_test_image():
            return False
    
    try:
        # Test single image prediction
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            data = {
                'use_tta': 'false',  # Set to 'true' for better accuracy but slower
                'return_image': 'true'
            }
            
            print("   Sending image for prediction...")
            response = requests.post(f"{API_BASE_URL}/predict", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Predicted Class: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Class Probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"     {class_name}: {prob:.4f}")
            
            # Clinical interpretation
            clinical = result['clinical_interpretation']
            print(f"   Clinical Interpretation:")
            print(f"     Primary Finding: {clinical['primary_finding']}")
            print(f"     Confidence Level: {clinical['confidence_level']}")
            print(f"     Recommendation: {clinical['recommendation']}")
            print(f"     Urgency: {clinical['urgency']}")
            
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction functionality"""
    print("\n📚 Testing Batch Prediction...")
    
    # Create multiple test images if needed
    test_images = []
    for i in range(2):
        img_path = f"/workspace/project/brain_mri_analysis/data/test_image_{i}.jpg"
        if not os.path.exists(img_path):
            create_test_image()
            # Rename the created image
            os.rename(TEST_IMAGE_PATH, img_path)
        test_images.append(img_path)
    
    try:
        files = []
        for img_path in test_images:
            files.append(('files', (os.path.basename(img_path), open(img_path, 'rb'), 'image/jpeg')))
        
        data = {'use_tta': 'false'}
        
        print("   Sending batch of images for prediction...")
        response = requests.post(f"{API_BASE_URL}/batch-predict", files=files, data=data)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"   Total Files: {result['total_files']}")
            print(f"   Successful Predictions: {result['successful_predictions']}")
            
            for i, prediction in enumerate(result['batch_results']):
                if 'error' not in prediction:
                    print(f"   Image {i+1}: {prediction['predicted_class']} (confidence: {prediction['confidence']:.4f})")
                else:
                    print(f"   Image {i+1}: Error - {prediction['error']}")
            
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 BRAIN MRI ANALYSIS - ML MODEL INTEGRATION TESTS")
    print("=" * 60)
    
    # Test sequence
    tests = [
        ("API Health Check", test_api_health),
        ("Model Loading", test_model_loading),
        ("Model Info", test_model_info),
        ("Single Image Prediction", test_image_prediction),
        ("Batch Image Prediction", test_batch_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your ML model integration is working perfectly!")
        print("\n📋 Next Steps:")
        print("1. Train your model: python train_model.py")
        print("2. Update TEST_MODEL_PATH in this script to your .pth file")
        print("3. Start uploading real MRI images for analysis!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        
        if not os.path.exists(TEST_MODEL_PATH):
            print("\n💡 Quick Fix:")
            print("   Train a model first: python train_model.py")
            print("   This will create the .pth file needed for testing.")

if __name__ == "__main__":
    main()