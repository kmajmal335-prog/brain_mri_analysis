#!/usr/bin/env python3
"""
Integration test script for Brain MRI Analysis Application
Tests the FastAPI backend endpoints and functionality
"""
import requests
import json
import time
import os
from pathlib import Path

class IntegrationTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def print_test(self, test_name, status="RUNNING"):
        """Print test status"""
        status_emoji = {
            "RUNNING": "🔄",
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️"
        }
        print(f"{status_emoji.get(status, '❓')} {test_name}")
        
    def test_health_check(self):
        """Test health check endpoint"""
        self.print_test("Health Check", "RUNNING")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Model Loaded: {data.get('model_loaded')}")
                print(f"   Device: {data.get('device')}")
                self.print_test("Health Check", "PASS")
                return True
            else:
                self.print_test("Health Check", "FAIL")
                print(f"   Status Code: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Health Check", "FAIL")
            print(f"   Error: {e}")
            return False
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        self.print_test("Root Endpoint", "RUNNING")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"   Message: {data.get('message')}")
                print(f"   Version: {data.get('version')}")
                self.print_test("Root Endpoint", "PASS")
                return True
            else:
                self.print_test("Root Endpoint", "FAIL")
                return False
        except Exception as e:
            self.print_test("Root Endpoint", "FAIL")
            print(f"   Error: {e}")
            return False
    
    def test_classes_endpoint(self):
        """Test classes endpoint"""
        self.print_test("Classes Endpoint", "RUNNING")
        try:
            response = self.session.get(f"{self.base_url}/classes")
            if response.status_code == 200:
                data = response.json()
                print(f"   Classes: {data.get('classes')}")
                print(f"   Number of Classes: {data.get('num_classes')}")
                self.print_test("Classes Endpoint", "PASS")
                return True
            else:
                self.print_test("Classes Endpoint", "FAIL")
                return False
        except Exception as e:
            self.print_test("Classes Endpoint", "FAIL")
            print(f"   Error: {e}")
            return False
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        self.print_test("Model Info Endpoint", "RUNNING")
        try:
            response = self.session.get(f"{self.base_url}/model-info")
            if response.status_code == 200:
                data = response.json()
                print(f"   Model Type: {data.get('model_type')}")
                print(f"   Device: {data.get('device')}")
                print(f"   Accuracy: {data.get('accuracy', 'N/A')}")
                self.print_test("Model Info Endpoint", "PASS")
                return True
            elif response.status_code == 404:
                print("   No model loaded (expected if no model is loaded)")
                self.print_test("Model Info Endpoint", "PASS")
                return True
            else:
                self.print_test("Model Info Endpoint", "FAIL")
                return False
        except Exception as e:
            self.print_test("Model Info Endpoint", "FAIL")
            print(f"   Error: {e}")
            return False
    
    def test_load_model_endpoint(self, model_path=None):
        """Test model loading endpoint"""
        if not model_path:
            self.print_test("Load Model Endpoint", "SKIP")
            print("   No model path provided")
            return True
            
        self.print_test("Load Model Endpoint", "RUNNING")
        try:
            response = self.session.post(
                f"{self.base_url}/load-model",
                params={"model_path": model_path}
            )
            if response.status_code == 200:
                data = response.json()
                print(f"   Message: {data.get('message')}")
                print(f"   Model Info: {data.get('model_info', {}).get('model_type')}")
                self.print_test("Load Model Endpoint", "PASS")
                return True
            else:
                self.print_test("Load Model Endpoint", "FAIL")
                print(f"   Status Code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            self.print_test("Load Model Endpoint", "FAIL")
            print(f"   Error: {e}")
            return False
    
    def test_predict_endpoint(self, image_path=None):
        """Test prediction endpoint"""
        if not image_path or not os.path.exists(image_path):
            self.print_test("Predict Endpoint", "SKIP")
            print("   No valid image path provided")
            return True
            
        self.print_test("Predict Endpoint", "RUNNING")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Predicted Class: {data.get('predicted_class')}")
                print(f"   Confidence: {data.get('confidence', 0):.2%}")
                print(f"   File: {data.get('file_info', {}).get('filename')}")
                self.print_test("Predict Endpoint", "PASS")
                return True
            else:
                self.print_test("Predict Endpoint", "FAIL")
                print(f"   Status Code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            self.print_test("Predict Endpoint", "FAIL")
            print(f"   Error: {e}")
            return False
    
    def create_test_image(self):
        """Create a simple test image for testing"""
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            test_path = "test_image.jpg"
            test_image.save(test_path)
            return test_path
        except ImportError:
            print("   PIL not available, skipping test image creation")
            return None
        except Exception as e:
            print(f"   Error creating test image: {e}")
            return None
    
    def cleanup_test_files(self):
        """Clean up test files"""
        test_files = ["test_image.jpg"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def run_all_tests(self, model_path=None, image_path=None):
        """Run all integration tests"""
        print("🧪 BRAIN MRI ANALYSIS - INTEGRATION TESTS")
        print("=" * 50)
        
        # Check if backend is running
        print("🔍 Checking backend availability...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            print(f"✅ Backend is running at {self.base_url}")
        except Exception as e:
            print(f"❌ Backend not available at {self.base_url}")
            print(f"   Error: {e}")
            print("   Please start the backend first: python backend/start_backend.py")
            return False
        
        print("\n🧪 Running Tests...")
        print("-" * 30)
        
        # Run tests
        results = []
        results.append(self.test_root_endpoint())
        results.append(self.test_health_check())
        results.append(self.test_classes_endpoint())
        results.append(self.test_model_info_endpoint())
        
        # Test model loading if path provided
        if model_path:
            results.append(self.test_load_model_endpoint(model_path))
        
        # Test prediction
        test_image_path = image_path or self.create_test_image()
        if test_image_path:
            results.append(self.test_predict_endpoint(test_image_path))
            if not image_path:  # Clean up only if we created the test image
                self.cleanup_test_files()
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("-" * 30)
        passed = sum(results)
        total = len(results)
        print(f"✅ Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check the output above.")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Brain MRI Analysis Integration")
    parser.add_argument("--backend-url", default="http://localhost:8000", 
                       help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--model-path", help="Path to model file for testing")
    parser.add_argument("--image-path", help="Path to test image file")
    
    args = parser.parse_args()
    
    tester = IntegrationTester(args.backend_url)
    success = tester.run_all_tests(args.model_path, args.image_path)
    
    if success:
        print("\n🚀 Integration test completed successfully!")
        print("   You can now use the application at http://localhost:12000")
    else:
        print("\n❌ Integration test failed!")
        print("   Please check the errors above and fix them before proceeding.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())