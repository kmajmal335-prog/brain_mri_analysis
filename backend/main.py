import os
import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from torchvision import transforms
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import base64

# Import your model classes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.classifier import BrainTumorClassifier
from train_model import ImprovedModel, AdvancedEnsemble as TrainingEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Brain MRI Analysis API",
    description="API for brain tumor classification using deep learning",
    version="1.0.0"
)

# CORS middleware for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and configuration
model = None
device = None
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
model_info = {}

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path: str = None):
    """
    Load the trained model from .pth file
    This function will be called when a model path is provided
    """
    global model, device, model_info
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if model_path and os.path.exists(model_path):
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Determine model type and create appropriate model
            if 'model_type' in checkpoint:
                model_type = checkpoint['model_type']
            else:
                # Try to infer from checkpoint keys
                if any('model1' in key for key in checkpoint.keys()):
                    model_type = 'ensemble'
                else:
                    model_type = 'single'
            
            # Create model based on type
            if model_type == 'ensemble':
                model = TrainingEnsemble(num_classes=4)
                logger.info("Loading ensemble model")
            else:
                model = ImprovedModel(num_classes=4)
                logger.info("Loading single model")
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            # Store model info
            model_info = {
                'model_type': model_type,
                'loaded_at': datetime.now().isoformat(),
                'model_path': model_path,
                'device': str(device),
                'num_classes': 4,
                'class_names': class_names
            }
            
            if 'epoch' in checkpoint:
                model_info['epoch'] = checkpoint['epoch']
            if 'best_val_acc' in checkpoint:
                model_info['accuracy'] = checkpoint['best_val_acc']
            
            logger.info(f"Model loaded successfully: {model_info}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    else:
        logger.warning("No model path provided or file doesn't exist")
        return False

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_image(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    global model, device
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            class_probs = probabilities[0].cpu().numpy()
            
            result = {
                'predicted_class': class_names[predicted.item()],
                'predicted_index': predicted.item(),
                'confidence': float(confidence.item()),
                'probabilities': {
                    class_names[i]: float(prob) for i, prob in enumerate(class_probs)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Brain MRI Analysis API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }

@app.post("/load-model")
async def load_model_endpoint(model_path: str):
    """
    Load model from specified path
    This endpoint allows you to load your trained .pth file
    """
    success = load_model(model_path)
    
    if success:
        return {
            "message": "Model loaded successfully",
            "model_info": model_info
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to load model")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return model_info

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor type from uploaded MRI image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please load a model first using /load-model endpoint")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        result = predict_image(image_tensor)
        
        # Add file info to result
        result['file_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(image_data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-base64")
async def predict_base64(data: dict):
    """
    Predict brain tumor type from base64 encoded image
    Expected format: {"image": "base64_string", "filename": "optional_filename"}
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please load a model first using /load-model endpoint")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        result = predict_image(image_tensor)
        
        # Add file info to result
        result['file_info'] = {
            'filename': data.get('filename', 'base64_image'),
            'size': len(image_data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get available class names"""
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Predict brain tumor types for multiple images
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please load a model first using /load-model endpoint")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'File must be an image'
                })
                continue
            
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            image_tensor = preprocess_image(image)
            
            # Make prediction
            result = predict_image(image_tensor)
            result['file_info'] = {
                'filename': file.filename,
                'content_type': file.content_type,
                'size': len(image_data)
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'results': results,
        'total_files': len(files),
        'successful_predictions': len([r for r in results if 'error' not in r])
    }

if __name__ == "__main__":
    import uvicorn
    
    # Try to load model if path is provided via environment variable
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        load_model(model_path)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )