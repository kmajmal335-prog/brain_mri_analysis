"""
Enhanced FastAPI Backend for Brain MRI Analysis
Integrates with trained .pth models for real-time MRI evaluation
"""

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from torchvision import transforms
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
import base64
import cv2

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
    description="Advanced API for brain tumor classification using deep learning models",
    version="2.0.0"
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
model_type = None

# Initialize device
def initialize_device():
    """Initialize and return the best available device"""
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for inference")
    return device

# Initialize device on startup
device = initialize_device()

# Enhanced MRI preprocessing functions
def preprocess_mri_image(image: Image.Image) -> Image.Image:
    """
    Enhanced preprocessing specifically for MRI brain images
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy for advanced processing
    img_array = np.array(image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # Noise reduction
    img_array = cv2.medianBlur(img_array, 3)
    
    # Convert back to PIL Image
    image = Image.fromarray(img_array)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Convert to RGB for model input
    image = image.convert('RGB')
    
    return image

def get_mri_transforms():
    """
    Get MRI-specific preprocessing transforms optimized for brain tumor detection
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalize using ImageNet statistics (works well for transfer learning)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_time_augmentation_transforms():
    """
    Get multiple transforms for test-time augmentation
    """
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

# Model loading functions
def load_model_from_path(model_path: str) -> tuple:
    """
    Load a trained model from .pth file
    Returns: (model, model_type, model_info)
    """
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Loading model from: {model_path}")
        
        # Determine model type and create model instance
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
            # Try to infer from state dict keys
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if any('model1' in key for key in state_dict.keys()):
                model_type = 'ensemble'
            else:
                model_type = 'single'
        
        # Create model instance
        if model_type == 'ensemble':
            model = TrainingEnsemble(num_classes=4)
            logger.info("Created AdvancedEnsemble model")
        else:
            model = ImprovedModel(num_classes=4)
            logger.info("Created ImprovedModel")
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Extract model info
        info = {
            'model_type': model_type,
            'num_classes': 4,
            'classes': class_names,
            'device': str(device),
            'loaded_at': datetime.now().isoformat(),
            'model_path': model_path
        }
        
        # Add training info if available
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'accuracy' in checkpoint:
            info['accuracy'] = checkpoint['accuracy']
        if 'loss' in checkpoint:
            info['loss'] = checkpoint['loss']
        
        logger.info(f"Model loaded successfully: {model_type}")
        return model, model_type, info
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict_single_image(image: Image.Image, use_tta: bool = False) -> Dict[str, Any]:
    """
    Predict tumor type for a single MRI image
    """
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        # Preprocess the image
        processed_image = preprocess_mri_image(image)
        
        if use_tta:
            # Test-time augmentation for better accuracy
            transforms_list = get_test_time_augmentation_transforms()
            predictions = []
            
            for transform in transforms_list:
                input_tensor = transform(processed_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predictions.append(probabilities.cpu().numpy())
            
            # Average predictions
            avg_predictions = np.mean(predictions, axis=0)
            probabilities = torch.from_numpy(avg_predictions)
            
        else:
            # Standard single prediction
            transform = get_mri_transforms()
            input_tensor = transform(processed_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
        
        # Get prediction results
        confidence_scores = probabilities[0].cpu().numpy()
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(confidence_scores[predicted_class_idx])
        
        # Create detailed results
        class_probabilities = {
            class_names[i]: float(confidence_scores[i]) 
            for i in range(len(class_names))
        }
        
        # Generate clinical interpretation
        clinical_info = generate_clinical_interpretation(predicted_class, confidence, class_probabilities)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'clinical_interpretation': clinical_info,
            'model_info': {
                'model_type': model_type,
                'device': str(device),
                'tta_used': use_tta
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def generate_clinical_interpretation(predicted_class: str, confidence: float, probabilities: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate clinical interpretation of the prediction
    """
    interpretation = {
        'primary_finding': predicted_class,
        'confidence_level': 'high' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'low',
        'recommendation': '',
        'differential_diagnosis': [],
        'urgency': 'routine'
    }
    
    # Sort probabilities for differential diagnosis
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    if predicted_class == 'glioma':
        interpretation['recommendation'] = 'Urgent neurosurgical consultation recommended. Consider MRI with contrast and possible biopsy.'
        interpretation['urgency'] = 'urgent'
        if probabilities['meningioma'] > 0.2:
            interpretation['differential_diagnosis'].append('meningioma')
    
    elif predicted_class == 'meningioma':
        interpretation['recommendation'] = 'Neurosurgical evaluation recommended. Monitor growth with serial imaging.'
        interpretation['urgency'] = 'semi-urgent'
        if probabilities['glioma'] > 0.2:
            interpretation['differential_diagnosis'].append('glioma')
    
    elif predicted_class == 'pituitary':
        interpretation['recommendation'] = 'Endocrinology and neurosurgery consultation. Assess hormonal function.'
        interpretation['urgency'] = 'semi-urgent'
    
    else:  # notumor
        interpretation['recommendation'] = 'No tumor detected. Continue routine monitoring if clinically indicated.'
        interpretation['urgency'] = 'routine'
        # Check if any tumor type has significant probability
        max_tumor_prob = max(probabilities['glioma'], probabilities['meningioma'], probabilities['pituitary'])
        if max_tumor_prob > 0.3:
            interpretation['recommendation'] += ' Consider follow-up imaging due to moderate probability of tumor presence.'
    
    return interpretation

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Brain MRI Analysis API - Enhanced Version",
        "version": "2.0.0",
        "description": "Advanced API for brain tumor classification with ML model integration",
        "endpoints": {
            "health": "GET /health - Check API health and model status",
            "load_model": "POST /load-model - Load a trained .pth model",
            "predict": "POST /predict - Analyze single MRI image",
            "batch_predict": "POST /batch-predict - Analyze multiple MRI images",
            "model_info": "GET /model-info - Get loaded model information",
            "classes": "GET /classes - Get available tumor classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_type": model_type if model is not None else None,
        "device": str(device),
        "classes": class_names
    }

@app.post("/load-model")
async def load_model(model_path: str = Form(...)):
    """Load a trained model from .pth file"""
    global model, model_type, model_info
    
    try:
        model, model_type, model_info = load_model_from_path(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        return {
            "status": "success",
            "message": f"Model loaded successfully",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_info": model_info,
        "status": "loaded",
        "device": str(device)
    }

@app.get("/classes")
async def get_classes():
    """Get available tumor classes"""
    return {
        "classes": class_names,
        "descriptions": {
            "glioma": "Malignant brain tumor arising from glial cells",
            "meningioma": "Usually benign tumor arising from meninges",
            "pituitary": "Tumor of the pituitary gland",
            "notumor": "No tumor detected - normal brain tissue"
        }
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    use_tta: bool = Form(False),
    return_image: bool = Form(False)
):
    """
    Analyze a single MRI image for brain tumor classification
    
    Parameters:
    - file: MRI image file (JPEG, PNG, etc.)
    - use_tta: Use test-time augmentation for better accuracy (slower)
    - return_image: Return processed image as base64
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first using /load-model endpoint")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get prediction
        prediction_result = predict_single_image(image, use_tta=use_tta)
        
        # Add file info
        prediction_result['file_info'] = {
            'filename': file.filename,
            'size': len(image_data),
            'format': image.format,
            'mode': image.mode,
            'dimensions': image.size
        }
        
        # Optionally return processed image
        if return_image:
            processed_image = preprocess_mri_image(image)
            buffer = io.BytesIO()
            processed_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            prediction_result['processed_image'] = f"data:image/png;base64,{img_base64}"
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    use_tta: bool = Form(False)
):
    """
    Analyze multiple MRI images for brain tumor classification
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first using /load-model endpoint")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                'filename': file.filename,
                'error': 'File must be an image'
            })
            continue
        
        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Get prediction
            prediction_result = predict_single_image(image, use_tta=use_tta)
            prediction_result['filename'] = file.filename
            
            results.append(prediction_result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'batch_results': results,
        'total_files': len(files),
        'successful_predictions': len([r for r in results if 'error' not in r])
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)