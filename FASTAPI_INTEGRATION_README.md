# Brain MRI Analysis - FastAPI Integration Guide

This guide explains how to set up and use the complete Brain MRI Analysis application with FastAPI backend and React frontend integration.

## 🏗️ Architecture Overview

```
Brain MRI Analysis Application
├── 📁 backend/                 # FastAPI Backend
│   ├── main.py                # Main FastAPI application
│   ├── start_backend.py       # Backend startup script
│   └── requirements.txt       # Backend dependencies
├── 📁 project/                # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ModelLoader.jsx    # Model loading interface
│   │   │   └── ...               # Other components
│   │   └── utils/
│   │       └── apiService.js     # API service for backend communication
│   └── package.json
├── 📁 src/                    # Model source code
│   ├── models/               # Model architectures
│   └── ...
├── train_model.py            # Model training script
└── start_full_application.py # Complete application launcher
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd project
npm install
cd ..
```

### 2. Train Your Model (if not already done)

```bash
# Train the model - this will create a .pth file
python train_model.py
```

### 3. Start the Complete Application

```bash
# Option 1: Start everything at once
python start_full_application.py

# Option 2: Start components separately
# Terminal 1 - Backend
cd backend
python start_backend.py

# Terminal 2 - Frontend
cd project
npm run dev -- --host 0.0.0.0 --port 12000
```

### 4. Access the Application

- **Frontend**: http://localhost:12000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🧠 Model Loading

### Step 1: Load Your Trained Model

1. Open the application in your browser
2. In the "Model Configuration" section, enter the path to your trained `.pth` file
3. Click "Load Model"

Example model paths:
```
/workspace/project/brain_mri_analysis/models/best_model.pth
/path/to/your/trained_model.pth
```

### Step 2: Verify Model Loading

Once loaded, you'll see:
- ✅ Model type (single/ensemble)
- 🖥️ Device (CPU/CUDA)
- 📊 Accuracy (if available)
- 🏷️ Supported classes: glioma, meningioma, notumor, pituitary

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and info |
| `/health` | GET | Health check |
| `/load-model` | POST | Load trained model |
| `/model-info` | GET | Get loaded model information |
| `/predict` | POST | Single image prediction |
| `/batch-predict` | POST | Multiple image predictions |
| `/classes` | GET | Get available classes |

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Load model
response = requests.post(
    "http://localhost:8000/load-model",
    params={"model_path": "/path/to/model.pth"}
)

# Make prediction
with open("mri_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## 🔧 Configuration

### Backend Configuration

The backend can be configured via environment variables:

```bash
# Set model path (optional)
export MODEL_PATH="/path/to/your/model.pth"

# Start backend
python backend/start_backend.py
```

### Frontend Configuration

Update the API base URL in `project/src/utils/apiService.js`:

```javascript
constructor() {
  this.baseURL = 'http://localhost:8000';  // Change if needed
}
```

## 🧪 Testing the Integration

### 1. Backend Testing

```bash
# Test backend directly
cd backend
python -c "
import requests
try:
    r = requests.get('http://localhost:8000/health')
    print('Backend Status:', r.json())
except:
    print('Backend not running')
"
```

### 2. Frontend Testing

1. Open browser to http://localhost:12000
2. Check "Model Configuration" section shows backend status
3. Load a model using the interface
4. Upload an MRI image for prediction

### 3. End-to-End Testing

```bash
# Use the test script
python project/test_backend.py
```

## 🔍 Troubleshooting

### Common Issues

#### Backend Not Starting
```bash
# Check if port 8000 is available
lsof -i :8000

# Check Python dependencies
pip install -r requirements.txt
```

#### Frontend Not Connecting
- Verify backend is running on port 8000
- Check browser console for CORS errors
- Ensure API base URL is correct

#### Model Loading Fails
- Verify model file path exists
- Check model file is a valid PyTorch .pth file
- Ensure model architecture matches training code

#### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Debug Mode

Start backend in debug mode:
```bash
cd backend
python main.py  # Direct execution with reload=True
```

## 📊 Model Training Integration

### Training Script Integration

The training script (`train_model.py`) automatically saves models that can be loaded by the backend:

```python
# Model is saved with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'model_type': 'ensemble' if use_ensemble else 'single',
    'epoch': epoch,
    'best_val_acc': best_val_acc,
    'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary']
}, model_path)
```

### Custom Model Loading

To load custom models, ensure they follow this structure:

```python
# Your model should be saveable as:
checkpoint = {
    'model_state_dict': your_model.state_dict(),
    'model_type': 'single',  # or 'ensemble'
    'epoch': training_epoch,
    'best_val_acc': validation_accuracy
}
torch.save(checkpoint, 'your_model.pth')
```

## 🔒 Production Deployment

### Security Considerations

1. **CORS Configuration**: Update CORS settings for production
2. **API Authentication**: Add authentication if needed
3. **File Upload Limits**: Configure appropriate file size limits
4. **HTTPS**: Use HTTPS in production

### Production Configuration

```python
# backend/main.py - Production settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📈 Performance Optimization

### Backend Optimization
- Use GPU for inference when available
- Implement model caching
- Add request batching for multiple predictions

### Frontend Optimization
- Implement image compression before upload
- Add progress indicators for long operations
- Cache API responses where appropriate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the integration
5. Submit a pull request

## 📄 License

This project is for research and educational purposes.

---

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Check console logs for error messages
4. Ensure model files are accessible and valid

For additional help, please check the project documentation or create an issue in the repository.