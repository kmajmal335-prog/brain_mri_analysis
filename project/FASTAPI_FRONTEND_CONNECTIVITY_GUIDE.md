# Brain Tumor Detection - Integration Guide

This document explains how the FastAPI backend (`brain_mri-master`) is integrated with the React frontend (`project`).

## Integration Overview

The application consists of two main components:
1. **Backend**: FastAPI application in `brain_mri-master` directory
2. **Frontend**: React application in `project` directory

## How the Integration Works

### Backend (FastAPI)
- Runs on port `12000`
- Provides REST API endpoints under `/api` prefix
- Uses PyTorch model (`best_model.pth`) for brain tumor detection
- Stores uploaded scans and results in SQLite database (`mri.db`)
- Serves uploaded images from `/media` endpoint

### Frontend (React)
- Runs on port `5173` (Vite default)
- Communicates with backend via API calls
- Uses `apiService.js` for all backend communications
- Displays scan results and history

## Key Integration Points

### 1. API Service
The `src/utils/apiService.js` file handles all communication between frontend and backend:
- `uploadMRI(file)` - Uploads and processes MRI scans
- `getScans()` - Retrieves scan history
- `getScanById(scanId)` - Retrieves specific scan details
- `getAnalytics()` - Retrieves analytics data

### 2. API Endpoints
The backend provides these endpoints:
- `POST /api/upload` - Upload MRI scan
- `GET /api/scans` - Get all scans
- `GET /api/scans/{scan_id}` - Get specific scan
- `GET /api/analytics` - Get analytics data

### 3. Data Flow
1. User uploads MRI through frontend
2. Frontend sends file to `POST /api/upload`
3. Backend saves file, processes with ML model, and stores results
4. Backend returns processed results to frontend
5. Frontend displays results to user

## Running the Application

### Prerequisites
- Python 3.8+
- Node.js 16+
- pip (Python package manager)
- npm (Node package manager)

### Installation
1. Install backend dependencies:
   ```
   cd brain_mri-master
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```
   cd project
   npm install
   ```

### Starting the Application
Run the provided script:
```
cd project
start_all.bat
```

This will:
1. Start the FastAPI backend server
2. Start the React frontend development server
3. Open both applications in your browser

## Troubleshooting

### Backend Not Starting
- Check if port 12000 is already in use
- Verify `best_model.pth` exists in `brain_mri-master/models/`
- Check backend logs in `brain_mri-master/server.log`

### Frontend Not Connecting to Backend
- Verify backend is running on port 12000
- Check browser console for CORS errors
- Ensure API endpoints are accessible

### Model Loading Issues
- Verify `best_model.pth` file exists and is not corrupted
- Check file permissions
- Ensure PyTorch is properly installed