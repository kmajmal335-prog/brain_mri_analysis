# Brain Tumor Detection Application

This is a full-stack application for detecting brain tumors in MRI scans using a PyTorch model.

## Project Structure

- `brain_mri-master/` - FastAPI backend with PyTorch model
- `project/` - React frontend

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd brain_mri-master
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. The PyTorch model file (`best_model.pth`) should be in the `models/` directory.

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd project
   ```

2. Install Node.js dependencies:
   ```
   npm install
   ```

## Running the Application

### Option 1: Using the start script (Windows)

1. Navigate to the project directory:
   ```
   cd project
   ```

2. Run the start script:
   ```
   start_all.bat
   ```

### Option 2: Manual start

1. Start the backend server:
   ```
   cd brain_mri-master
   python manage.py
   ```

2. In a new terminal, start the frontend:
   ```
   cd project
   npm run dev
   ```

## API Endpoints

The backend provides the following REST API endpoints:

1. `POST /api/upload` - Upload an MRI scan
2. `GET /api/scans` - Get scan history
3. `GET /api/scans/{scan_id}` - Get scan details
4. `GET /api/analytics` - Get analytics data

## Accessing the Application

Once both servers are running:
- Frontend: http://localhost:5173
- Backend API: http://localhost:12000/api
- Backend health check: http://localhost:12000/healthz