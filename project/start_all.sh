#!/bin/bash
# Script to start both frontend and backend servers

echo "Starting Brain Tumor Detection Application..."

# Start the Django backend server in the background
echo "Starting Django backend server..."
cd backend
python manage.py runserver > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Start the React frontend server
echo "Starting React frontend server..."
npm run dev

# When the script is terminated, kill the backend server
trap "kill $BACKEND_PID" EXIT