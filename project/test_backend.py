import requests
import os

# Test if the backend is running
try:
    response = requests.get('http://localhost:12000/healthz')
    print(f"Backend health check: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Could not connect to backend: {e}")
    print("Make sure the backend server is running on port 12000")