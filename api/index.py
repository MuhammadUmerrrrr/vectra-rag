import sys
import os

# Add backend directory to sys.path to allow imports to work
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_dir)

# Import the FastAPI app from backend/main.py
from main import app

# Configure the app root path for routing under Vercel
app.root_path = "/api"

# Expose a root route under the /api prefix
@app.get("/")
def api_root():
    return {"status": "ok", "message": "Vectra API is running"}
