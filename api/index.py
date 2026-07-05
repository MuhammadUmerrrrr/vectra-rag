import sys
import os

# Add backend directory to sys.path to allow imports to work
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_dir)

# Import the FastAPI app from backend/main.py
from main import app
