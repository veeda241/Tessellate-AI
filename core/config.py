import os
from pathlib import Path

# Build an absolute filesystem anchor relative to this configuration script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_DIR = DATA_DIR / "fixtures"
FRONTEND_DIR = PROJECT_ROOT / "ui"
BRIDGE_PATH = DATA_DIR / "live_state_bridge.json"

# Ensure essential output directories exist at initialization
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIXTURES_DIR, exist_ok=True)
