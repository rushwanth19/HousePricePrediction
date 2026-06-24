"""Central configuration for the House Price Prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

for _d in (MODELS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)
