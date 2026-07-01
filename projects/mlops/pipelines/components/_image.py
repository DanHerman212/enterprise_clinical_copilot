"""
Resolved training image URI for pipeline components.
Import this instead of hardcoding the image in each file.
"""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ID

TRAINING_IMAGE = f"us-east1-docker.pkg.dev/{PROJECT_ID}/readmission/training:latest"
