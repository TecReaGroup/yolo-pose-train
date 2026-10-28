"""Project-local resource locations, independent of the working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / 'data' / 'dataset'
ANNOTATION_DIR = PROJECT_ROOT / 'data' / 'annotation'
VIDEO_DIR = PROJECT_ROOT / 'data' / 'video'
MODEL_DIR = PROJECT_ROOT / 'data' / 'model'
WEIGHT_DIR = PROJECT_ROOT / 'data' / 'weight'
TEMP_DIR = PROJECT_ROOT / 'temp'
RUN_DIR = TEMP_DIR / 'run'
LOG_DIR = PROJECT_ROOT / 'log'
