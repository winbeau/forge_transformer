from pathlib import Path

# pwd forge_transformer/src/forge_transformer/utils/path.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
