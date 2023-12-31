# --- Imports ---
from pathlib import Path
import os


# --- Constants ---
PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
TRANSFORMED_DATA_DIR = DATA_DIR / 'transformed'
MODELS_DIR = PARENT_DIR / 'models'


# --- Functions ---
def create_dir_if_not_exists(directory: Path) -> None:
    """Creates the specified directory if it doesn't exist."""
    if not directory.exists():
        os.mkdir(directory)


# --- Directory Creation ---
create_dir_if_not_exists(DATA_DIR)
create_dir_if_not_exists(RAW_DATA_DIR)
create_dir_if_not_exists(TRANSFORMED_DATA_DIR)
create_dir_if_not_exists(MODELS_DIR)
