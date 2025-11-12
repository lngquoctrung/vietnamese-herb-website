import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

root_dir = Path(__file__).parent.parent.absolute()

class Config:
    RAW_DATA_PATH = str(root_dir / "data" / "raw")
    PROCESSED_DATA_PATH = str(root_dir / "data" / "processed")
    IMAGE_DATA_PATH = str(root_dir / "data" / "images")
    LOG_PATH = str(root_dir / "logs")

    # Vector database configuration
    QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")