import os

from pydantic_settings import BaseSettings
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()

class Settings(BaseSettings):
    # Pathes
    RAW_DATA_PATH: str = str(root_dir / "data" / "raw")
    PROCESSED_DATA_PATH: str = str(root_dir / "data" / "processed")
    IMAGE_DATA_PATH: str = str(root_dir / "data" / "images")
    LOG_PATH: str = str(root_dir / "logs")
    OUTPUT_JSON: str = str(root_dir / "data" / "processed" / "thuoc_dong_y.json")
    OUTPUT_CSV_VI_THUOC: str = str(root_dir / "data" / "processed" / "vi_thuoc.csv")
    OUTPUT_CSV_BAI_THUOC: str = str(root_dir / "data" / "processed" / "bai_thuoc.csv")
    OUTPUT_CSV_CONG_THUC: str = str(root_dir / "data" / "processed" / "cong_thuc.csv")

    # Processing config
    PAGES_PER_CHUNK: int = 200
    OVERLAP_PAGES: int = 30
    PROCESS_PAGES_PER_REQUEST: int = 8
    PROCESS_OVERLAP_PAGES: int = 3

    # Model config
    GEMINI_API_KEY: str
    MODEL_NAME: str = "gemini-2.0-flash"
    MAX_OUTPUT_TOKENS: int = 16384

    # Request limitation
    REQUESTS_PER_MINUTE: int = 15
    DELAY_BETWEEN_REQUESTS: float = 4.5
    MAX_RETRIES: int = 5
    INITIAL_BACKOFF: int = 3
    BACKOFF_MULTIPLIER: float = 1.5
    MAX_BACKOFF: int = 120

    # Deduplication config
    SIMILARITY_THRESHOLD: float = 0.85
    USE_FUZZY_MATCHING: bool = True

    # Embedding config
    MAX_CHUNK_SIZE: int = 6000
    OVERLAP_SIZE: int = 600
    MIN_CHUNK_LENGTH: int = 150

    model_config = {
        "env_file": ".env", 
        "env_file_encoding": "utf-8" 
    }

settings = Settings()