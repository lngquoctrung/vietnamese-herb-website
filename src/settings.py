import os

from pydantic_settings import BaseSettings
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()

class Settings(BaseSettings):
    # Pathes
    RAW_DATA_PATH: str = str(root_dir / "data" / "raw")
    PROCESSED_DATA_PATH: str = str(root_dir / "data" / "processed")
    LOG_PATH: str = str(root_dir / "logs")
    OUTPUT_JSON: str = str(root_dir / "data" / "processed" / "thuoc_dong_y.json")
    OUTPUT_CSV_VI_THUOC: str = str(root_dir / "data" / "processed" / "vi_thuoc.csv")
    OUTPUT_CSV_BAI_THUOC: str = str(root_dir / "data" / "processed" / "bai_thuoc.csv")
    OUTPUT_CSV_CONG_THUC: str = str(root_dir / "data" / "processed" / "cong_thuc.csv")
    CHECKPOINT_FILE: str = str(root_dir / "data" / "processed" / "checkpoint.json")

    # Processing config
    PAGES_PER_CHUNK: int = 200
    OVERLAP_PAGES: int = 10
    PROCESS_PAGES_PER_REQUEST: int = 6
    PROCESS_OVERLAP_PAGES: int = 3

    # Model config
    GEMINI_API_KEY: str
    OCR_MODEL_NAME: str = "gemini-2.0-flash-lite"
    OCR_FALLBACK_MODEL: str = "gemini-2.0-flash"
    TEXT_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # Request limitation
    REQUESTS_PER_MINUTE: int = 8
    DELAY_BETWEEN_REQUESTS: float = 10.0
    MAX_RETRIES: int = 5
    INITIAL_BACKOFF: int = 10.0
    BACKOFF_MULTIPLIER: float = 2.0
    MAX_BACKOFF: int = 120

    # OCR config
    MIN_OCR_LENGTH: int = 1000
    RETRY_WITH_FALLBACK: bool = True
    MAX_OUTPUT_TOKENS: int = 8192

    # Deduplication config
    SIMILARITY_THRESHOLD: float = 0.85
    USE_FUZZY_MATCHING: bool = True

    # Deduplication
    USE_FUZZY_MATCHING: bool = True
    SIMILARITY_THRESHOLD: float = 0.85

    model_config = {
        "env_file": ".env", 
        "env_file_encoding": "utf-8" 
    }

settings = Settings()