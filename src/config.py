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

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.0-flash"

    REQUESTS_PER_MINUTE = 8
    DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE + 2

    MAX_CHUNK_SIZE = 6000
    OVERLAP_SIZE = 600
    MIN_CHUNK_LENGTH = 150

    MAX_RETRIES = 3
    INITIAL_BACKOFF = 5
    MAX_OUTPUT_TOKENS = 8192

    OUTPUT_JSON = str(root_dir / "data" / "processed" / "thuoc_dong_y.json")
    OUTPUT_CSV_VI_THUOC = str(root_dir / "data" / "processed" / "vi_thuoc.csv")
    OUTPUT_CSV_BAI_THUOC = str(root_dir / "data" / "processed" / "bai_thuoc.csv")
    OUTPUT_CSV_CONG_THUC = str(root_dir / "data" / "processed" / "cong_thuc.csv")
