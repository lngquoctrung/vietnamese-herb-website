import logging
import sys
import os
from pathlib import Path

def get_logger(name: str, filepath: str = None, level=logging.INFO):
    logger = logging.getLogger(name=name)

    if logger.handlers:
        return logger

    logger.setLevel(level=level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level=level)
    console_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=console_handler)

    if filepath:
        os.makedirs(
            name=os.path.dirname(p=filepath),
            exist_ok=True
        )

        file_handler = logging.FileHandler(filename=filepath)
        file_handler.setLevel(level=level)
        file_handler.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=file_handler)

    logger.propagate = False
    return logger

def safe_path(path):
    root_path = Path(__file__).resolve().parent.parent
    relative_path = os.path.relpath(path, root_path)
    relative_parts = list(Path(relative_path).parts)

    if ".." in relative_parts:
        dot_idx = len(relative_parts) - 1 - relative_parts[::-1].index("..")
        return "/".join(relative_parts[dot_idx:])

    return "/".join(relative_parts)

def validate_json_structure(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"vi_thuoc": [], "bai_thuoc": [], "cong_thuc": []}

    data.setdefault('vi_thuoc', [])
    data.setdefault('bai_thuoc', [])
    data.setdefault('cong_thuc', [])

    filtered_herbs = []
    for herb in data['vi_thuoc']:
        if herb.get('ten_viet_nam') and herb.get('ten_khoa_hoc'):
            filtered_herbs.append(herb)
    data['vi_thuoc'] = filtered_herbs

    return data
