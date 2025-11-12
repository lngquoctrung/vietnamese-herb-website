import logging
import sys
import os

def get_logger(name: str, log_filepath: str = None, level=logging.INFO):
    # Initialize logger
    logger = logging.getLogger(name=name)

    # Return existing logger
    if logger.handlers:
        return logger
    
    # Logger level
    logger.setLevel(level=level)
    # Logger format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level=level)
    console_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=console_handler)

    # File handler
    if log_filepath:
        # Create logging directory
        os.makedirs(
            name=os.path.dirname(p=log_filepath), 
            exist_ok=True
        )
        file_handler = logging.FileHandler(filename=log_filepath)
        file_handler.setLevel(level=level)
        file_handler.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=file_handler)
    
    # Disable duplicate logging
    logger.propagate = False

    return logger
