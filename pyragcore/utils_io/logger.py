# pyragcore/utils_io/logger.py
import logging

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def set_up_logging(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )