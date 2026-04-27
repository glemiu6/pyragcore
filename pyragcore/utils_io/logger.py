import logging

def set_up_logging(level=logging.INFO):
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_logger(name):
    return logging.getLogger(name)