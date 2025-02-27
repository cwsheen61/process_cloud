import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(log_file, console_logging=False):
    """
    Sets up a rotating file logger with optional console logging.
    
    Args:
        log_file (str): Path to the log file.
        console_logging (bool): If True, also logs to the console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("PLYProcessor")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplicate logs

    # Rotating file handler
    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Optional console logging
    if console_logging:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
