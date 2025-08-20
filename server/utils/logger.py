import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str,
    log_file: str = 'ivr_system.log',
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024, # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Sets up a logger with console and file handlers.
    The file handler uses RotatingFileHandler to manage log file size.

    Args:
        name (str): The name of the logger (e.g., 'stt_logger', 'llm_logger').
                    This helps identify log messages from different modules.
        log_file (str): The name of the log file. Defaults to 'ivr_system.log'.
        level (int): The minimum logging level to capture (e.g., logging.INFO,
                     logging.DEBUG, logging.ERROR). Defaults to logging.INFO.
        max_bytes (int): The maximum size of the log file before rotation.
                         Defaults to 10 MB.
        backup_count (int): The number of backup log files to keep. Defaults to 5.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if the logger is already configured
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Ensure log directory exists
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_file)

        # Create file handler and set level
        # RotatingFileHandler rotates logs when they reach a certain size
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Example usage (for demonstration purposes, not part of the utility itself)
# To use this, you would import setup_logger in your modules:
"""
# In stt_module.py:
from ..utils.logger import setup_logger

stt_logger = setup_logger('STTModule', level=logging.DEBUG)

class STTModule(Runnable):
    async def invoke(self, audio_data: bytes) -> str:
        stt_logger.info("Starting audio transcription.")
        try:
            # ... Faster Whisper transcription logic ...
            transcribed_text = "Hello, world." # Placeholder
            stt_logger.debug(f"Transcribed text: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            stt_logger.error(f"Error during transcription: {e}", exc_info=True)
            raise
"""
