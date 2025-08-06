# FILE: nanovllm/utils/logging.py
import logging
import os

# Set up the logger based on an environment variable
# Example: export NANOVLLM_LOG_LEVEL=DEBUG
log_level_str = os.environ.get("NANOVLLM_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Create a logger
logger = logging.getLogger("nanovllm")
logger.setLevel(log_level)

# Avoid adding duplicate handlers if this module is imported multiple times
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
