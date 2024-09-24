"""
    In this file Implemented the Logging functionality.
"""

import os
import sys
import logging

logging_str = "[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"

logging_directory = "../../logs"
log_filepath = os.path.join(logging_directory, "running_logs.log")
os.makedirs(logging_directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # handle the log folder and file
        logging.StreamHandler(sys.stdout) # will help to print the log in the terminal
    ]
)

logger = logging.getLogger("autoencoderlogger")