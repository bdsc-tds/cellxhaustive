"""
Functions shared between several scripts.
"""


# Import utility modules
import logging
import sys


# Function to configurate logging
def setup_log(logfile, loglevel):

    # Reset logging handler to avoid conflicts
    logging.root.handlers = []

    if loglevel.upper() == 'DEBUG':
        level = logging.DEBUG
    elif loglevel.upper() == 'INFO':
        level = logging.INFO
    else:
        level = logging.WARNING

    # Set up logger level, format and handlers
    logging.basicConfig(
        level=level,
        format="[%(asctime)s]  %(levelname)-8s  %(message)s",
        datefmt='%Y.%m.%d - %H:%M:%S',
        handlers=[logging.FileHandler(logfile, mode='w'),  # Output logging in file
                  logging.StreamHandler(sys.stdout)]  # Output logging in stdout
    )
