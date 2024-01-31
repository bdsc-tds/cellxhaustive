"""
Functions shared between several scripts.
"""


# Import utility modules
import logging
import sys


# Function to configurate logging
def setup_log(logfile):

    # Reset logging handler to avoid conflicts
    logging.root.handlers = []

    # Set up logger level, format and handlers
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]  %(levelname)-8s  %(message)s",
        datefmt='%Y.%m.%d - %H:%M:%S',
        handlers=[logging.FileHandler(logfile, mode='w'),  # Output logging in file
                  logging.StreamHandler(sys.stdout)]  # Output logging in stdout
    )
