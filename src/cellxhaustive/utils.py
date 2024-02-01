"""
Functions shared between several scripts.
"""


# Import utility modules
import logging
import sys


# Custom logging format
class CustomFormatter(logging.Formatter):

    # Colors and format
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[1;31;20m"
    reset = "\x1b[0m"
    format = "[%(asctime)s]  %(levelname)-8s  %(message)s"
    datefmt = '%Y.%m.%d - %H:%M:%S'

    FORMATS = {logging.DEBUG: grey + format + reset,
               logging.INFO: green + format + reset,
               logging.WARNING: yellow + format + reset,
               logging.ERROR: red + format + reset,
               logging.CRITICAL: bold_red + format + reset}

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,
                                      datefmt='%Y.%m.%d - %H:%M:%S')
        return formatter.format(record)


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

    # Sets handlers with proper redirection and format
    # File handler
    fh_formatter = logging.Formatter(fmt='[%(asctime)s]  %(levelname)-8s  %(message)s',
                                     datefmt='%Y.%m.%d - %H:%M:%S')
    handler_fh = logging.FileHandler(logfile, mode='w')
    handler_fh.setFormatter(fh_formatter)
    # Stream handler
    handler_sh = logging.StreamHandler(sys.stdout)
    handler_sh.setFormatter(CustomFormatter())
    handlers = [handler_fh,  # Output logging in file
                handler_sh]  # Output logging in stdout

    # Set up logger level, format and handlers
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
