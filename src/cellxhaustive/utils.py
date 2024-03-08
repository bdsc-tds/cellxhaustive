"""
Functions shared between several scripts.
"""


# Import utility modules
import logging
import multiprocessing
import multiprocessing.pool
import sys


# Log-related functions and classes
# Custom logging format
class CustomFormatter(logging.Formatter):

    # Colors and format
    grey = '\x1b[38;20m'
    green = '\x1b[32;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[1;31;20m'
    reset = '\x1b[0m'
    format = '[%(asctime)s]  %(filename)28s  %(levelname)-8s  %(message)s'
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


# Function to configurate logging, used in cellxhaustive.py
def setup_log(logfile, loglevel):
    """
    Function to set-up logging format, log file and log level.

    Parameters:
    -----------
    logfile: str
      Path to log file.

    nb_cell_type: str
      Verbosity level of log file.
    """

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
    fh_formatter = logging.Formatter(fmt='[%(asctime)s]  %(filename)28s  %(levelname)-8s  %(message)s',
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


# Multiprocessing-related classes
# Custom multiprocessing process without daemons
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


# Custom multiprocessing context
class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# Sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool because
# latter is only a wrapper function, not a proper class
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
