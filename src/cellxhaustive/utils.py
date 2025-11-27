"""
Functions and classes shared between several scripts.
"""

# Import utility modules
import logging
import sys
from collections import Counter


# Functions and classes related to logging
# Custom logging format; used in setup_log()
class CustomFormatter(logging.Formatter):
    # Colors and format
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[1;31;20m"
    reset = "\x1b[0m"
    format = "[%(asctime)s]  [PID:%(process)9d]  %(filename)28s  %(levelname)-8s  %(message)s"
    datefmt = "%Y.%m.%d - %H:%M:%S"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y.%m.%d - %H:%M:%S")
        return formatter.format(record)


# Function to configurate logging; used in cellxhaustive.py
def setup_log(log_file, log_level):
    """
    Function to set-up logging format, log file and log level.

    Parameters:
    -----------
    log_file: str
      Path to log file.

    log_file: str
      Verbosity level of log file.

    Returns:
    --------
      None
    """

    # Parse log level
    if log_level.upper() == "DEBUG":
        level = logging.DEBUG
    elif log_level.upper() == "INFO":
        level = logging.INFO
    else:
        level = logging.WARNING

    # Get root logger
    root_log = logging.getLogger()

    # Set root logger level
    root_log.setLevel(level)

    # Create handlers with redirection
    f_handler = logging.FileHandler(log_file, mode="w")
    s_handler = logging.StreamHandler(sys.stdout)

    # Set stream handlers format
    fh_formatter = logging.Formatter(
        fmt="[%(asctime)s]  [PID:%(process)9d]  %(filename)28s  %(levelname)-8s  %(message)s",
        datefmt="%Y.%m.%d - %H:%M:%S",
    )
    f_handler.setFormatter(fh_formatter)  # Set handler format
    s_handler.setFormatter(CustomFormatter())  # Set handler format

    # Add handlers to root logger
    root_log.addHandler(f_handler)  # Output logging to file
    root_log.addHandler(s_handler)  # Output logging to stdout


# Function to find representative markers; used in cellxhaustive.py
def get_repr_markers(markers_rep_batches, nb_batch):
    """
    Function that finds markers shared between a specific number of batches.

    Parameters:
    -----------
    markers_rep_batches: list(str)
      List containing representative markers of every batch.

    nb_batch: int
      Number of batches to consider during marker selection.

    Returns:
    --------
    markers_representative: set(str)
      Set of markers present in 'nb_batch' batches.

    missing_markers: set(str)
      Set of markers present in less than 'nb_batch' batches.

    nb_batch: int
      Number of batches that was finally considered during marker selection.
    """

    # Count markers using Counter
    marker_counts = Counter(markers_rep_batches)

    # Select markers present in 'nb_batch' batches
    markers_representative = {
        mk for mk, count in marker_counts.items() if count == nb_batch
    }

    # Check if representative markers were found
    if markers_representative:  # At least one representative marker found
        mk_rep_str = ", ".join(sorted(markers_representative))
        logging.info(
            f"\t\t\tFound {len(markers_representative)} markers: {mk_rep_str}"
        )
        # Find missing markers
        missing_markers = {
            mk for mk, count in marker_counts.items() if count < nb_batch
        }
        str1 = "s" if len(missing_markers) > 1 else ""
        missing_mk_str = ", ".join(sorted(missing_markers))
        logging.info(
            f"\t\t\tFiltered out {len(missing_markers)} marker{str1}: {missing_mk_str}"
        )
        return markers_representative, missing_markers, nb_batch
    else:  # No representative marker found: retry with one less batch
        nb_batch -= 1
        logging.info(
            f"\t\t\tNo markers found. Retrying with {nb_batch} batches"
        )
        return get_repr_markers(markers_rep_batches, nb_batch)
