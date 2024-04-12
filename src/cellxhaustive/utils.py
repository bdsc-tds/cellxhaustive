"""
Functions shared between several scripts.
"""


# Import utility modules
import itertools as ite
import logging
import multiprocessing
import multiprocessing.pool
import sys
from math import prod


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
    format = '[%(asctime)s]  [PID:%(process)9d]  %(filename)28s  %(levelname)-8s  %(message)s'
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


# Function to configurate logging; used in cellxhaustive.py
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
    fh_formatter = logging.Formatter(fmt='[%(asctime)s]  [PID:%(process)9d]  %(filename)28s  %(levelname)-8s  %(message)s',
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


# Multiprocessing-related functions and classes
# Function to distribute CPUs across functions; used in cellxhaustive.py
def get_cpu(nb_cpu, nb_cell_type):
    """
    Function to split provided cores across the different tasks.

    Parameters:
    -----------
    nb_cpu: int
      Total number of cores provided to workflow.

    nb_cell_type: int
      Number of unique cell types in input_dataset.

    Returns:
    --------
    nb_cpu_id: int
      Number of cores dedicated to 'identify_phenotypes' (from cellxhaustive.py).

    nb_cpu_eval: int
      Number of cores dedicated to 'evaluate_comb' (from check_all_combinations.py).

    nb_cpu_keep: int
      Number of cores dedicated to 'keep_relevant_phntp' and 'get_marker_status'
      (from score_marker_combinations.py and determine_marker_status.py, respectively).
    """

    # Create possible CPU amounts
    if nb_cell_type == 1:  # Can't multiprocess by cell type, so increase nb_cpu_keep
        cpu_id = [1]
        cpu_eval = range(1, 9)
        cpu_keep = range(1, 9)

    else:
        cpu_id = range(1, nb_cell_type + 1)
        cpu_eval = range(1, 9)
        cpu_keep = range(1, 3)

    # Create all possible CPU combinations
    cpu_comb = list(ite.product(cpu_id, cpu_eval, cpu_keep))

    # Compute all products
    results = []
    for item in cpu_comb:
        results.append(nb_cpu - prod(item))  # Subtract CPU given by user

    # Find positive minimum
    min_diff = min(n for n in results if n >= 0)

    # Find minimum indices
    min_idx = [i for i, x in enumerate(results) if x == min_diff]

    # Get list of solution
    cpu_solutions = [cpu_comb[i] for i in min_idx]

    # Sort by nb_cpu_eval then nb_cpu_id then nb_cpu_keep
    cpu_solutions = sorted(cpu_solutions, key=lambda x: (-x[1], -x[0], -x[2]))

    # Extract CPU values
    nb_cpu_id, nb_cpu_eval, nb_cpu_keep = cpu_solutions[0]

    logging.info(f'\tSetting nb_cpu_id to {nb_cpu_id}, nb_cpu_eval to {nb_cpu_eval}, and nb_cpu_keep to {nb_cpu_keep}')

    if min_diff == 0:
        logging.info('\tThere will be no idle CPU.')

    else:
        cpu_tot = prod(cpu_solutions[0])
        logging.warning(f'\tThere will be {min_diff} idle CPUs. Consider decreasing')
        logging.warning(f"\t'-t' parameter to {cpu_tot} to save resources or increase")
        logging.warning(f"\tby 1 to speed up analyses")

    return nb_cpu_id, nb_cpu_eval, nb_cpu_keep


# Function to determine chunksize to split an iterable; used in cellxhaustive.py,
# check_all_combinations.py, score_marker_combinations.py and determine_marker_status.py
def get_chunksize(iterable, nb_cpu):
    """
    In parallelised computing methods such as 'map()', it is often faster to split
    an iterable into a number of chunks which are submitted to the process pool
    as separate tasks. This function aims to determine the chunksize. It is
    identical to the one used in multiprocessing package.

    Parameters:
    -----------
    iterable: iter()
      Iterable to split for parallel-computing.

    nb_cpu: int
      Number of cores provided to process 'iterable'.

    Returns:
    --------
    chunksize: int
      Value of the chunksize parameter in 'map()' functions.
    """
    chunksize, extra = divmod(len(iterable), nb_cpu * 4)
    if extra:
        chunksize += 1
    return chunksize
