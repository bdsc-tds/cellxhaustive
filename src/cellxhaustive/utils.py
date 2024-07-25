"""
Functions and classes shared between several scripts.
"""


# Import utility modules
import itertools as ite
import logging
import numpy as np
import sys
import yaml
from math import prod
import itertools as ite


# Functions and classes related to logging
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
def setup_log(log_file, log_level):
    """
    Function to set-up logging format, log file and log level.

    Parameters:
    -----------
    log_file: str
      Path to log file.

    log_file: str
      Verbosity level of log file.
    """

    # Parse log level
    if log_level.upper() == 'DEBUG':
        level = logging.DEBUG
    elif log_level.upper() == 'INFO':
        level = logging.INFO
    else:
        level = logging.WARNING

    # Get root logger
    root_log = logging.getLogger()

    # Set root logger level
    root_log.setLevel(level)

    # Create handlers with redirection
    f_handler = logging.FileHandler(log_file)
    s_handler = logging.StreamHandler(sys.stdout)

    # Set stream handlers format
    fh_formatter = logging.Formatter(fmt='[%(asctime)s]  [PID:%(process)9d]  %(filename)28s  %(levelname)-8s  %(message)s',
                                     datefmt='%Y.%m.%d - %H:%M:%S')
    f_handler.setFormatter(fh_formatter)  # Set handler format
    s_handler.setFormatter(CustomFormatter())  # Set handler format

    # Add handlers to root logger
    root_log.addHandler(f_handler)  # Output logging to file
    root_log.addHandler(s_handler)  # Output logging to stdout


# Functions and classes related to config file parsing
# Function to check status of 'detection_method' parameter
def get_detection_method(detection_method, offset):
    """
    Helper function to check status of 'detection_method' parameter.

    Parameters:
    -----------
    detection_method: str
      Value passed to 'detection_method' via command line or config file.

    offset: int
      Number of tabs to add in front of log messages.

    Returns:
    --------
    method: 'auto' or int
      Processed value of 'detection_method'.
    """

    try:  # Integer was provided
        method = int(detection_method)
        logging.info(f"{'\t' * offset}Limiting combinations to {method} markers")
    except ValueError:
        if detection_method == 'auto':  # Default setting was provided
            logging.info(f"{'\t' * offset}Using default algorithm")
            method = 'auto'
        else:
            logging.error(f"{'\t' * offset}Unknown detection method. Please provide 'auto' or an integer superior or equal to 2.")
            sys.exit(1)

    return method


# Function to parse config file and return parameter values
def parse_config(config_path, uniq_labels, markers_interest, detection_method):
    """
    Function to parse config file and return parameter values.

    Parameters:
    -----------
    config_path: str
      Path to config file.

    uniq_labels: array(str)
      Array containing all cell types present in the dataset.

    Returns:
    --------
    markers_interest_lst: list(array(str))
      List containing arrays of markers of interest for each cell type. If no
      markers of interest are provided, array of markers will be empty.

    detection_method_lst: list('auto' or int)
      List containing detection method for each cell type. Detection method can
      be 'auto' or an integer superior or equal to 2.
    """

    cell_pop_nb = len(uniq_labels)  # Get number of cell types
    try:
        with open(config_path) as config_input:
            config_dict = yaml.safe_load(config_input)
        if config_dict is None:
            logging.error('\tConfig file is empty')
            sys.exit(1)
    except FileNotFoundError:  # Without config file
        logging.warning('\tNo config file found, using CLI arguments')
        # Process markers of interest
        logging.info('\t\tChecking for markers of interest')
        if markers_interest:  # With markers in CLI
            markers_interest = np.array(markers_interest.split(','), dtype='str')
            logging.info(f"\t\t\tFound {len(markers_interest)} markers: {', '.join(markers_interest)}")
        else:  # # Without markers in CLI
            logging.warning('\t\t\tNo markers provided in CLI, using default empty array')
            markers_interest = np.empty(0, dtype='str')
        logging.info('\t\t\tPropagating this array to all cell types')
        markers_interest_lst = [markers_interest for _ in range(cell_pop_nb)]
        # Process method to decide final combination length
        logging.info('\t\tChecking for detection method')
        detection_method = get_detection_method(detection_method, 3)
        logging.info('\t\t\tPropagating this setting to all cell types')
        detection_method_lst = [detection_method] * cell_pop_nb
    else:  # With config file
        logging.info('\tFound config file. Parsing its content')
        # Process markers of interest
        logging.info('\t\tChecking for markers of interest')
        try:  # Information on markers of interest in config file
            markers_dict = config_dict['markers_interest']
            if markers_dict is None:  # User probably forgot to delete field, use default
                logging.warning("\t\t\t'markers_interest' field is present but empty. Using default empty array")
                markers_interest = np.empty(0, dtype='str')
                logging.info('\t\t\tPropagating this setting to all cell types')
                markers_interest_lst = [markers_interest for _ in range(cell_pop_nb)]
            elif 'all' in markers_dict.keys():
                if markers_dict['all'] is None:  # User probably forgot to delete field, use default
                    logging.warning("\t\t\t'markers_interest: all' field is present but empty. Using default empty array")
                    markers_interest = np.empty(0, dtype='str')
                else:
                    logging.info(f"\t\t\tFound global setting: {', '.join(markers_dict['all'])}")
                    markers_interest = np.array(markers_dict['all'])
                logging.info('\t\t\tPropagating this setting to all cell types')
                markers_interest_lst = [markers_interest for _ in range(cell_pop_nb)]
            else:
                logging.info('\t\t\tFound cell types specific settings')
                markers_interest_lst = []
                for label in uniq_labels:
                    if label in markers_dict.keys():
                        if markers_dict[label] is None:  # User probably forgot to delete field, use default
                            logging.warning(f"\t\t\t\t'markers_interest: all: {label}' field is present but empty. Using default empty array")
                            markers_interest_lst.append(np.empty(0, dtype='str'))
                        else:
                            logging.info(f"\t\t\t\tFound markers for cell type '{label}': {', '.join(markers_dict[label])}")
                            markers_interest_lst.append(np.array(markers_dict[label]))
                    else:
                        logging.warning(f"\t\t\t\tNo markers found for cell type '{label}'. Using default empty array")
                        markers_interest_lst.append(np.empty(0, dtype='str'))
        except KeyError:  # No information on markers of interest in config file
            logging.warning('\t\t\tNo markers settings provided, using default empty array')
            markers_interest = np.empty(0, dtype='str')
            logging.info('\t\t\tPropagating this array to all cell types')
            markers_interest_lst = [markers_interest for _ in range(cell_pop_nb)]
        # Process method to decide final combination length
        logging.info('\t\tChecking for detection method')
        try:  # Information on detection method in config file
            detection_dict = config_dict['detection_method']
            if detection_dict is None:  # User probably forgot to delete field, use default
                logging.warning("\t\t\t'detection_method' field is present but empty. Using default algorithm")
                detection_method_lst = ['auto'] * cell_pop_nb
            elif 'all' in detection_dict.keys():
                if detection_dict['all'] is None:  # User probably forgot to delete field, use default
                    logging.warning("\t\t\t'detection_method: all' field is present but empty. Using default algorithm")
                    detection_method = 'auto'
                else:
                    logging.info(f"\t\t\tFound global setting: {detection_dict['all']}")
                    detection_method = get_detection_method(detection_dict['all'], 4)
                logging.info('\t\t\tPropagating this setting to all cell types')
                detection_method_lst = [detection_method] * cell_pop_nb
            else:
                logging.info('\t\t\tFound cell types specific settings')
                detection_method_lst = []
                for label in uniq_labels:
                    if label in detection_dict.keys():
                        if detection_dict[label] is None:  # User probably forgot to delete field, use default
                            logging.warning(f"\t\t\t\t'detection_method: all: {label}' field is present but empty. Using default algorithm")
                            detection_method_lst.append('auto')
                        else:
                            logging.info(f"\t\t\t\tFound detection method for cell type '{label}': {detection_dict[label]}")
                            detection_method = get_detection_method(detection_dict[label], 5)
                            detection_method_lst.append(detection_method)
                    else:
                        logging.info(f"\t\t\t\tNo detection method found for cell type '{label}'. Using default algorithm")
                        detection_method_lst.append('auto')
                detection_method_lst = np.array(detection_method_lst)  # Convert list back to array
                del detection_method_lst
        except KeyError:  # No information on detection method in config file
            logging.warning('\t\t\tNo detection method settings provided, using default algorithm')
            detection_method = 'auto'
            logging.info('\t\t\tPropagating this setting to all cell types')
            detection_method_lst = [detection_method] * cell_pop_nb

    return markers_interest_lst, detection_method_lst
