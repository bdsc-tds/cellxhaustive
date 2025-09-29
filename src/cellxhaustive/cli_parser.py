"""
Functions to parse CLI arguments and configuration files.
"""


# Import utility modules
import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import yaml
from math import floor


# Import other function from package
from utils import setup_log  # AT. Double-check path
# from cellxhaustive.utils import setup_log


# Function to create a default parser; used in cellxhaustive.py
def get_default_parser():
    """
    Function to initialize a parser with default argument values.

    Parameters:
    -----------
    No parameters.

    Returns:
    --------
    parser: argparse.ArgumentParser object
      Parser object with default argument values.
    """

    # Initialise parser
    parser = argparse.ArgumentParser(description='Package to annotate cell types using \
                                     CITE-seq ADT data.')

    # Add parser arguments with defaults
    parser.add_argument('-i', '--input', dest='input_path', type=str,
                        help='Path to input table with expression data and \
                        samples/batch/cell_type information',
                        required=True)
    parser.add_argument('-m', '--markers', dest='marker_path', type=str,
                        help='Path to file with list of markers to consider during \
                        analyses',
                        required=True)
    parser.add_argument('-o', '--output', dest='output_path', type=str,
                        help='Path to output table with annotations',
                        required=True)
    parser.add_argument('-a', '--max-markers', dest='max_markers', type=int,
                        help="Maximum number of markers to select among total list \
                        of markers. Must be an integer less than or equal to number \
                        of markers in total list provided with '-m' parameter [15]",
                        required=False, default=15)
    parser.add_argument('-mi', '--markers-interest', dest='markers_interest', type=str,
                        help='Comma-separated list of markers of interest that will \
                        appear in final combination. Global setting that applies to \
                        all cell types []',
                        required=False, default='')
    parser.add_argument('-dm', '--detection-method', dest='detection_method', type=str,
                        help="Method to identify length of best marker combination. \
                        Must be 'auto' to use default algorithm or an integer to set \
                        combination length. Integer must be less than '-a' parameter. \
                        Global setting that applies to all cell types [auto]",
                        required=False, default='auto')
    parser.add_argument('-b', '--config', dest='config_path', type=str,
                        help="Path to config file with cell type-specific \
                        detection method and markers of interest []",
                        required=False, default='')
    parser.add_argument('-c', '--cell-type-definition', dest='cell_type_path', type=str,
                        help='Path to file with cell types characterisation \
                        [../data/config/major_cell_types.yaml]',
                        required=False, default='../data/config/major_cell_types.yaml')
    parser.add_argument('-l', '--log', dest='log_path', type=str,
                        help='Path to log file [output_path.log]',
                        required=False, default='')
    parser.add_argument('-n', '--log-level', dest='log_level', type=str,
                        help="Verbosity level of log file. Must be 'debug', 'info' \
                        or 'warning' [info]",
                        required=False, default='info', choices=['debug', 'info', 'warning'])
    parser.add_argument('-e', '--three-peaks', dest='three_peak_markers', type=str,
                        help='Path to file with markers with three peaks or comma \
                        separated list of markers with three peaks []',
                        required=False, default='')
    parser.add_argument('-j', '--thresholds', dest='thresholds', type=str,
                        help='Comma-separated list of 3 floats defining thresholds \
                        to determine whether marker are negative or positive. 1st \
                        number is for two peaks markers, last 2 numbers are for \
                        three peaks markers [3,2,4]',
                        required=False, default='3,2,4')
    parser.add_argument('-q', '--min-samplesxbatch', dest='min_samplesxbatch', type=float,
                        help="Minimum proportion of samples within each batch with at \
                        least 'min_cellxsample' cells for a new annotation to be \
                        considered. Must be a float in [0.01; 1] [0.5]",
                        required=False, default=0.5)
    parser.add_argument('-r', '--min-cellxsample', dest='min_cellxsample', type=int,
                        help="Minimum number of cells within each sample in \
                        'min_samplesxbatch' %% of samples within each batch for a new \
                        annotation to be considered. Must be an integer in [1; 100] [10]",
                        required=False, default=10)
    parser.add_argument('-p', '--knn-min-probability', dest='knn_min_probability', type=float,
                        help='Confidence threshold for KNN classifier to reassign a \
                        new cell type to previously undefined cells. Must be a float \
                        in [0; 1] [0.5]',
                        required=False, default=0.5)
    parser.add_argument('-t', '--threads', dest='cores', type=int,
                        help='Number of cores to use. Specifying more than one core \
                        will run parallel jobs which will increase speed. Must be \
                        a strictly positive integer [1]',
                        required=False, default=1)
    parser.add_argument('-d', '--dry-run', dest='dryrun',
                        help='Use dry-run mode to check input files and configuration [False]',
                        required=False, default=False, action="store_true")

    return parser


# Function to check status of 'detection_method' parameter; used in parse_config_file()
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
        logging.info('{offset}Limiting combinations to {method} markers'.format(offset='\t' * offset, method=method))
    except ValueError:
        if detection_method == 'auto':  # Default setting was provided
            logging.info('{offset}Using default algorithm'.format(offset='\t' * offset))
            method = 'auto'
        else:
            logging.error("{offset}Unknown detection method. Please provide 'auto' or an integer superior or equal to 2".format(offset='\t' * offset))
            sys.exit(1)

    return method


# Function to parse config file and return parameter values; used in cellxhaustive.py
def parse_config_file(config_path, uniq_labels, markers_interest, detection_method):
    """
    Function to parse config file and return values for markers of interest and
    detection method.

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
            logging.error('\tConfig file is empty. Plese fill it or do not use it in the command line')
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
        except KeyError:  # No information on detection method in config file
            logging.warning('\t\t\tNo detection method settings provided, using default algorithm')
            detection_method = 'auto'
            logging.info('\t\t\tPropagating this setting to all cell types')
            detection_method_lst = [detection_method] * cell_pop_nb

    return markers_interest_lst, detection_method_lst


# Function to validate CLI arguments; used in cellxhaustive.py
def validate_cli(args):
    """
    Function to validate CLI arguments and parse config file if one is provided.

    Parameters:
    --------
    args: argparse.Namespace object
      Namespace containing arguments provided by user (as well as default ones)
      and their respective values.

    Returns:
    --------
    args_dict: dict({str: obj})
      Dictionary with argument values and objects (int, str, arrays) created by
      processing CLI arguments.
    """

    # Set log level and create log filename from output filename if not provided
    if not args.log_path:
        args.log_path = f'{os.path.splitext(args.output_path)[0]}.log'

    # Create log directory if not empty and missing
    log_dir = os.path.dirname(args.log_path)
    if log_dir and not os.path.exists(log_dir):
        logging.debug(f"Creating log directory '{log_dir}'")
        os.makedirs(log_dir, exist_ok=True)

    # Set-up logging configuration
    setup_log(args.log_path, args.log_level)

    # Get 1-D markers array
    logging.info(f"Importing marker list from '{args.marker_path}'")
    try:
        markers = pd.read_csv(args.marker_path, sep='\t', header=None).to_numpy(dtype=str).flatten()
    except FileNotFoundError:
        logging.error(f"\tCould not find '{args.marker_path}'. Please double-check file path")
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f'\tFound {len(markers)} markers')

    # Import input file and extract data of interest in arrays
    logging.info(f"Importing cell data from '{args.input_path}'")
    try:
        input_table = pd.read_csv(args.input_path, sep='\t', header=0, index_col=0)
    except FileNotFoundError:
        logging.error(f"\tCould not find '{args.input_path}'. Please double-check file path")
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f'\tFound {len(input_table.index)} cells')

    # Get 2-D expression array using 'markers'
    logging.info(f"Selecting ADT counts in '{args.input_path}'")
    try:
        mat = input_table.loc[:, markers].to_numpy(dtype=float)
    except KeyError as e:
        logging.error(f"\tCould not find marker '{e}' in '{args.input_path}'. Please double-check marker list")
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)

    # Get 1-D batches array; add common batch value if information is missing
    input_col_low = input_table.columns.str.lower()  # Make column names case-insensitive
    logging.info(f"Retrieving batch information in '{args.input_path}'")
    try:
        batches_idx = input_col_low.get_loc('batch')
    except KeyError:
        logging.warning(f"\tNo batch information in '{args.input_path}'")
        logging.warning("\tSetting batch value to 'batch0' for all cells")
        batches = np.full(input_table.shape[0], 'batch0')
    else:
        batches = input_table.iloc[:, batches_idx].to_numpy(dtype=str)
        logging.info(f'\tFound {len(np.unique(batches))} batches')

    # Get 1-D samples array; add common sample value if information is missing
    logging.info(f"Retrieving sample information in '{args.input_path}'")
    try:
        samples_idx = input_col_low.get_loc('sample')
    except KeyError:
        logging.warning(f"\tNo sample information in '{args.input_path}'")
        logging.warning("\tSetting sample value to 'sample0' for all cells")
        samples = np.full(input_table.shape[0], 'sample0')
    else:
        samples = input_table.iloc[:, samples_idx].to_numpy(dtype=str)
        logging.info(f'\tFound {len(np.unique(samples))} samples')

    # Get 1-D pre-annotated cell types array
    logging.info(f"Retrieving cell type information in '{args.input_path}'")
    try:
        cell_type_idx = input_col_low.get_loc('cell_type')
    except KeyError:
        logging.warning(f"\tNo cell type information in '{args.input_path}'")
        logging.warning("\tSetting cell type value to 'cell_type0' for all cells")
        cell_labels = np.full(input_table.shape[0], 'cell_type0')
    else:
        cell_labels = input_table.iloc[:, cell_type_idx].to_numpy(dtype=str)
        logging.info(f'\tFound {len(np.unique(cell_labels))} cell types: {np.unique(cell_labels)}')

    # Get unique labels array
    uniq_labels = np.unique(cell_labels)

    # Get 1-D three peaks markers array, otherwise use empty array
    try:
        logging.info(f"Checking for 3 peaks markers file at '{args.three_peak_markers}'")
        with open(args.three_peak_markers) as file:
            three_peak_markers = np.array(file.read().splitlines(), dtype='str')
    except FileNotFoundError:
        logging.warning('\tNo file provided, using CLI arguments')
        if args.three_peak_markers:
            three_peak_markers = list(filter(None, args.three_peak_markers.split(',')))
            three_peak_markers = np.array(three_peak_markers, dtype='str')
            logging.info(f"\t\tFound {len(three_peak_markers)} markers: {', '.join(three_peak_markers)}")
        else:
            logging.warning('\t\tNo markers provided in CLI, using default empty array')
            three_peak_markers = np.empty(0, dtype='str')
    else:
        logging.info(f"\tFound {len(three_peak_markers)} markers in '{three_peak_markers}'")

    # Get markers thresholds
    logging.info('Parsing marker expression thresholds from CLI')
    try:
        if len(three_peak_markers) == 0:  # Only two peaks markers
            two_peak_threshold = [float(x) for x in args.thresholds.split(',')][0]
            three_peak_low = None  # Not needed so set to None
            three_peak_high = None  # Not needed so set to None
        else:  # Some three peaks markers
            two_peak_threshold, three_peak_low, three_peak_high = [float(x) for x in args.thresholds.split(',')]
    except ValueError as e:
        logging.error(f'\t{e}')
        logging.error('Please make sure you provide 1 or 3 numbers separated by comma')
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f'\ttwo_peak_threshold set to {two_peak_threshold}')
        logging.info(f'\tthree_peak_low set to {three_peak_low}')
        logging.info(f'\tthree_peak_high set to {three_peak_high}')

    # Get cell types definitions dictionary
    logging.info(f"Importing cell type definitions from '{args.cell_type_path}'")
    try:
        with open(args.cell_type_path) as cell_types_input:
            cell_types_dict = yaml.safe_load(cell_types_input)
        # Note: this file was created using data from
        # https://github.com/RGLab/rcellontologymapping/blob/main/src/src/ImmportDefinitions.hs
    except FileNotFoundError:
        logging.error(f"\tCould not find '{args.cell_type_path}'. Please double-check file path")
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f"\tFound {len(cell_types_dict)} cell types: {', '.join(cell_types_dict.keys())}")

    # Parse config file
    logging.info(f"Checking for config file at '{args.config_path}'")
    config_args = [args.config_path, uniq_labels,
                   args.markers_interest, args.detection_method]
    markers_interest_lst, detection_method_lst = parse_config_file(*config_args)

    # Get CPU settings
    logging.info('Determining parallelisation settings')
    if args.cores == 1:  # Can't parallelise with only 1 core
        logging.info('\tOnly 1 CPU provided in total')
        nb_cpu_id = nb_cpu_eval = 1
    else:  # Parallelise as much as possible
        logging.info(f'\t{args.cores} CPUs provided in total')
        # Adapt number of CPUs in ThreadPool to number of cell types
        if len(uniq_labels) == 1:  # Can only parallelise combinations testing
            nb_cpu_id = 1  # Use only 1 CPU in ThreadPool
            nb_cpu_eval = args.cores - nb_cpu_id  # Assign remaining CPUs to ProcessPool
        else:  # Can parallelise combinations testing and cell type processing
            nb_cpu_eval = floor(0.9 * args.cores)  # Assign 90% of CPUs to ProcessPool
            nb_cpu_id = args.cores - nb_cpu_eval  # Assign remaining CPUs to ThreadPool
            # If more threads than cell types, reassign excess CPUs from
            # ThreadPool to ProcessPool
            diff = nb_cpu_id - len(uniq_labels)
            if diff > 0:
                nb_cpu_id -= diff
                nb_cpu_eval += diff
    logging.info(f'\tSetting nb_cpu_id to {nb_cpu_id} and nb_cpu_eval to {nb_cpu_eval}')

    logging.info('Checking remaining parameter values')
    logging.info('\tChecking presence of markers of interest in general marker list')
    mask_interest_inv_lst = []
    for label, markers_interest in zip(uniq_labels, markers_interest_lst):
        if sum(np.isin(markers_interest, markers)) == len(markers_interest):
            mask_interest_inv = np.isin(markers, markers_interest, invert=True)
            mask_interest_inv_lst.append(mask_interest_inv)
            logging.info(f"\t\tAll markers of interest located for cell type '{label}'")
        else:
            missing_interest = markers_interest[np.isin(markers_interest,
                                                        markers, invert=True)]
            logging.error(f"\t\tPlease double-check markers of interest for cell type '{label}'")
            logging.error(f"\t\tSome markers are missing in general marker list: {', '.join(missing_interest)}")
            sys.exit(1)

    logging.info('\tChecking value of detection method')
    for label, detection_method in zip(uniq_labels, detection_method_lst):
        if isinstance(detection_method, int):  # Only process integers
            if (detection_method > len(markers)):
                logging.error(f"\t'-dm/--detection-method' for cell type '{label}' must be lower than number of markers in {args.marker_path}")
                sys.exit(1)
            elif (detection_method < len(markers_interest)):
                logging.error(f"\t'-dm/--detection-method' for cell type '{label}' must be higher than number of markers in {markers_interest}")
                sys.exit(1)
            else:
                logging.info(f"\t\tValid detection method value for cell type '{label}'")

    logging.info('\tChecking other parameters')
    if not (1 <= args.max_markers <= len(markers)):
        logging.error(f"\t'-a/--max-markers' must be an integer between 1 and {len(markers)}")
        sys.exit(1)
    if not (0.01 <= args.min_samplesxbatch <= 1):
        logging.error("\t'-q/--min-samplesxbatch' must be a float between 0.01 and 1")
        sys.exit(1)
    if not (1 <= args.min_cellxsample <= 100):
        logging.error("\t'-r/--min-cellxsample' must be an integer between 1 and 100")
        sys.exit(1)
    if not (0 <= args.knn_min_probability <= 1):
        logging.error("\t'-p/--knn-min-probability' must be a float between 0 and 1")
        sys.exit(1)
    knn_refine = (False if args.knn_min_probability == 0 else True)
    logging.info('\tAll parameter values within range')

    # Check if dryrun mode is used
    if args.dryrun:
        logging.info('Dryrun finished. Exiting...')
        sys.exit(0)

    # Use dictionary to avoid returning too many objects
    args_dict = {'batches': batches,
                 'cell_labels': cell_labels,
                 'cell_types_dict': cell_types_dict,
                 'detection_method_lst': detection_method_lst,
                 'input_table': input_table,
                 'knn_min_probability': args.knn_min_probability,
                 'knn_refine': knn_refine,
                 'markers': markers,
                 'markers_interest_lst': markers_interest_lst,
                 'mask_interest_inv_lst': mask_interest_inv_lst,
                 'mat': mat,
                 'max_markers': args.max_markers,
                 'min_cellxsample': args.min_cellxsample,
                 'min_samplesxbatch': args.min_samplesxbatch,
                 'nb_cpu_eval': nb_cpu_eval,
                 'nb_cpu_id': nb_cpu_id,
                 'output_path': args.output_path,
                 'samples': samples,
                 'three_peak_high': three_peak_high,
                 'three_peak_low': three_peak_low,
                 'three_peak_markers': three_peak_markers,
                 'two_peak_threshold': two_peak_threshold,
                 'uniq_labels': uniq_labels}

    return args_dict
