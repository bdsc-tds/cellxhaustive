"""
Package to determine marker combination phenotypes and assign cell types from
CITE-seq ADT data through automated gating, feature selection and clustering.


Minimum requirements to run the analyses and associated parameters:
-i INPUT_PATH, --input INPUT_PATH
  Path to input table with expression data and samples/batch information. Rows
  should be cells and columns should be one of the following:
  - marker expression (float)
  - sample information (str)
  - batch information (str)
  - cell type information (after main-gating for example) (str)
  Columns must be tab-separated.

-m MARKER_PATH, --markers MARKER_PATH
  Path to file with list of markers of interest that will be used during the
  analyses. There should be one marker per line.

-o OUTPUT_PATH, --output OUTPUT_PATH
  Path to output table with all input data as well as phenotypes, cell types and
  associated scores determined during the analyses.
"""


# Import utility modules
import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import yaml
from functools import partial
from math import floor
from pathos.pools import ProcessPool, ThreadPool


# Import local functions
from identify_phenotypes import identify_phenotypes  # AT. Double-check path
from utils import setup_log  # AT. Double-check path
# from cellxhaustive.identify_phenotypes import identify_phenotypes
# from cellxhaustive.utils import setup_log


# Parse arguments
parser = argparse.ArgumentParser(description='Package to annotate cell types using \
                                 CITE-seq ADT data.')
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
                    all cell populations []',
                    required=False, default='')
parser.add_argument('-dm', '--detection-method', dest='detection_method', type=str,
                    help="Method to identify length of best marker combination. \
                    Must be 'auto' to use default algorithm or an integer to set \
                    combination length. Integer must be less than '-a' parameter. \
                    Global setting that applies to all cell populations [auto]",
                    required=False, default='auto')
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
                    help='Path to file with markers with three peaks []',
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
args = parser.parse_args()


# Main script execution
if __name__ == '__main__':  # AT. Double check behaviour inside package

    # Set log level and determine log file name
    log_level = args.log_level
    output_path = args.output_path
    log_path = args.log_path
    if not log_path:
        log_file = f'{os.path.splitext(output_path)[0]}.log'
    else:
        log_file = log_path

    # Make log variables as environment variables
    os.environ['LOG_FILE'] = log_file
    os.environ['LOG_LEVEL'] = log_level

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    logging.debug(f'Creating log directory <{log_dir}>')
    os.makedirs(log_dir, exist_ok=True)

    # Set-up logging configuration
    setup_log(log_file, log_level, 'w')

    # Get 1-D array for markers
    pd.options.mode.copy_on_write = True  # Save memory by using CoW mode

    # Import marker list
    marker_path = args.marker_path
    try:
        logging.info(f'Importing marker list from <{marker_path}>')
        markers = pd.read_csv(marker_path, sep='\t', header=None).to_numpy(dtype=str).flatten()
    except FileNotFoundError:
        logging.error(f'\tCould not find <{marker_path}>. Please double-check file path')
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f'\tFound {len(markers)} markers')

    # Parse general input files into several arrays
    input_path = args.input_path
    try:
        logging.info(f'Importing cell data from <{input_path}>')
        input_table = pd.read_csv(input_path, sep='\t', header=0, index_col=0)
    except FileNotFoundError:
        logging.error(f'\tCould not find <{input_path}>. Please double-check file path')
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f'\tFound {len(input_table.index)} cells')

    # Get 2-D array for expression using 'markers'
    try:
        logging.info(f'Selecting ADT counts in <{input_path}>')
        mat = input_table.loc[:, markers].to_numpy(dtype=float)
    except KeyError as e:
        logging.error(f'\tCould not find marker <{e}> in <{input_path}>. Please double-check marker list')
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)

    # Get 1-D array for batches; add common batch value if information is missing
    input_col_low = input_table.columns.str.lower()  # Make column names case-insensitive
    try:
        logging.info(f'Retrieving batch information in <{input_path}>')
        batches_idx = input_col_low.get_loc('batch')
    except KeyError:
        logging.warning(f'\tNo batch information in <{input_path}>')
        logging.warning('\tSetting batch value to <batch0> for all cells')
        batches = np.full(input_table.shape[0], 'batch0')
    else:
        batches = input_table.iloc[:, batches_idx].to_numpy(dtype=str)
        logging.info(f'\tFound {len(np.unique(batches))} batches')

    # Get 1-D array for samples; add common sample value if information is missing
    try:
        logging.info(f'Retrieving sample information in <{input_path}>')
        samples_idx = input_col_low.get_loc('sample')
    except KeyError:
        logging.warning(f'\tNo sample information in <{input_path}>')
        logging.warning('\tSetting sample value to <sample0> for all cells')
        samples = np.full(input_table.shape[0], 'sample0')
    else:
        samples = input_table.iloc[:, samples_idx].to_numpy(dtype=str)
        logging.info(f'\tFound {len(np.unique(samples))} samples')

    # Get 1-D array for pre-annotated cell types
    try:
        logging.info(f'Retrieving cell type information in <{input_path}>')
        cell_type_idx = input_col_low.get_loc('cell_type')
    except KeyError:
        logging.warning(f'\tNo cell type information in <{input_path}>')
        logging.warning('\tSetting cell type value to <cell_type0> for all cells')
        cell_labels = np.full(input_table.shape[0], 'cell_type0')
    else:
        cell_labels = input_table.iloc[:, cell_type_idx].to_numpy(dtype=str)
        logging.info(f'\tFound {len(np.unique(cell_labels))} cell types: {np.unique(cell_labels)}')

    # Get array of unique labels
    uniq_labels = np.unique(cell_labels)
    multipop = False if len(uniq_labels) == 1 else True

    # Get list of arrays describing cells matching each cell type of 'uniq_labels'
    is_label_list = [(cell_labels == label) for label in uniq_labels]

    # Get three peaks markers if file exists, otherwise use empty array
    three_path = args.three_peak_markers
    try:
        logging.info('Checking for file with 3 peaks markers')
        with open(three_path) as file:
            three_peak_markers = np.array(file.read().splitlines(), dtype='str')
    except FileNotFoundError:
        logging.warning('\tNo file provided, using default empty array')
        three_peak_markers = np.empty(0, dtype='str')
    else:
        logging.info(f'\tFound {len(three_peak_markers)} markers in <{three_path}>')

    # Get markers thresholds
    thresholds = args.thresholds
    logging.info('Parsing markers thresholds')
    try:
        if three_peak_markers.size == 0:  # Only two peaks markers
            two_peak_threshold = [float(x) for x in thresholds.split(',')][0]
            three_peak_low = None  # Not needed so set to None
            three_peak_high = None  # Not needed so set to None
        else:  # Some three peaks markers
            two_peak_threshold, three_peak_low, three_peak_high = [float(x) for x in thresholds.split(',')]
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

    # Import cell types definitions
    cell_type_path = args.cell_type_path
    try:
        logging.info('Checking for file with cell type definitions')
        with open(cell_type_path) as cell_types_input:
            cell_types_dict = yaml.safe_load(cell_types_input)
        # Note: this file was created using data from
        # https://github.com/RGLab/rcellontologymapping/blob/main/src/src/ImmportDefinitions.hs
    except FileNotFoundError:
        logging.error(f'\tCould not find <{cell_type_path}>. Please double-check file path')
        sys.exit(1)
    except Exception as e:
        logging.error(f'\t{e}')
        sys.exit(1)
    else:
        logging.info(f'\tFound {len(cell_types_dict)} cell types in <{cell_type_path}>')

    # Get markers of interest, otherwise use empty array
    markers_interest = args.markers_interest
    logging.info('Checking for markers of interest')
    if markers_interest:
        markers_interest = np.array(markers_interest.split(','), dtype='str')
        logging.info(f'\tFound {len(markers_interest)} markers')
        logging.info('\tIdentifying markers of interest in general marker list')
        if sum(np.isin(markers_interest, markers)) == len(markers_interest):
            mask_interest_inv = np.isin(markers, markers_interest, invert=True)
            logging.info('\t\tAll markers of interest located')
        else:
            missing_interest = markers_interest[np.isin(markers_interest, markers, invert=True)]
            logging.error(f'\t\tSome markers of interest are missing in general marker list')
            logging.error(f"\t\tMissing markers: {', '.join(missing_interest)}")
            sys.exit(1)
    else:
        logging.warning('\tNo markers provided, using default empty array')
        markers_interest = np.empty(0, dtype='str')

    # Get method to decide final combination length
    logging.info('Determining detection method')
    detection_method = args.detection_method
    if detection_method == 'auto':  # Check if default setting is chosen
        logging.info('\tUsing default algorithm')
    elif detection_method.isdecimal():  # Check if an integer is provided
        detection_method = int(detection_method)
        if (detection_method > len(markers)):
            logging.error(f"\t'-dm/--detection-method' must be lower than number of markers in {marker_path}")
            sys.exit(1)
        if (detection_method < len(markers_interest)):
            logging.error(f"\t'-dm/--detection-method' must be higher than number of markers in {markers_interest}")
            sys.exit(1)
        logging.info(f'\tWill look for combinations with <{detection_method}> markers')
    else:
        logging.error("\tUnknown detection method. Please provide 'auto' or a positive integer")
        sys.exit(1)

    # Get other parameter values from argument parsing
    logging.info('Parsing remaining parameters')
    max_markers = args.max_markers
    min_samplesxbatch = args.min_samplesxbatch
    min_cellxsample = args.min_cellxsample
    knn_min_probability = args.knn_min_probability
    logging.info('\tDone')

    logging.info('Checking parameter values')
    if not (1 <= max_markers <= len(markers)):
        logging.error(f"\t'-a/--max-markers' must be an integer between 1 and {len(markers)}")
        sys.exit(1)
    if not (0.01 <= min_samplesxbatch <= 1):
        logging.error("\t'-q/--min-samplesxbatch' must be a float between 0.01 and 1")
        sys.exit(1)
    if not (1 <= min_cellxsample <= 100):
        logging.error("\t'-r/--min-cellxsample' must be an integer between 1 and 100")
        sys.exit(1)
    if not (0 <= knn_min_probability <= 1):
        logging.error("\t'-p/--knn-min-probability' must be a float between 0 and 1")
        sys.exit(1)
    knn_refine = (False if knn_min_probability == 0 else True)
    logging.info('\tAll parameter values within range')

    # Get CPU settings
    logging.info('Determining parallelisation settings')
    cores = args.cores
    if cores == 1:  # Can't parallelise with only 1 core
        logging.info('\tOnly 1 CPU provided in total')
        nb_cpu_id = nb_cpu_eval = 1
    else:  # Parallelise as much as possible
        logging.info(f'\t{cores} CPUs provided in total')
        # Adapt number of CPUs in ThreadPool to number of cell populations
        if len(uniq_labels) == 1:  # Can only parallelise combinations testing
            nb_cpu_id = 1  # Use only 1 CPU in ThreadPool
            nb_cpu_eval = cores - nb_cpu_id  # Assign remaining CPUs to ProcessPool
        else:  # Can parallelise combinations testing and cell population processing
            nb_cpu_eval = floor(0.9 * cores)  # Assign 90% of CPUs to ProcessPool
            nb_cpu_id = cores - nb_cpu_eval  # Assign remaining CPUs to ThreadPool
            # If more threads than cell populations, reassign excess CPUs from
            # ThreadPool to ProcessPool
            diff = nb_cpu_id - len(uniq_labels)
            if diff > 0:
                nb_cpu_id -= diff
                nb_cpu_eval += diff
    logging.info(f'\tSetting nb_cpu_id to {nb_cpu_id} and nb_cpu_eval to {nb_cpu_eval}')

    # Initialise ProcessPool if needed, otherwise set it to None
    processpool = (None if nb_cpu_eval == 1 else ProcessPool(ncpus=nb_cpu_eval))

    if args.dryrun:
        logging.info('Dryrun finished. Exiting...')
        sys.exit(0)

    logging.info('Gating markers and extracting associated data in each cell type')
    # Determine path for marker filtering results
    marker_file_path = f'{os.path.splitext(output_path)[0]}_markers.txt'

    # Initialise list of arrays to store results
    mat_subset_rep_list = []
    batches_label_list = []
    samples_label_list = []
    markers_representative_list = []

    # Specific context manager to save information on marker filtering
    with open(marker_file_path, 'w') as file:
        for label, is_label in zip(uniq_labels, is_label_list):  # Loop over cell types
            # Select relevant markers in each batch of each cell population (main gating)
            logging.info(f'\tProcessing <{label}> cells')
            markers_rep_batches = []
            for batch in np.unique(batches):
                logging.debug(f'\t\tProcessing batch <{batch}>')
                logging.debug('\t\t\tSelecting cells matching current batch')
                is_batch = (batches == batch)
                # Create boolean array to select cells matching current 'label' and 'batch'
                is_label_batch = np.logical_and(is_batch, is_label)

                # Subset expression matrix to cells of current 'batch' and 'label' only
                logging.debug('\t\t\tSelecting corresponding expression data')
                try:
                    mat_subset = mat[is_label_batch, :][:, mask_interest_inv]  # With markers of interest
                except NameError:
                    mat_subset = mat[is_label_batch, :]  # Without markers of interest

                # Check bimodality of markers and select best ones
                logging.debug('\t\t\tEvaluating marker bimodality')
                marker_center_values = np.abs(np.mean(mat_subset, axis=0) - two_peak_threshold)
                marker_threshold_value = np.sort(marker_center_values)[max_markers - 1]  # Select max_markers-th center value (in ascending order)
                # Note: '- 1' is used to compensate 0-based indexing
                is_center_greater = (marker_center_values <= marker_threshold_value)
                try:  # Select markers with center higher than max_markers-th center
                    markers_rep = markers[mask_interest_inv][is_center_greater]  # With markers of interest
                except NameError:
                    markers_rep = markers[is_center_greater]  # Without markers of interest

                # Store list of relevant markers for every batch
                logging.debug('\t\t\tStoring relevant markers')
                markers_rep_batches.extend(list(markers_rep))

            if len(np.unique(batches)) == 1:  # No need to filter markers if there is only 1 batch
                logging.info(f'\t\tFound only one batch, no need to filter markers')
                markers_representative = markers_rep_batches
                logging.info(f"\t\t\tFound {len(markers_representative)} markers: {', '.join(markers_representative)}")
            else:
                logging.info(f'\t\tFound {len(np.unique(batches))} batches. Selecting markers present in at least 2 batches')
                markers_representative = set([mk for mk in markers_rep_batches
                                              if markers_rep_batches.count(mk) >= 2])
                logging.info(f"\t\t\tFound {len(markers_representative)} markers: {', '.join(markers_representative)}")
                missing_markers = set([mk for mk in markers_rep_batches
                                       if markers_rep_batches.count(mk) < 2])
                logging.info(f"\t\t\tFiltered out {len(missing_markers)} marker{'s' if len(missing_markers) > 1 else ''}: {', '.join(missing_markers)}")

            # Convert format back to array
            markers_representative = np.array(list(markers_representative))

            # Save marker information in different file
            file.write(f'{label} cells:\n')
            file.write(f"\tMarkers found: {', '.join(markers_rep_batches)}\n")
            file.write(f"\tMarkers kept (found in at least 2 batches): {', '.join(markers_representative)}\n")
            file.write(f"\tMarkers removed (present in only 1 batch): {', '.join(missing_markers)}\n")

            # Extract expression, batch and sample information across all batches
            # for cell population 'label'
            logging.info('\t\tExtracting matching expression, batch and sample information')
            mat_subset_label = mat[is_label, :]
            batches_label = batches[is_label]
            samples_label = samples[is_label]

            # Merge back markers_representative with markers_interest
            markers_interest_rep = np.append(markers_interest, markers_representative)

            # Slice matrix to keep only expression of relevant markers
            markers_rep_int_mask = np.isin(markers, markers_interest_rep)
            markers_representative = markers[markers_rep_int_mask]  # Reorder markers
            mat_subset_rep_markers = mat_subset_label[:, markers_rep_int_mask]

            logging.info('\t\tStoring data in arrays')
            mat_subset_rep_list.append(mat_subset_rep_markers)  # Store slice matrix
            batches_label_list.append(batches_label)  # Store batch information
            samples_label_list.append(samples_label)  # Store sample information
            markers_representative_list.append(markers_representative)  # Store representative markers
            # Note: keeping data in list of arrays makes multiprocessing easier and
            # reduce overall memory usage

    # Free memory by deleting heavy objects
    del (is_batch, is_label_batch, mat_subset, mat_subset_label, mat, batches, samples)

    # Process cells by pre-existing annotations
    if nb_cpu_id == 1:  # Use for loop to avoid creating new processes
        logging.info('Starting population analyses without parallelisation')
        annot_results_lst = []
        for is_label, cell_name, mat_representative, batches_label, \
                samples_label, markers_representative in zip(is_label_list,
                                                             uniq_labels,
                                                             mat_subset_rep_list,
                                                             batches_label_list,
                                                             samples_label_list,
                                                             markers_representative_list):
            results_dict = identify_phenotypes(is_label=is_label,
                                               cell_name=cell_name,
                                               mat_representative=mat_representative,
                                               batches_label=batches_label,
                                               samples_label=samples_label,
                                               markers_representative=markers_representative,
                                               markers_interest=markers_interest,
                                               detection_method=detection_method,
                                               cell_types_dict=cell_types_dict,
                                               two_peak_threshold=two_peak_threshold,
                                               three_peak_markers=three_peak_markers,
                                               three_peak_low=three_peak_low,
                                               three_peak_high=three_peak_high,
                                               max_markers=max_markers,
                                               min_samplesxbatch=min_samplesxbatch,
                                               min_cellxsample=min_cellxsample,
                                               knn_refine=knn_refine,
                                               knn_min_probability=knn_min_probability,
                                               multipop=multipop,
                                               processpool=processpool)
            annot_results_lst.append(results_dict)
    else:  # Use ThreadPool to parallelise cell type processing
        logging.info('Starting population analyses with parallelisation')
        with ThreadPool(nthreads=nb_cpu_id) as threadpool:
            annot_results_lst = list(threadpool.map(partial(identify_phenotypes,
                                                            markers_interest=markers_interest,  # AT. Ultimately, this should be out of the function and become an iterable, with 1 array/cell-type
                                                            detection_method=detection_method,  # AT. Ultimately, this should be out of the function and become an iterable, with 1 keyword or integer/cell-type
                                                            cell_types_dict=cell_types_dict,
                                                            two_peak_threshold=two_peak_threshold,
                                                            three_peak_markers=three_peak_markers,
                                                            three_peak_low=three_peak_low,
                                                            three_peak_high=three_peak_high,
                                                            max_markers=max_markers,
                                                            min_samplesxbatch=min_samplesxbatch,
                                                            min_cellxsample=min_cellxsample,
                                                            knn_refine=knn_refine,
                                                            knn_min_probability=knn_min_probability,
                                                            multipop=multipop,
                                                            processpool=processpool),  # AT. Rework this
                                                    is_label_list,
                                                    uniq_labels,
                                                    mat_subset_rep_list,
                                                    batches_label_list,
                                                    samples_label_list,
                                                    markers_representative_list))
        # Note: 'partial()' is used to iterate over 'is_label_list', 'uniq_labels',
        # 'mat_subset_rep_list', 'batches_label_list', 'samples_label_list' and
        # 'markers_representative_list' and keep other parameters constant

    # Make sure ProcessPool is closed. It should already be, but this makes sure
    # there are no zombie processes
    if processpool:
        processpool.close()
        processpool.join()
    del processpool

    # Reset logging configuration
    setup_log(log_file, log_level, 'a')

    # Convert results back to dictionary
    annot_dict = dict(zip(uniq_labels, annot_results_lst))

    # Post-process results to add them to original table and save whole table
    logging.info('Gathering results in annotation table')

    # Find maximum number of optimal combinations across all 'cell_labels'
    logging.debug('\tDetermining total maximum number of optimal combinations')
    max_comb = max([len(annot_dict[label].keys()) for label in uniq_labels])
    logging.debug(f"\t\tFound {max_comb} combination{'s' if max_comb > 1 else ''}")

    # Build list with all column names
    logging.debug('\tBuilding column names')
    col_names = []
    for i in range(max_comb):
        col_names.extend([f'Annotations_{i + 1}', f'Phenotypes_{i + 1}'])
        if knn_refine:
            col_names.extend([f'KNN_annotations_{i + 1}',
                              f'KNN_phenotype_{i + 1}',
                              f'KNN_proba_{i + 1}'])

    # Initialise empty dataframe to store all annotation results
    logging.debug('\tInitialising empty annotation table')
    annot_df = pd.DataFrame(None,
                            index=range(input_table.shape[0]),
                            columns=col_names)

    # Fill annotation dataframe with results
    logging.info('\tFilling annotation table with analyses results')
    for label, is_label in zip(uniq_labels, is_label_list):
        logging.info(f'\t\tCreating result subtable for <{label}> annotations')

        # Slice general results dictionary
        logging.debug(f'\t\t\tSelecting associated results')
        sub_results = annot_dict[label]

        # Find number of optimal combinations for 'label' cells
        logging.info(f'\t\t\tDetermining maximum number of optimal combinations')
        label_nb_comb = list(sub_results.keys())
        logging.info(f"\t\t\t\tFound {len(label_nb_comb)} combination{'s' if len(label_nb_comb) > 1 else ''}")

        # Get number of cells
        logging.info(f'\t\t\tCounting cells')
        cell_nb = sub_results[list(sub_results)[0]]['new_labels'].shape[0]
        # Note: 'list(sub_results)[0]' is used because it will always exist
        logging.info(f'\t\t\t\tFound {cell_nb} cells')

        # Get column names
        logging.debug('\t\t\tBuilding column names')
        end = (5 * len(label_nb_comb)) if knn_refine else (2 * len(label_nb_comb))
        col_names_sub = col_names[:end]

        # Initialise empty dataframe to store annotation results for 'label'
        logging.debug('\t\t\tInitialising empty table to proper dimensions')
        annot_df_label = pd.DataFrame(np.nan,
                                      index=range(cell_nb),
                                      columns=col_names_sub)

        # Create dataframe results for all optimal combinations of 'label'
        logging.debug('\t\t\tFilling table')
        for idx, comb_nb in enumerate(label_nb_comb):
            # Extract results
            sub_res_df = pd.DataFrame.from_dict(sub_results[comb_nb], orient='index').transpose()

            # Fill 'label' annotation dataframe
            start = 5 * idx
            annot_df_label.iloc[:, start:(start + 5)] = sub_res_df

        # Fill general annotation dataframe with 'label' annotations
        logging.info(f'\t\t\tIntegrating subtable to general annotation table')
        annot_df.iloc[is_label, :end] = annot_df_label.copy(deep=True)

    # Merge input dataframe and annotation dataframe
    logging.info('\tMerging input data and annotation table')
    annot_df.set_index(input_table.index, inplace=True)
    # Note: set indices to avoid problem during concatenation
    output_table = pd.concat([input_table, annot_df], axis=1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    logging.debug(f'Creating output directory <{output_dir}>')
    os.makedirs(output_dir, exist_ok=True)

    # Save general table with annotations and phenotypes
    logging.info(f'Saving final table to <{output_path}>')
    output_table.to_csv(output_path, sep='\t', header=True, index=True)
