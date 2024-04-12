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
import json
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# Import local functions
from identify_phenotypes import identify_phenotypes  # AT. Double-check path
from utils import get_chunksize, get_cpu, setup_log  # AT. Double-check path
# from cellxhaustive.identify_phenotypes import identify_phenotypes
# from cellxhaustive.utils import get_chunksize, get_cpu, setup_log


# Parse arguments
parser = argparse.ArgumentParser(description='Package to annotate cell types using \
                                CITE-seq ADT data.')
parser.add_argument('-i', '--input', dest='input_path', type=str,
                    help='Path to input table with expression data and \
                    samples/batch/cell_type information',
                    required=True)
parser.add_argument('-m', '--markers', dest='marker_path', type=str,
                    help='Path to file with list of markers of interest',
                    required=True)
parser.add_argument('-o', '--output', dest='output_path', type=str,
                    help='Path to output table with annotations',
                    required=True)
parser.add_argument('-c', '--cell-type-definition', dest='cell_type_path', type=str,
                    help='Path to file with cell types characterisation \
                    [../data/config/major_cell_types.json]',
                    required=False, default='../data/config/major_cell_types.json')
parser.add_argument('-l', '--log', dest='log_path', type=str,
                    help='Path to log file [output_path.log]',
                    required=False, default='')
parser.add_argument('-n', '--log-level', dest='log_level', type=str,
                    help='Verbosity level of log file [info]',
                    required=False, default='info', choices=['debug', 'info', 'warning'])
parser.add_argument('-j', '--two-peak-threshold', dest='two_peak_threshold', type=float,
                    help='Threshold to determine whether a two-peaks marker is\
                    negative or positive [3]',
                    required=False, default=3)
parser.add_argument('-e', '--three-peaks', dest='three_peak_markers', type=str,
                    help='Path to file with markers that have three peaks []',
                    required=False, default='')
parser.add_argument('-f', '--three-peak-low', dest='three_peak_low', type=float,
                    help='Threshold to determine whether three-peaks marker is\
                    negative or low_positive [2]',
                    required=False, default=2)
parser.add_argument('-g', '--three-peak-high', dest='three_peak_high', type=float,
                    help='Threshold to determine whether three-peaks marker is\
                    positive or low_positive [4]',
                    required=False, default=4)
parser.add_argument('-a', '--max-markers', dest='max_markers', type=int,
                    help="Maximum number of relevant markers to select among \
                    total list of markers. Must be less than or equal to \
                    'len(markers)' [15]",
                    required=False, default=15)
parser.add_argument('-b', '--min-annotations', dest='min_annotations', type=int,
                    help="Minimum number of phenotypes for a combination of markers \
                    to be taken into account as a potential cell population. Must \
                    be in '[2; len(markers)]', but it is advised to choose a value \
                    in '[3; len(markers) - 1]' [3]",
                    required=False, default=3)
parser.add_argument('-s', '--max-solutions', dest='max_solutions', type=int,
                    help='Maximum number of optimal solutions to keep [10]',
                    required=False, default=10)
parser.add_argument('-q', '--min-samplesxbatch', dest='min_samplesxbatch', type=float,
                    help="Minimum proportion of samples within each batch with at \
                    least 'min_cellxsample' cells for a new annotation to be \
                    considered [0.5]",
                    required=False, default=0.5)
parser.add_argument('-r', '--min-cellxsample', dest='min_cellxsample', type=int,
                    help="Minimum number of cells within each sample in \
                    'min_samplesxbatch' %% of samples within each batch for a new \
                    annotation to be considered [10]",
                    required=False, default=10)
parser.add_argument('-k', '--no-knn', dest='knn_refine',
                    help='If present, do not refine annotations with a KNN classifier',
                    required=False, default=True, action="store_false")
parser.add_argument('-p', '--knn-min-probability', dest='knn_min_probability', type=float,
                    help='Confidence threshold for KNN classifier to reassign a new \
                    cell type to previously undefined cells [0.5]',
                    required=False, default=0.5)
parser.add_argument('-t', '--threads', dest='cores', type=int,
                    help='Number of cores to use. Specifying more than one core \
                    will run parallel jobs which will increase speed [1]',
                    required=False, default=1)
parser.add_argument('-d', '--dry-run', dest='dryrun',
                    help='Use dry-run mode to check input files and configuration [False]',
                    required=False, default=False, action="store_true")
args = parser.parse_args()


# Main script execution
if __name__ == '__main__':  # AT. Double check behaviour inside package

    # Determine log file name
    if not args.log_path:
        log_file = f'{os.path.splitext(args.output_path)[0]}.log'
    else:
        log_file = args.log_path

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    logging.debug(f'Creating log directory <{log_dir}>')
    os.makedirs(log_dir, exist_ok=True)

    # Set-up logging configuration
    setup_log(log_file, args.log_level)

    # Get 1-D array for markers
    logging.info(f'Importing markers list from <{args.marker_path}>')
    markers = pd.read_csv(args.marker_path, sep='\t', header=None).to_numpy(dtype=str).flatten()
    logging.info(f'\tFound {len(markers)} markers')

    # Parse general input files into several arrays
    logging.info(f'Importing cell data from <{args.input_path}>')
    input_table = pd.read_csv(args.input_path, sep='\t', header=0, index_col=0)
    logging.info(f'\tFound {len(input_table.index)} cells')

    # Get 2-D array for expression using 'markers'
    mat = input_table.loc[:, markers].to_numpy(dtype=float)

    # Get 1-D array for batch; add common batch value if information is missing
    logging.info(f'Retrieving batch information in <{args.input_path}>')
    if 'batch' in input_table.columns:
        batches = input_table.loc[:, 'batch'].to_numpy(dtype=str)
    else:
        logging.warning(f'\tNo batch information in <{args.input_path}>')
        logging.warning('\tSetting batch value to <batch0> for all cells')
        batches = np.full(input_table.shape[0], 'batch0')

    # Get 1-D array for sample; add common sample value if information is missing
    logging.info(f'Retrieving sample information in <{args.input_path}>')
    if 'sample' in input_table.columns:
        samples = input_table.loc[:, 'sample'].to_numpy(dtype=str)
    else:
        logging.warning(f'\tNo sample information in <{args.input_path}>')
        logging.warning('\tSetting sample value to <sample0> for all cells')
        samples = np.full(input_table.shape[0], 'sample0')

    # Get 1-D array for pre-annotated cell type
    logging.info(f'Retrieving cell type information in <{args.input_path}>')
    if 'cell_type' in input_table.columns:
        cell_labels = input_table.loc[:, 'cell_type'].to_numpy(dtype=str)
        logging.info(f'Found {np.unique(cell_labels)} pre-annotated cell types')
    else:
        logging.warning(f'\tNo cell type information in <{args.input_path}>')
        logging.warning('\tSetting cell type value to <cell_type0> for all cells')
        cell_labels = np.full(input_table.shape[0], 'cell_type0')

    # Get array of unique labels
    uniq_labels = np.unique(cell_labels)

    # Get list of arrays describing cells matching each cell type of 'uniq_labels'
    is_label_list = [(cell_labels == label) for label in uniq_labels]

    # Get three peaks markers if a file is specified, otherwise use default list
    logging.info('Checking for existence of markers with 3 peaks')
    three_path = args.three_peak_markers
    if (not isinstance(three_path, list) and pathlib.Path(three_path).is_file()):
        with open(three_path) as file:
            three_peak_markers = np.array(file.read().splitlines(), dtype=np.dtype('U15'))
        logging.info(f'\tFound {len(three_peak_markers)} markers in <{three_path}>')
    else:
        three_peak_markers = np.empty(0, dtype=np.dtype('U15'))
        logging.warning('\tNo file provided, using default empty list')

    # Import cell types definitions
    logging.info(f'Importing cell type marker definition from <{args.cell_type_path}>')
    with open(args.cell_type_path) as in_cell_types:
        cell_types_dict = json.load(in_cell_types)
    logging.info(f'\tFound {len(cell_types_dict)} cell types')
    # Note: this file was created using data from
    # https://github.com/RGLab/rcellontologymapping/blob/main/src/src/ImmportDefinitions.hs

    # Get other parameter values from argument parsing
    logging.info('Parsing remaining parameters')
    two_peak_threshold = args.two_peak_threshold
    three_peak_low = args.three_peak_low
    three_peak_high = args.three_peak_high
    max_markers = args.max_markers
    min_annotations = args.min_annotations
    max_solutions = args.max_solutions
    min_cellxsample = args.min_cellxsample
    min_samplesxbatch = args.min_samplesxbatch
    knn_refine = args.knn_refine
    knn_min_probability = args.knn_min_probability

    # Get CPU settings
    logging.info('Determining parallelisation settings')
    if args.cores == 1:  # Can't multiprocess with only 1 core
        logging.info('\tOnly 1 CPU provided, no parallelisation possible')
        nb_cpu_id = nb_cpu_eval = nb_cpu_keep = 1
        logging.info('\tSetting nb_cpu_id, nb_cpu_eval, and nb_cpu_keep to 1')
    else:  # Maximise CPU usage
        logging.info(f'\t{args.cores} CPUs provided')
        nb_cpu_id, nb_cpu_eval, nb_cpu_keep = get_cpu(args.cores, len(uniq_labels))

    if args.dryrun:
        logging.info('Dryrun finished. Exiting...')
        sys.exit(0)

    # Initialise empty arrays and dictionary to store new annotations and results
    logging.debug('Initialising empty objects to store results')
    annotations = np.asarray(['undefined'] * len(cell_labels)).astype('U150')
    phenotypes = np.asarray(['undefined'] * len(cell_labels)).astype('U150')
    annot_dict = {}

    # Process cells by pre-existing annotations using multiprocessing
    logging.info('Starting analyses')
    chunksize = get_chunksize(uniq_labels, nb_cpu_id)
    with ProcessPoolExecutor(max_workers=nb_cpu_id) as executor:
        annot_results_lst = list(executor.map(partial(identify_phenotypes,
                                                      mat=mat,
                                                      batches=batches,
                                                      samples=samples,
                                                      markers=markers,
                                                      cell_types_dict=cell_types_dict,
                                                      two_peak_threshold=two_peak_threshold,
                                                      three_peak_markers=three_peak_markers,
                                                      three_peak_low=three_peak_low,
                                                      three_peak_high=three_peak_high,
                                                      max_markers=max_markers,
                                                      min_annotations=min_annotations,
                                                      max_solutions=max_solutions,
                                                      min_samplesxbatch=min_samplesxbatch,
                                                      min_cellxsample=min_cellxsample,
                                                      knn_refine=knn_refine,
                                                      knn_min_probability=knn_min_probability,
                                                      cpu_eval_keep=(nb_cpu_eval, nb_cpu_keep)),
                                              is_label_list, uniq_labels,
                                              chunksize=chunksize))
    # Note: only 'is_label_list' and 'uniq_labels' are iterated over, hence the
    # use of 'partial()' to keep the other parameters constant

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
            annot_df_label.iloc[:, start:(start + 5)] = sub_res_df.copy()
            # Note: copy() is used to avoid reassignation problems

        # Fill general annotation dataframe with 'label' annotations
        logging.info(f'\t\t\tIntegrating subtable to general annotation table')
        annot_df.iloc[is_label, :end] = annot_df_label.copy()

    # Merge input dataframe and annotation dataframe
    logging.info('\tMerging input data and annotation table')
    annot_df.set_index(input_table.index, inplace=True)
    # Note: set indices to avoid problem during concatenation
    output_table = pd.concat([input_table, annot_df], axis=1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    logging.debug(f'Creating output directory <{output_dir}>')
    os.makedirs(output_dir, exist_ok=True)

    # Save general table with annotations and phenotypes
    logging.info(f'Saving final table to <{args.output_path}>')
    output_table.to_csv(args.output_path, sep='\t', header=True, index=True)
