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
import numpy as np
import os
import pandas as pd
import pathlib


# Import local functions
from identify_phenotypes import identify_phenotypes
# from cellxhaustive.identify_phenotypes import identify_phenotypes  # AT. Double-check path


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
parser.add_argument('-t', '--two-peak-threshold', dest='two_peak_threshold', type=float,
                    help="Threshold to determine whether a two-peaks marker is\
                    negative or positive [3]",
                    required=False, default=3)
parser.add_argument('-tp', '--three-peaks', dest='three_peak_markers', type=str,
                    help="Path to file with markers that have three peaks []",
                    required=False, default=[])
parser.add_argument('-tpl', '--three-peak-low', dest='three_peak_low', type=float,
                    help="Threshold to determine whether three-peaks marker is\
                    negative or low_positive [2]",
                    required=False, default=2)
parser.add_argument('-tph', '--three-peak-high', dest='three_peak_high', type=float,
                    help="Threshold to determine whether three-peaks marker is\
                    positive or low_positive [4]",
                    required=False, default=4)
parser.add_argument('-c', '--cell-type-definition', dest='cell_type_path', type=str,
                    help='Path to file with cell types characterisation \
                    [../data/config/major_cell_types.json]',
                    required=False, default='../data/config/major_cell_types.json')
parser.add_argument('-mm', '--max-markers', dest='max_markers', type=int,
                    help="Maximum number of relevant markers to select among \
                    total list of markers. Must be less than or equal to \
                    'len(markers)' [15]",
                    required=False, default=15)
parser.add_argument('-ma', '--min-annotations', dest='min_annotations', type=int,
                    help="Minimum number of phenotypes for a combination of markers \
                    to be taken into account as a potential cell population. Must \
                    be in '[2; len(markers)]', but it is advised to choose a value \
                    in '[3; len(markers) - 1]' [3]",
                    required=False, default=3)
parser.add_argument('-s', '--max-solutions', dest='max_solutions', type=int,
                    help="Maximum number of optimal solutions to keep",
                    required=False, default=10)
parser.add_argument('-ms', '--min-samplesxbatch', dest='min_samplesxbatch', type=float,
                    help="Minimum proportion of samples within each batch with at \
                    least 'min_cellxsample' cells for a new annotation to be \
                    considered [0.5]",
                    required=False, default=0.5)
parser.add_argument('-mc', '--min-cellxsample', dest='min_cellxsample', type=int,
                    help="Minimum number of cells within each sample in \
                    'min_samplesxbatch' %% of samples within each batch for a new \
                    annotation to be considered [10]",
                    required=False, default=10)
parser.add_argument('-k', '--knn-refine', dest='knn_refine', type=bool,
                    help='If True, clustering done via permutations of relevant \
                    markers will be refined using a KNN classifier [True]',
                    required=False, default=True, choices=[True, False])
parser.add_argument('-knn', '--knn-min-probability', dest='knn_min_probability', type=float,
                    help='Confidence threshold for KNN classifier to reassign a new \
                    cell type to previously undefined cells [0.5]',
                    required=False, default=0.5)
args = parser.parse_args()


# Main script execution
if __name__ == '__main__':  # AT. Double check behaviour inside package

    # Get 1-D array for markers
    markers = pd.read_csv(args.marker_path, sep='\t', header=None).to_numpy().flatten()

    # Parse general input files into several arrays
    input_table = pd.read_csv(args.input_path, sep='\t', header=0, index_col=0)

    # Get 2-D array for expression using 'markers'
    mat = input_table.loc[:, markers].to_numpy()

    # Get 1-D array for batch; add common batch value if information is missing
    if 'batch' in input_table.columns:
        batches = input_table.loc[:, 'batch'].to_numpy()
    else:
        batches = np.full(input_table.shape[0], 'batch0')

    # Get 1-D array for sample; add common sample value if information is missing
    if 'sample' in input_table.columns:
        samples = input_table.loc[:, 'sample'].to_numpy()
    else:
        samples = np.full(input_table.shape[0], 'sample0')

    # Get 1-D array for pre-annotated cell type
    cell_labels = input_table.loc[:, 'cell_type'].to_numpy()

    # Get three peaks markers if a file is specified, otherwise use default list
    three_path = args.three_peak_markers
    if (not isinstance(three_path, list) and pathlib.Path(three_path).is_file()):
        with open(three_path) as file:
            lines = file.read().splitlines()
    else:
        three_peak_markers = args.three_peak_markers

    # Import cell types definitions
    with open(args.cell_type_path) as in_cell_types:
        cell_types_dict = json.load(in_cell_types)
    # Note: this file was created using data from
    # https://github.com/RGLab/rcellontologymapping/blob/main/src/src/ImmportDefinitions.hs

    # Get other parameter values from argument parsing
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

    # Initialise empty arrays and dictionary to store new annotations and results
    annotations = np.asarray(['undefined'] * len(cell_labels)).astype('U150')
    phenotypes = np.asarray(['undefined'] * len(cell_labels)).astype('U150')
    annot_dict = {}

    # Process cells by pre-annotations
    for label in np.unique(cell_labels):
        # Create boolean array to select cells matching current label
        is_label = (cell_labels == label)

        # Get annotations for all cells of type 'label'
        results_dict = identify_phenotypes(
            mat=mat,
            batches=batches,
            samples=samples,
            markers=markers,
            is_label=is_label,
            cell_types_dict=cell_types_dict,
            two_peak_threshold=two_peak_threshold,
            three_peak_markers=three_peak_markers,
            three_peak_low=three_peak_low,
            three_peak_high=three_peak_high,
            cell_name=label,
            max_markers=max_markers,
            min_annotations=min_annotations,
            max_solutions=max_solutions,
            min_cellxsample=min_cellxsample,
            min_samplesxbatch=min_samplesxbatch,
            knn_refine=knn_refine,
            knn_min_probability=knn_min_probability)

        # Store results in another dictionary for post-processing
        annot_dict[label] = results_dict

    # Post-process results to add them to original table and save whole table

    # Find maximum number of optimal combinations across all 'cell_labels'
    max_comb = max([len(annot_dict[label].keys()) for label in np.unique(cell_labels)])

    # Build list with all column names
    col_names = []
    for i in range(max_comb):
        col_names.extend([f'Annotations_{i + 1}', f'Phenotypes_{i + 1}'])
        if knn_refine:
            col_names.extend([f'KNN_annotations_{i + 1}', f'KNN_proba_{i + 1}'])

    # Initialise empty dataframe to store all annotation results
    annot_df = pd.DataFrame(None,
                            index=range(input_table.shape[0]),
                            columns=col_names)

    # Fill annotation dataframe with results
    for label in np.unique(cell_labels):
        # Create boolean array to select cells matching current 'label'
        is_label = (cell_labels == label)

        # Slice general results dictionary
        sub_results = annot_dict[label]

        # Find number of optimal combinations for 'label' cells
        label_nb_comb = len(sub_results.keys())

        # Get number of cells
        cell_nb = sub_results[0]['new_labels'].shape[0]
        # Note: 'sub_results[0]' is used because it will always exist

        # Get column names
        end = (4 * label_nb_comb) if knn_refine else (2 * label_nb_comb)
        col_names_sub = col_names[:end]

        # Initialise empty dataframe to store annotation results for 'label'
        annot_df_label = pd.DataFrame(np.nan,
                                      index=range(cell_nb),
                                      columns=col_names_sub)

        # Create dataframe results for all optimal combinations of 'label'
        for comb_nb in range(label_nb_comb):
            # Extract results
            sub_res_df = pd.DataFrame.from_dict(sub_results[comb_nb], orient='index').transpose()

            # Fill 'label' annotation dataframe
            start = 4 * comb_nb
            annot_df_label.iloc[:, start:(start + 4)] = sub_res_df.copy()
            # Note: copy() is used to avoid reassignation problems

        # Fill general annotation dataframe with 'label' annotations
        annot_df.iloc[is_label, :end] = annot_df_label.copy()

    # Merge input dataframe and annotation dataframe
    output_table = pd.concat([input_table, annot_df], axis=1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save general table with annotations and phenotypes
    output_table.to_csv(args.output_path, sep='\t', header=True, index=False)
