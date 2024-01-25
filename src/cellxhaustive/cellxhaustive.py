"""
# AT. Improve general description.
# AT. Add what is returned by the script

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Utility functions to run pipeline for phenotype identification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

--------------------------------------------------------------------
PURPOSE:
These are just the different components of the pipeline:
Main gating, selecting relevant markers, finding new phenotypes...
--------------------------------------------------------------------
NOTES:
Here I am sticking with the clean version of it, but I did try\
several versions of the same pipeline (see github history)
--------------------------------------------------------------------

Pipeline for automated gating, feature selection, and clustering to
generate new annotations.
# AT. Update description

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
parser = argparse.ArgumentParser(description='Script to annotate cell types using \
                                CITE-seq ADT data.')

parser.add_argument('-i', '--input', dest='input_path', type=str,
                    help='Path to input table with expression data and samples/\
                    batch information',
                    required=True)
parser.add_argument('-m', '--markers', dest='marker_path', type=str,
                    help='Path to file with list of markers of interest',
                    required=True)
parser.add_argument('-o', '--output', dest='output_path', type=str,
                    help='Path to output table with annotations',
                    required=True)
parser.add_argument('-t', '--three-peaks', dest='three_peak_markers', type=str,
                    help="Path to file with markers that have three peaks ['CD4']",
                    required=False, default=['CD4'])
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
if __name__ == '__main__':

    # Get 1-D array for markers
    markers = pd.read_csv(args.marker_path, sep='\t', header=None).to_numpy().flatten()

    # Parse general input files into several arrays
    input_table = pd.read_csv(args.input_path, sep='\t', header=0, index_col=0)

    # Get 2-D array for expression using 'markers'
    mat = input_table.loc[: , markers].to_numpy()

    # Get 1-D array for batch; add common batch value if information is missing
    if 'batch' in input_table.columns:
        batches = input_table.loc[: , 'batch'].to_numpy()
    else:
        batches = np.full(input_table.shape[0], 'batch0')

    # Get 1-D array for sample; add common sample value if information is missing
    if 'sample' in input_table.columns:
        samples = input_table.loc[: , 'sample'].to_numpy()
    else:
        samples = np.full(input_table.shape[0], 'sample0')

    # Get 1-D array for pre-annotated cell type
    cell_labels = input_table.loc[: , 'cell_type'].to_numpy()

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
    max_markers = args.max_markers
    min_annotations = args.min_annotations
    min_cellxsample = args.min_cellxsample
    min_samplesxbatch = args.min_samplesxbatch
    knn_refine = args.knn_refine
    knn_min_probability = args.knn_min_probability

    # Initialise empty array to store new annotations
    annotations = np.asarray(['undefined'] * len(cell_labels)).astype('U100')
    phenotypes = np.asarray(['undefined'] * len(cell_labels)).astype('U100')

    # Process cells by pre-annotations
    for label in np.unique(cell_labels):

        # Create boolean array to select cells matching current label
        is_label = (cell_labels == label)

        # Get annotations for all cells of type 'label'
        new_labels, cell_phntp_comb = identify_phenotypes(
            mat=mat,
            batches=batches,
            samples=samples,
            markers=markers,
            is_label=is_label,
            cell_types_dict=cell_types_dict,
            three_peak_markers=three_peak_markers,
            cell_name=label,
            max_markers=max_markers,
            min_annotations=min_annotations,
            min_cellxsample=min_cellxsample,
            min_samplesxbatch=min_samplesxbatch,
            knn_refine=knn_refine,
            knn_min_probability=knn_min_probability)

        # AT. Problem if there are several solutions...
        # Store results in arrays
        annotations[is_label] = new_labels  # AT. May need to rework this
        phenotypes[is_label] = cell_phntp_comb  # AT. May need to rework this

    # AT. Problem if there are several solutions...
    # Combine arrays into a dataframe
    new_col = pd.DataFrame({'Annotations': annotations, 'Phenotypes': phenotypes})
    output_table = pd.concat([input_table, new_col], axis=1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save table with annotations and phenotypes
    output_table.to_csv(args.output_path, sep='\t', header=True, index=False)
    # AT. Add na_rep in case of cells with several solutions, which means a different number of columns
