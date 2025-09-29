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
import logging
import numpy as np
import os
import pandas as pd
import sys
from functools import partial
from pathos.pools import ProcessPool, ThreadPool


# Import other functions from package
from cli_parser import get_default_parser, validate_cli  # AT. Double-check path
from identify_phenotypes import identify_phenotypes  # AT. Double-check path
from utils import get_repr_markers  # AT. Double-check path
# from cellxhaustive.cli_parser import get_default_parser, validate_cli
# from cellxhaustive.identify_phenotypes import identify_phenotypes
# from cellxhaustive.utils import get_repr_markers


# Function to run cellxhaustive; used in if __name__ == '__main__'
def main():
    """
    Main function to run cellxhaustive package.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    # Save memory by forcing Copy on Write mode
    pd.options.mode.copy_on_write = True

    # Initialise parser and its arguments
    parser = get_default_parser()

    # Parse CLI arguments to fill Namespace
    args = parser.parse_args()

    # Validate arguments and parse config file
    args_dict = validate_cli(args)

    # Unpack argument dictionary to make it easier to work with
    batches = args_dict['batches']
    cell_labels = args_dict['cell_labels']
    cell_types_dict = args_dict['cell_types_dict']
    detection_method_lst = args_dict['detection_method_lst']
    input_table = args_dict['input_table']
    knn_min_probability = args_dict['knn_min_probability']
    knn_refine = args_dict['knn_refine']
    markers = args_dict['markers']
    markers_interest_lst = args_dict['markers_interest_lst']
    mask_interest_inv_lst = args_dict['mask_interest_inv_lst']
    mat = args_dict['mat']
    max_markers = args_dict['max_markers']
    min_cellxsample = args_dict['min_cellxsample']
    min_samplesxbatch = args_dict['min_samplesxbatch']
    nb_cpu_eval = args_dict['nb_cpu_eval']
    nb_cpu_id = args_dict['nb_cpu_id']
    output_path = args_dict['output_path']
    samples = args_dict['samples']
    three_peak_high = args_dict['three_peak_high']
    three_peak_low = args_dict['three_peak_low']
    three_peak_markers = args_dict['three_peak_markers']
    two_peak_threshold = args_dict['two_peak_threshold']
    uniq_labels = args_dict['uniq_labels']

    # Get multiple population status
    multipop = (False if len(uniq_labels) == 1 else True)

    # Get list of arrays describing cells matching each cell type of 'uniq_labels'
    is_label_lst = [(cell_labels == label) for label in uniq_labels]

    logging.info('Gating markers and extracting associated data in each cell type')
    # Determine path for marker filtering results
    marker_file_path = f'{os.path.splitext(output_path)[0]}_markers.txt'

    # Initialise list of arrays to store results
    mat_subset_rep_lst = []
    batches_label_lst = []
    samples_label_lst = []
    markers_representative_lst = []

    # Specific context manager to save information on marker filtering
    with open(marker_file_path, 'w') as file:
        # Aim is to select relevant markers in each batch of each cell type
        for label, is_label, \
                markers_interest, mask_interest_inv in zip(uniq_labels,
                                                           is_label_lst,
                                                           markers_interest_lst,
                                                           mask_interest_inv_lst):  # Loop over cell types
            logging.info(f"\tProcessing {label}' cells'")
            markers_rep_batches = []
            label_unique_batch = np.unique(batches[is_label])  # Get batch names in 'label' cells
            nb_batch = len(label_unique_batch)  # Get number of batch in 'label' cells
            for batch in label_unique_batch:
                logging.debug(f"\t\tProcessing batch '{batch}'")
                logging.debug('\t\t\tSelecting matching cells')
                is_batch = (batches == batch)
                is_label_batch = np.logical_and(is_batch, is_label)
                logging.debug('\t\t\tSelecting corresponding expression data')
                try:
                    mat_subset = mat[is_label_batch, :][:, mask_interest_inv]  # With markers of interest
                except NameError:
                    mat_subset = mat[is_label_batch, :]  # Without markers of interest
                logging.debug('\t\t\tEvaluating marker bimodality')
                marker_center_values = np.abs(np.mean(mat_subset, axis=0) - two_peak_threshold)
                marker_threshold_value = np.sort(marker_center_values)[max_markers - 1]  # Select max_markers-th center value (in ascending order)
                # Note: '- 1' is used to compensate 0-based indexing
                # Select markers with center smaller than max_markers-th center
                is_center_smaller = (marker_center_values <= marker_threshold_value)
                try:
                    markers_rep = markers[mask_interest_inv][is_center_smaller]  # With markers of interest
                except NameError:
                    markers_rep = markers[is_center_smaller]  # Without markers of interest
                logging.debug('\t\t\tStoring relevant markers')
                markers_rep_batches.extend(list(markers_rep))

            if nb_batch == 1:  # No need to filter markers if there is only 1 batch
                logging.info('\t\tOnly one batch in total, no need to filter markers')
                markers_representative, missing_markers, nb_batch_end = markers_rep_batches, [], 1
                logging.info(f"\t\t\tFound {len(markers_representative)} markers: {', '.join(markers_representative)}")
            else:
                logging.info(f'\t\t{nb_batch} batches in total. Selecting markers present in all batches')
                markers_representative, missing_markers, nb_batch_end = get_repr_markers(markers_rep_batches,
                                                                                         nb_batch)

            # Convert format back to array
            markers_representative = np.array(list(markers_representative))

            # Save marker information to file
            file.write(f'{label} cells:\n')
            file.write(f"\tMarkers found: {', '.join(markers_rep_batches)}\n")
            file.write(f"\tMarkers kept (found in {nb_batch_end} batches): {', '.join(markers_representative)}\n")
            file.write(f"\tMarkers removed (found in less than {nb_batch_end} batches): {', '.join(missing_markers)}\n\n")

            # If 'auto' detection method isn't used, check that number of
            # representative markers is bigger than requested combination length
            label_idx = uniq_labels.tolist().index(label)
            if detection_method_lst[label_idx] != 'auto':
                if detection_method_lst[label_idx] > len(markers_representative):
                    logging.error(f"\t\tNumber of markers kept is less than detection_method value for cell type '{label}'")
                    logging.error(f'\t\tNumber of representative markers: {len(markers_representative)}')
                    logging.error(f'\t\tRequested combination length: {detection_method_lst[label_idx]}')
                    logging.error("\t\tTry to decrease detection_method value for this cell type or use 'auto' mode")
                    sys.exit(1)

            # Extract expression, batch and sample information across all batches
            # for cell type 'label'
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
            mat_subset_rep_lst.append(mat_subset_rep_markers)  # Store slice matrix
            batches_label_lst.append(batches_label)  # Store batch information
            samples_label_lst.append(samples_label)  # Store sample information
            markers_representative_lst.append(markers_representative)  # Store representative markers
            # Note: keeping data in list of arrays makes multiprocessing easier and
            # reduce overall memory usage

    # Free memory by deleting heavy objects
    del (is_batch, is_label_batch, mat_subset, mat_subset_label, mat, batches, samples)

    # Initialise ProcessPool if needed, otherwise set it to None
    processpool = (None if nb_cpu_eval == 1 else ProcessPool(ncpus=nb_cpu_eval))

    # Process cells by pre-existing annotations
    if nb_cpu_id == 1:  # Use for loop to avoid creating new processes
        logging.info('Starting analyses without parallelisation')
        annot_results_lst = []
        for is_label, cell_name, mat_representative, \
                batches_label, samples_label, markers_representative, \
                markers_interest, detection_method in zip(is_label_lst,
                                                          uniq_labels,
                                                          mat_subset_rep_lst,
                                                          batches_label_lst,
                                                          samples_label_lst,
                                                          markers_representative_lst,
                                                          markers_interest_lst,
                                                          detection_method_lst):
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
        logging.info('Starting analyses with parallelisation')
        with ThreadPool(nthreads=nb_cpu_id) as threadpool:
            annot_results_lst = list(threadpool.map(partial(identify_phenotypes,
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
                                                            processpool=processpool),
                                                    is_label_lst,
                                                    uniq_labels,
                                                    mat_subset_rep_lst,
                                                    batches_label_lst,
                                                    samples_label_lst,
                                                    markers_representative_lst,
                                                    markers_interest_lst,
                                                    detection_method_lst))
        # Note: 'partial()' is used to iterate over 'is_label_lst', 'uniq_labels',
        # 'mat_subset_rep_lst', 'batches_label_lst', 'samples_label_lst',
        # 'markers_representative_lst', 'markers_interest_lst' and
        # 'detection_method_lst' and keep other parameters constant

    # Make sure ProcessPool is closed. It should already be, but this makes sure
    # there are no zombie processes
    if processpool:
        processpool.close()
        processpool.join()
    del processpool

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
    for label, is_label in zip(uniq_labels, is_label_lst):
        logging.info(f"\t\tCreating result subtable for '{label}' annotations")

        # Slice general results dictionary
        logging.debug('\t\t\tSelecting associated results')
        sub_results = annot_dict[label]

        # Find number of optimal combinations for 'label' cells
        logging.info('\t\t\tDetermining maximum number of optimal combinations')
        label_nb_comb = list(sub_results.keys())
        logging.info(f"\t\t\t\tFound {len(label_nb_comb)} combination{'s' if len(label_nb_comb) > 1 else ''}")

        # Get number of cells
        logging.info('\t\t\tCounting cells')
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
                                      columns=col_names_sub,
                                      dtype=object)

        # Create dataframe results for all optimal combinations of 'label'
        logging.debug('\t\t\tFilling table')
        for idx, comb_nb in enumerate(label_nb_comb):
            # Extract results
            sub_res_df = pd.DataFrame.from_dict(sub_results[comb_nb], orient='index').transpose()

            # Fill 'label' annotation dataframe
            start = 5 * idx
            annot_df_label.iloc[:, start:(start + 5)] = sub_res_df

        # Fill general annotation dataframe with 'label' annotations
        logging.info('\t\t\tIntegrating subtable to general annotation table')
        annot_df.iloc[is_label, :end] = annot_df_label.copy(deep=True)

    # Merge input dataframe and annotation dataframe
    logging.info('\tMerging input data and annotation table')
    annot_df.set_index(input_table.index, inplace=True)
    # Note: set indices to avoid problem during concatenation
    output_table = pd.concat([input_table, annot_df], axis=1)

    # Create output directory if not empty and missing
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.debug(f"Creating output directory '{output_dir}'")
        os.makedirs(output_dir, exist_ok=True)

    # Save general table with annotations and phenotypes
    logging.info(f"Saving final table to '{output_path}'")
    output_table.to_csv(output_path, sep='\t', header=True, index=True)


# Script execution
if __name__ == '__main__':
    main()
