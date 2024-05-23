"""
Script that determines best marker combinations representing a cell type by
maximizing number of phenotypes detected, proportion of samples within a batch
displaying the phenotypes, number of cells within each sample displaying the
phenotypes and minimizing number of cells without phenotypes.
"""


# Import utility modules
import itertools as ite
import logging
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context


# Import local functions
from score_marker_combinations import score_marker_combinations  # AT. Double-check path
from utils import get_chunksize, setup_log  # AT. Double-check path
# from cellxhaustive.score_marker_combinations import score_marker_combinations
# from cellxhaustive.utils import get_chunksize, setup_log


# Convenience function to return specific dictionary values
def return_outputs(dict1, dict2, dict3, idx1, idx2):
    out1 = dict1[idx1[idx2[0]]]
    out2 = dict2[idx1[idx2[0]]]
    out3 = np.concatenate(dict3[idx1[idx2[0]]])
    return out1, out2, out3


# Function used in check_all_combinations()
def evaluate_comb(idx, comb, mat_representative, batches_label, samples_label,
                  markers_representative, two_peak_threshold, three_peak_markers,
                  three_peak_low, three_peak_high, min_annotations,
                  x_samplesxbatch_space, y_cellxsample_space, nb_cpu_keep):
    """
    Function that scores a marker combination and checks whether it contains
    relevant solutions depending on number of phenotypes detected, proportion of
    samples within batch displaying the phenotypes, number of cells within each
    sample displaying the phenotypes and number of cells without phenotypes.

    Parameters:
    -----------
    idx: int
      Integer index to keep track of 'comb'.

    comb: tuple(str)
      Tuple of strings marker combination to score.

    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_representative'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_representative'.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of 'mat_representative'.

    two_peak_threshold: float (default=3)
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means marker will
      be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
      low_positive or positive. Expression above this threshold means marker will
      be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

    min_annotations: int (default=3)
      Minimum number of phenotypes for a combination of markers to be taken into
      account as a potential cell population. Must be in '[2; len(markers)]',
      but it is advised to choose a value in '[3; len(markers) - 1]'.

    x_samplesxbatch_space: array(float) (default=[0.5, 0.6, 0.7, ..., 1])
      Minimum proportion of samples within each batch with at least
      'y_cellxsample_space' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10, 20...
      100 cells/sample (see description of next parameter) in at least 50%, 60%...
      100% of samples within a batch to be considered.

    y_cellxsample_space: array(float) (default=[10, 20, 30, ..., 100])
      Minimum number of cells within each sample in 'x_samplesxbatch_space' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10, 20...
      100 cells/sample in at least 50%, 60%... 100% of samples (see description
      of previous parameter) within a batch to be considered.

    nb_cpu_keep: int (default=1)
      Integer to set up CPU numbers in downstream nested functions.

    Returns:
    --------
    comb_result_dict: dict {str: obj}
      Dictionary with 1 or 8 key-value pairs. If no relevant solution was found,
      dictionary will have following structure {idx: None}. If relevant solution
      was found, keys will be 'idx', 'comb', 'phntp_per_cell', 'max_nb_phntp',
      'min_undefined', 'max_x_values', 'max_y_values', and 'best_phntp_lst'
    """

    # Set-up logging configuration
    setup_log(os.environ['LOG_FILE'], os.environ['LOG_LEVEL'], 'a')

    logging.debug(f'\t\t\t\tTesting {comb}')
    # Slice data based on current marker combination 'comb'
    markers_comb = markers_representative[np.isin(markers_representative, np.asarray(comb))]
    mat_comb = mat_representative[:, np.isin(markers_representative, markers_comb)]

    # Find number of phenotypes and undefined cells for a given marker combination
    # 'comb' across 'samplesxbatch' and 'cellxsample' grid
    logging.debug(f'\t\t\t\t\tScoring combination')
    nb_phntp, phntp_to_keep, nb_undef_cells, phntp_per_cell = score_marker_combinations(
        mat_comb=mat_comb,
        batches_label=batches_label,
        samples_label=samples_label,
        markers_comb=markers_comb,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
        x_samplesxbatch_space=x_samplesxbatch_space,
        y_cellxsample_space=y_cellxsample_space,
        nb_cpu_keep=nb_cpu_keep)

    # Constrain matrix given minimum number of phenotype conditions
    mask = (nb_phntp < min_annotations)
    nb_phntp = np.where(mask, np.nan, nb_phntp)
    nb_undef_cells = np.where(mask, np.nan, nb_undef_cells)

    # If there are possible good solutions, further process them
    logging.debug(f'\t\t\t\t\tChecking presence of good solutions')
    if np.any(np.isfinite(nb_phntp)):
        # Create metrics grid matching x and y space
        x_values, y_values = np.meshgrid(x_samplesxbatch_space,
                                         y_cellxsample_space,
                                         indexing='ij')

        # Constrain grid
        x_values = np.where(mask, np.nan, x_values)
        y_values = np.where(mask, np.nan, y_values)

        # Best solution has maximum number of new phenotypes...
        max_nb_phntp = np.nanmax(nb_phntp)
        nb_undef_cells[nb_phntp != max_nb_phntp] = np.nan

        # ... and minimum number of undefined cells...
        min_undefined = np.nanmin(nb_undef_cells)
        x_values[nb_undef_cells != min_undefined] = np.nan

        # ... and maximum percentage within batch
        max_x_values = np.nanmax(x_values)
        y_values[x_values != max_x_values] = np.nan

        # ... and maximum cells per sample
        max_y_values = np.nanmax(y_values)
        best_phntp_lst = phntp_to_keep[y_values == max_y_values]

        # Gather all results in dict
        comb_result_dict = {'idx': idx,
                            'comb': comb,
                            'phntp_per_cell': phntp_per_cell,
                            'max_nb_phntp': max_nb_phntp,
                            'min_undefined': min_undefined,
                            'max_x_values': max_x_values,
                            'max_y_values': max_y_values,
                            'best_phntp_lst': best_phntp_lst}

    else:  # No good solution, so return None to facilitate post-processing
        comb_result_dict = {idx: None}

    return comb_result_dict


# Function used in identify_phenotypes.py
def check_all_combinations(mat_representative, batches_label, samples_label,
                           markers_representative, two_peak_threshold,
                           three_peak_markers, three_peak_low, three_peak_high,
                           max_markers, min_annotations,
                           min_samplesxbatch, min_cellxsample, cpu_eval_keep):
    """
    Function that determines best marker combinations representing a cell type by
    maximizing number of phenotypes detected, proportion of samples within a batch
    displaying the phenotypes, number of cells within each sample displaying the
    phenotypes and minimizing number of cells without phenotypes.

    Parameters:
    -----------
    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_representative'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_representative'.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of 'mat_representative'.

    two_peak_threshold: float (default=3)
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means marker will
      be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
      low_positive or positive. Expression above this threshold means marker will
      be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

    max_markers: int (default=15)
      Maximum number of relevant markers to select among total list of markers
      from total markers array. Must be less than or equal to 'len(markers)'.

    min_annotations: int (default=3)
      Minimum number of phenotypes for a combination of markers to be taken into
      account as a potential cell population. Must be in '[2; len(markers)]',
      but it is advised to choose a value in '[3; len(markers) - 1]'.

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least 'min_cellxsample'
      cells for a new annotation to be considered. In other words, by default, an
      annotation needs to be assigned to at least 10 cells/sample (see description
      of next parameter) in at least 50% of samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of samples
      within each batch for a new annotation to be considered. In other words, by
      default, an annotation needs to be assigned to at least 10 cells/sample in at
      least 50% of samples (see description of previous parameter) within a batch
      to be considered.

    cpu_eval_keep: tuple(int) (default=(1, 1))
      Tuple of integers to set up CPU numbers in downstream nested functions.

    Returns:
    --------
    nb_solution: int
      Number of optimal combinations found when checking and comparing all possible
      marker combinations.

    best_marker_comb: tuple(str) or list(tuple(str))
      Tuple of strings or list of tuples of strings with optimal combinations
      found during comparison process. Each tuple contains one combination.
      Number of tuples in 'best_marker_comb' is equal to 'nb_solution'.

    cell_phntp_comb: array(str) or list(array(str))
      1-D numpy array of strings or list of 1-D numpy arrays of strings with phenotype
      found for each cell using markers from associated 'best_marker_comb' tuple.
      Number of arrays in 'cell_phntp_comb' is equal to 'nb_solution'.

    best_phntp_comb: array(str) or list(array(str))
      1-D numpy array of strings or list of 1-D numpy arrays of strings with
      representative phenotypes among all possible phenotypes from 'best_marker_comb'.
      Number of arrays in 'best_phntp_comb' is equal to 'nb_solution'.
    """

    # Set-up logging configuration
    setup_log(os.environ['LOG_FILE'], os.environ['LOG_LEVEL'], 'a')

    # Create total space for each metrics ('samplesxbatch' and 'cellxsample')
    logging.info('\t\t\tCreating spaces for each test metric')
    x_samplesxbatch_space = np.round(np.arange(min_samplesxbatch, 1.01, 0.01), 2)  # x-axis
    # Note: 'np.round()' is used to avoid floating point problem
    y_cellxsample_space = np.arange(min_cellxsample, 101)  # y-axis

    # Find theoretical maximum number of markers in combination
    max_combination = min(max_markers, len(markers_representative))

    # Initialise counters and objects to store results. Note that by default, it
    # is assumed that minimum number of relevant markers is 2 (only 1 marker can
    # not define a phenotype)
    enum_start = 0
    marker_counter = 2
    max_nb_phntp_marker = 0
    max_nb_phntp_tot = -1
    comb_dict = {}
    cell_phntp_dict = {}
    phntp_list_dict = {}

    # Empty arrays to store results and find best marker combinations
    best_comb_idx = np.empty(0)
    best_nb_phntp = np.empty(0)
    best_nb_undefined = np.empty(0)
    best_x_values = np.empty(0)
    best_y_values = np.empty(0)

    # Go through all combinations until no better solution can be found: stop
    # while loop if maximum number of markers is reached or if possible solution
    # using more markers are worse than current best. If loop isn't stopped, it
    # means scores can still be improved
    logging.info('\t\t\tTesting all combinations')
    while ((marker_counter <= max_combination)
           and (max_nb_phntp_tot < max_nb_phntp_marker)):

        # Save new higher (or equal) maximum number of phenotypes
        max_nb_phntp_tot = max_nb_phntp_marker

        # Get all possible combinations containing 'marker_counter' markers
        poss_comb = list(ite.combinations(markers_representative, marker_counter))
        # Note: iterator is converted to list because it is used several times

        # For a given number of markers, check all possible combinations using multiprocessing
        chunksize = get_chunksize(list(poss_comb), cpu_eval_keep[0])
        indices = range(enum_start, enum_start + len(poss_comb))
        with ProcessPoolExecutor(max_workers=cpu_eval_keep[0], mp_context=get_context('spawn')) as executor:
            score_results_lst = list(executor.map(partial(evaluate_comb,
                                                          mat_representative=mat_representative,
                                                          batches_label=batches_label,
                                                          samples_label=samples_label,
                                                          markers_representative=markers_representative,
                                                          two_peak_threshold=two_peak_threshold,
                                                          three_peak_markers=three_peak_markers,
                                                          three_peak_low=three_peak_low,
                                                          three_peak_high=three_peak_high,
                                                          min_annotations=min_annotations,
                                                          x_samplesxbatch_space=x_samplesxbatch_space,
                                                          y_cellxsample_space=y_cellxsample_space,
                                                          nb_cpu_keep=cpu_eval_keep[1]),
                                                  indices, poss_comb,
                                                  timeout=None,
                                                  chunksize=chunksize))
        # Notes: 'indices' and 'poss_comb' are iterated over, hence the use of
        # 'partial()' to keep the other parameters constant

        # Remove combinations without solution and turn list into dict using idx as keys
        score_results_dict = {}
        for dct in score_results_lst:
            if len(dct) > 1:
                idx = dct.pop('idx')
                score_results_dict[idx] = dct

        # Increase marker counter; it doesn't matter whether a solution is found
        marker_counter += 1

        # Reset enumerate start to avoid overwriting data in next iteration
        enum_start = len(score_results_lst)

        # Post-process results
        if len(score_results_dict) == 0:  # No combination is relevant, skip to next iteration
            max_nb_phntp_marker = 0  # Re-initialise counter of maximum phenotype
            continue
        else:  # At least one combination is relevant
            # Get maximum number of phenotypes with 'marker_counter' markers
            max_nb_phntp_marker = max(dct['max_nb_phntp'] for dct in score_results_dict.values())

            # Only process better results: if 'm' and 'm + 1' markers give same
            # number of phenotypes, keep only solutions with 'm' markers
            if max_nb_phntp_marker > max_nb_phntp_tot:
                # Filter out combinations not reaching maximum number of phenotype
                score_max_phntp = {k: v for k, v in score_results_dict.items() if v['max_nb_phntp'] == max_nb_phntp_marker}

                # Filter out combinations not reaching minimum number of undefined cells
                min_nb_undef = max(dct['min_undefined'] for dct in score_max_phntp.values())
                score_min_undef = {k: v for k, v in score_max_phntp.items() if v['min_undefined'] == min_nb_undef}

                # Filter out combinations not reaching maximum samplesxbatch
                max_x_val = max(dct['max_x_values'] for dct in score_min_undef.values())
                score_max_x = {k: v for k, v in score_min_undef.items() if v['max_x_values'] == max_x_val}

                # Filter out combinations not reaching maximum cellxsample
                max_y_val = max(dct['max_y_values'] for dct in score_max_x.values())
                score_final = {k: v for k, v in score_max_x.items() if v['max_y_values'] == max_y_val}

                # Save best results in general dictionaries and arrays
                comb_dict = {k: v['comb'] for k, v in score_final.items()}
                cell_phntp_dict = {k: v['phntp_per_cell'] for k, v in score_final.items()}
                phntp_list_dict = {k: v['best_phntp_lst'] for k, v in score_final.items()}
                best_comb_idx = np.fromiter((score_final.keys()), dtype=int)
                best_nb_phntp = np.fromiter((d['max_nb_phntp'] for d in score_final.values()), dtype=float)
                best_nb_undefined = np.fromiter((d['min_undefined'] for d in score_final.values()), dtype=float)
                best_x_values = np.fromiter((d['max_x_values'] for d in score_final.values()), dtype=float)
                best_y_values = np.fromiter((d['max_y_values'] for d in score_final.values()), dtype=float)

                # Free memory by deleting heavy objects
                del score_max_phntp, score_min_undef, score_max_x, score_final

    # If no marker combination was found, stop now
    if len(best_nb_phntp) == 0:
        nb_solution = 0
        best_marker_comb = ()
        cell_phntp_comb = np.empty(0)
        best_phntp_comb = np.empty(0)
        return nb_solution, best_marker_comb, cell_phntp_comb, best_phntp_comb

    # If several possible marker combinations were found, further refine results
    # according to metrics previously defined: number of phenotypes, number of
    # undefined cells, x and y values
    logging.info('\t\t\tRefining results to reduce number of possible combinations')

    # Find combination(s) with maximum number of phenotypes
    max_phntp_idx = np.where(best_nb_phntp == np.max(best_nb_phntp))[0]

    # Most likely only one solution, but it will be updated if there are more
    nb_solution = 1

    # Further refine results according to number of phntp
    if len(max_phntp_idx) > 1:  # Several combinations with maximum number of phenotypes
        # Subset arrays to keep combination(s) with maximum number of phenotypes
        best_comb_idx = best_comb_idx[max_phntp_idx]
        best_nb_undefined = best_nb_undefined[max_phntp_idx]
        # Find combination(s) with minimum number of undefined cells
        min_undefined_idx = np.where(best_nb_undefined == np.min(best_nb_undefined))[0]

        # Further refine results according to number of undefined cells
        if len(min_undefined_idx) > 1:  # Several combinations with minimum number of undefined cells
            # Subset arrays to keep combination(s) with minimum number of undefined cells
            best_comb_idx = best_comb_idx[min_undefined_idx]
            best_x_values = best_x_values[max_phntp_idx][min_undefined_idx]
            # Find combination(s) with maximum x value
            max_xvalues_idx = np.where(best_x_values == np.max(best_x_values))[0]

            # Further refine results according to x value
            if len(max_xvalues_idx) > 1:  # Several combinations with maximum x value
                # Subset arrays to keep combination(s) with maximum x value
                best_comb_idx = best_comb_idx[max_xvalues_idx]
                best_y_values = best_y_values[max_phntp_idx][min_undefined_idx][max_xvalues_idx]
                # Find combination(s) with maximum y value
                max_yvalues_idx = np.where(best_y_values == np.max(best_y_values))[0]
                nb_solution = len(max_yvalues_idx)

                if nb_solution == 1:
                    best_marker_comb, cell_phntp_comb, best_phntp_comb = return_outputs(comb_dict,
                                                                                        cell_phntp_dict,
                                                                                        phntp_list_dict,
                                                                                        best_comb_idx,
                                                                                        max_yvalues_idx)

                else:  # If there are several combinations, keep all of them
                    best_marker_comb = list(comb_dict.get(k)
                                            for k in best_comb_idx[max_yvalues_idx])
                    cell_phntp_comb = list(cell_phntp_dict.get(k)
                                           for k in best_comb_idx[max_yvalues_idx])
                    best_phntp_comb = list(np.concatenate(phntp_list_dict.get(k))
                                           for k in best_comb_idx[max_yvalues_idx])
                    # Note: 'np.concatenate' is used to convert an array of list
                    # into an array

            else:  # Only one combination with maximum x value
                best_marker_comb, cell_phntp_comb, best_phntp_comb = return_outputs(comb_dict,
                                                                                    cell_phntp_dict,
                                                                                    phntp_list_dict,
                                                                                    best_comb_idx,
                                                                                    max_xvalues_idx)

        else:  # Only one combination with minimum number of undefined cells
            best_marker_comb, cell_phntp_comb, best_phntp_comb = return_outputs(comb_dict,
                                                                                cell_phntp_dict,
                                                                                phntp_list_dict,
                                                                                best_comb_idx,
                                                                                min_undefined_idx)

    else:  # Only one combination with maximum number of phenotypes
        best_marker_comb, cell_phntp_comb, best_phntp_comb = return_outputs(comb_dict,
                                                                            cell_phntp_dict,
                                                                            phntp_list_dict,
                                                                            best_comb_idx,
                                                                            max_phntp_idx)

    return nb_solution, best_marker_comb, cell_phntp_comb, best_phntp_comb
