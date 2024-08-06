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
from functools import partial


# Import other function from package
from score_marker_combinations import score_marker_combinations  # AT. Double-check path
# from cellxhaustive.score_marker_combinations import score_marker_combinations


# Function used in check_all_combinations()
def get_poss_comb(marker_counter, markers_representative, markers_interest):
    """
    Function that takes into account presence of markers of interest to generate
    marker combinations to score.

    Parameters:
    -----------
    marker_counter: int
      Number of markers in combinations to create.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of 'mat_representative'.

    markers_interest: array(str) or empty array
      1-D numpy array with markers that must appear in optimal marker combinations.

    Returns:
    --------
    poss_comb: list(tuple(str))
      List of tuples of strings with marker combinations to score.
    """

    if len(markers_interest) > 0:  # With markers of interest
        # Determine number of representative markers to add
        missing_counter = marker_counter - len(markers_interest)
        # Generate combinations of representative markers
        complementation_comb = ite.combinations(markers_representative, missing_counter)
        # Append combinations of representative markers to markers of interest
        poss_comb = [tuple(markers_interest) + cb for cb in complementation_comb]
    else:  # Without markers of interest
        # Generate combinations of 'marker_counter' representative markers
        poss_comb = list(ite.combinations(markers_representative, marker_counter))
        # Note: iterator is converted to list because it is used several times

    return poss_comb


# Function used in check_all_combinations()
def evaluate_comb(idx, comb, mat_representative, batches_label, samples_label,
                  markers_representative, two_peak_threshold, three_peak_markers,
                  three_peak_low, three_peak_high,
                  x_samplesxbatch_space, y_cellxsample_space):
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
      Tuple of strings with marker combination to score.

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

    Returns:
    --------
    comb_result_dict: dict({str: obj})
      Dictionary with 1 or 8 key-value pairs. If no relevant solution was found,
      dictionary will have following structure {'idx': None}. If relevant solution
      was found, keys will be 'idx', 'comb', 'max_nb_phntp', 'min_undefined',
      'max_x_values', and 'max_y_values'
    """

    logging.debug(f"\t\t\t\tTesting ({', '.join(comb)})")
    # Slice data based on current marker combination 'comb'
    markers_mask = np.isin(markers_representative, np.asarray(comb))
    markers_comb = markers_representative[markers_mask]
    mat_comb = mat_representative[:, markers_mask]

    # Find number of phenotypes and undefined cells for a given marker combination
    # 'comb' across 'samplesxbatch' and 'cellxsample' grid
    logging.debug('\t\t\t\t\tScoring combination')
    nb_phntp, nb_undef_cells = score_marker_combinations(
        mat_comb=mat_comb,
        batches_label=batches_label,
        samples_label=samples_label,
        markers_comb=markers_comb,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
        x_samplesxbatch_space=x_samplesxbatch_space,
        y_cellxsample_space=y_cellxsample_space)

    # Constrain matrix given minimum number of phenotype conditions
    logging.debug('\t\t\t\t\tChecking presence of possible solutions')
    mask = (nb_phntp < 3)
    nb_phntp = np.where(mask, np.nan, nb_phntp)
    nb_undef_cells = np.where(mask, np.nan, nb_undef_cells)

    # If there are possible good solutions, further process them
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

        # Gather all results in dict
        comb_result_dict = {'idx': idx,
                            'comb': comb,
                            'max_nb_phntp': max_nb_phntp,
                            'min_undefined': min_undefined,
                            'max_x_values': max_x_values,
                            'max_y_values': max_y_values}

    else:  # No good solution, so return None to facilitate post-processing
        comb_result_dict = {'idx': None}

    return comb_result_dict


# Function used in identify_phenotypes.py
def check_all_combinations(mat_representative, batches_label, samples_label,
                           markers_representative, markers_interest,
                           detection_method, two_peak_threshold,
                           three_peak_markers, three_peak_low, three_peak_high,
                           max_markers, min_samplesxbatch, min_cellxsample, processpool):
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

    markers_interest: array(str) or empty array
      1-D numpy array with markers that must appear in optimal marker combinations.

    detection_method: 'auto' or int
      Method used to stop search for optimal marker combinations. If 'auto', use
      default algorithm relying on maximum number of phenotypes. If int, create a
      combination with exactly this number of markers.

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

    processpool: None or pathos.pools.ProcessPool object
      If not None, ProcessPool object to use in downstream nested functions.

    Returns:
    --------
    nb_solution: int
      Number of optimal combinations found when checking and comparing all possible
      marker combinations.

    best_marker_comb: tuple(str) or list(tuple(str))
      Tuple of strings or list of tuples of strings with optimal combinations
      found during comparison process. Each tuple contains one combination.
      Number of tuples in 'best_marker_comb' is equal to 'nb_solution'.
    """

    # Create total space for each metrics ('samplesxbatch' and 'cellxsample')
    logging.info('\t\t\tCreating spaces for each test metric')
    x_samplesxbatch_space = np.round(np.arange(min_samplesxbatch, 1.01, 0.01), 2)  # x-axis
    # Note: 'np.round()' is used to avoid floating point problem
    y_cellxsample_space = np.arange(min_cellxsample, 101)  # y-axis

    logging.info('\t\t\tSetting start parameters from detection method and markers of interest')
    # Split markers of interest from representative markers
    markers_rep_only = markers_representative[np.isin(markers_representative,
                                                      markers_interest,
                                                      invert=True)]
    if detection_method == 'auto':  # Default algorithm for combinations length
        # Theoretical maximum number of markers in combination
        max_combination = min(max_markers, len(markers_rep_only))
        if len(markers_interest) > 0:  # With markers of interest
            marker_counter = len(markers_interest)
            max_combination += len(markers_interest)  # Account for markers of interest
        else:  # Without markers of interest
            marker_counter = 2
    else:  # Combinations with exactly 'detection_method' markers
        marker_counter = max_combination = detection_method
    logging.info(f'\t\t\t\tSet marker_counter to {marker_counter} and max_combination to {max_combination}')

    # Initialise counters and objects to store results. Note that by default, it
    # is assumed that minimum number of relevant markers is 2 (only 1 marker can
    # not define a phenotype)
    enum_start = 0
    max_nb_phntp_marker = 0
    max_nb_phntp_tot = -1
    comb_dict = {}

    # Empty arrays to store results and find best marker combinations
    best_nb_phntp = np.empty(0)

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
        poss_comb = get_poss_comb(marker_counter, markers_rep_only, markers_interest)

        # Create new range of indices
        indices = range(enum_start, enum_start + len(poss_comb))

        # For a given number of markers, check all possible combinations
        if not processpool:  # Use for loop to avoid creating new processes
            score_results_lst = []
            for idx, comb in zip(indices, poss_comb):
                comb_result_dict = evaluate_comb(idx=idx,
                                                 comb=comb,
                                                 mat_representative=mat_representative,
                                                 batches_label=batches_label,
                                                 samples_label=samples_label,
                                                 markers_representative=markers_representative,
                                                 two_peak_threshold=two_peak_threshold,
                                                 three_peak_markers=three_peak_markers,
                                                 three_peak_low=three_peak_low,
                                                 three_peak_high=three_peak_high,
                                                 x_samplesxbatch_space=x_samplesxbatch_space,
                                                 y_cellxsample_space=y_cellxsample_space)
                score_results_lst.append(comb_result_dict)
        else:  # Use ProcessPool to parallelise combination testing
            score_results_lst = list(processpool.map(partial(evaluate_comb,
                                                             mat_representative=mat_representative,
                                                             batches_label=batches_label,
                                                             samples_label=samples_label,
                                                             markers_representative=markers_representative,
                                                             two_peak_threshold=two_peak_threshold,
                                                             three_peak_markers=three_peak_markers,
                                                             three_peak_low=three_peak_low,
                                                             three_peak_high=three_peak_high,
                                                             x_samplesxbatch_space=x_samplesxbatch_space,
                                                             y_cellxsample_space=y_cellxsample_space),
                                                     indices, poss_comb))
            # Note: 'partial()' is used to iterate over 'indices' and 'poss_comb'
            # and keep other parameters constant

        # Remove combinations without solution and turn list into dict using
        # combination indices as keys
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
            max_nb_phntp_marker = 0  # Re-initialise counter of maximum number of phenotype
            continue
        else:  # At least one combination is relevant
            # Get maximum number of phenotypes with 'marker_counter' markers
            max_nb_phntp_marker = max(dct['max_nb_phntp'] for dct in score_results_dict.values())

            # Only process better results: if 'm' and 'm + 1' markers give same
            # number of phenotypes, keep only solutions with 'm' markers
            if max_nb_phntp_marker > max_nb_phntp_tot:
                # Filter out combinations not reaching maximum number of phenotype
                score_max_phntp = {indx: v for indx, v in score_results_dict.items() if v['max_nb_phntp'] == max_nb_phntp_marker}

                # Filter out combinations not reaching minimum number of undefined cells
                min_nb_undef = max(dct['min_undefined'] for dct in score_max_phntp.values())
                score_min_undef = {indx: v for indx, v in score_max_phntp.items() if v['min_undefined'] == min_nb_undef}

                # Filter out combinations not reaching maximum samplesxbatch
                max_x_val = max(dct['max_x_values'] for dct in score_min_undef.values())
                score_max_x = {indx: v for indx, v in score_min_undef.items() if v['max_x_values'] == max_x_val}

                # Filter out combinations not reaching maximum cellxsample
                max_y_val = max(dct['max_y_values'] for dct in score_max_x.values())
                score_final = {indx: v for indx, v in score_max_x.items() if v['max_y_values'] == max_y_val}

                # Save best results in general dictionaries and arrays
                comb_dict = {indx: v['comb'] for indx, v in score_final.items()}
                best_comb_idx = np.fromiter(score_final.keys(), dtype=int)
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
        return nb_solution, best_marker_comb

    # If several possible marker combinations were found, further refine results
    # according to metrics previously defined: number of phenotypes, number of
    # undefined cells, x and y values
    logging.info('\t\t\tRefining results to reduce number of possible combinations')

    # Find combination(s) with maximum number of phenotypes
    max_phntp_idx = np.where(best_nb_phntp == np.max(best_nb_phntp))[0]
    final_idx = max_phntp_idx

    # Most likely only one solution, but it will be updated if there are more
    nb_solution = 1

    # Further refine results according to number of phntp
    if len(max_phntp_idx) > 1:  # Several combinations with maximum number of phenotypes
        # Subset arrays to keep combination(s) with maximum number of phenotypes
        best_comb_idx = best_comb_idx[max_phntp_idx]
        best_nb_undefined = best_nb_undefined[max_phntp_idx]
        # Find combination(s) with minimum number of undefined cells
        min_undefined_idx = np.where(best_nb_undefined == np.min(best_nb_undefined))[0]
        final_idx = min_undefined_idx  # Store new final indices

        # Further refine results according to number of undefined cells
        if len(min_undefined_idx) > 1:  # Several combinations with minimum number of undefined cells
            # Subset arrays to keep combination(s) with minimum number of undefined cells
            best_comb_idx = best_comb_idx[min_undefined_idx]
            best_x_values = best_x_values[max_phntp_idx][min_undefined_idx]
            # Find combination(s) with maximum x value
            max_xvalues_idx = np.where(best_x_values == np.max(best_x_values))[0]
            final_idx = max_xvalues_idx  # Store new final indices

            # Further refine results according to x value
            if len(max_xvalues_idx) > 1:  # Several combinations with maximum x value
                # Subset arrays to keep combination(s) with maximum x value
                best_comb_idx = best_comb_idx[max_xvalues_idx]
                best_y_values = best_y_values[max_phntp_idx][min_undefined_idx][max_xvalues_idx]
                # Find combination(s) with maximum y value
                max_yvalues_idx = np.where(best_y_values == np.max(best_y_values))[0]
                final_idx = max_yvalues_idx  # Store new final indices
                nb_solution = len(max_yvalues_idx)  # If number of solution is still not 1

    # Keep remaining solution(s) that satisfied all previous conditions
    best_marker_comb = list(comb_dict.get(k)
                            for k in best_comb_idx[final_idx])

    return nb_solution, best_marker_comb
