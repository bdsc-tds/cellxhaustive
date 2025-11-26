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


# Import local functions
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
      1-D numpy array with markers that must appear in optimal marker
      combinations.

    Returns:
    --------
    poss_comb: list(tuple(str))
      List of tuples of strings with marker combinations to score.
    """

    if len(markers_interest) > 0:  # With markers of interest
        # Determine number of representative markers to add
        missing_counter = marker_counter - len(markers_interest)
        # Generate combinations of representative markers
        complementation_comb = ite.combinations(
            markers_representative, missing_counter
        )
        # Append combinations of representative markers to markers of interest
        markers_interest_tuple = tuple(markers_interest)
        poss_comb = [markers_interest_tuple + cb for cb in complementation_comb]
    else:  # Without markers of interest
        # Generate combinations of 'marker_counter' representative markers
        poss_comb = list(
            ite.combinations(markers_representative, marker_counter)
        )
        # Note: iterator is converted to list because it is used several times

    return poss_comb


# Function used in check_all_combinations()
def evaluate_comb(
    idx,
    comb,
    cell_name,
    mat_representative,
    batches_label,
    samples_label,
    markers_representative,
    two_peak_threshold,
    three_peak_markers,
    three_peak_low,
    three_peak_high,
    min_samplesxbatch,
    min_cellxsample,
):
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

    cell_name: str
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

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
      Threshold to consider when determining whether a two peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three peaks marker is
      negative or low positive. Expression below this threshold means marker
      will be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three peaks marker is
      low_positive or positive. Expression above this threshold means marker
      will be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      'min_cellxsample' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of next parameter) in at least 50% of
      samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample in at least 50% of samples (see description of previous
      parameter) within a batch to be considered.

    Returns:
    --------
    comb_result_dict: dict({str: obj})
      Dictionary with 1 or 4 key-value pairs. If no relevant solution was found,
      dictionary will have following structure {'idx': None}. If relevant
      solution(s) was(were) found, keys will be 'idx', 'comb', 'nb_phntp', and
      'nb_undef_cells'.
    """

    comb_name = ", ".join(comb)
    logging.debug(f"\t\t\t{cell_name}: Testing ({comb_name})")
    # Slice data based on current marker combination 'comb'
    markers_mask = np.isin(markers_representative, comb)
    markers_comb = markers_representative[markers_mask]
    mat_comb = mat_representative[:, markers_mask]

    # Count number of three peaks markers in 'markers_comb'
    nb_three = np.isin(markers_comb, three_peak_markers).sum()

    # Find number of phenotypes and undefined cells for a given marker
    # combination 'comb' across 'samplesxbatch' and 'cellxsample' grid
    logging.debug(f"\t\t\t\t{cell_name} - ({comb_name}): Scoring combination")
    nb_phntp, nb_undef_cells = score_marker_combinations(
        cell_name=cell_name,
        comb_name=comb_name,
        mat_comb=mat_comb,
        batches_label=batches_label,
        samples_label=samples_label,
        markers_comb=markers_comb,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
        min_samplesxbatch=min_samplesxbatch,
        min_cellxsample=min_cellxsample,
    )

    # Further check combination and normalise number of phenotype for
    # three peaks markers
    logging.debug(
        f"\t\t\t\t{cell_name} - ({comb_name}): Normalising combination score"
    )
    # Check whether combination satisfies minimum number of phenotype threshold
    if nb_phntp < 3:
        # Not enough phenotypes, so return None to facilitate post-processing
        comb_result_dict = {"idx": None}
    else:
        # If needed, normalise given number of three peaks markers
        if nb_three > 0:
            nb_phntp = np.round(nb_phntp * ((2 / 3) ** nb_three))

        # Gather all results in dict
        comb_result_dict = {
            "idx": idx,
            "comb": comb,
            "nb_phntp": nb_phntp,
            "nb_undef_cells": nb_undef_cells,
        }

    return comb_result_dict


# Function used in identify_phenotypes.py
def check_all_combinations(
    cell_name,
    mat_representative,
    batches_label,
    samples_label,
    markers_representative,
    markers_interest,
    detection_method,
    two_peak_threshold,
    three_peak_markers,
    three_peak_low,
    three_peak_high,
    max_markers,
    min_samplesxbatch,
    min_cellxsample,
    processpool,
):
    """
    Function that determines best marker combinations representing a cell type
    by maximizing number of phenotypes detected, proportion of samples within a
    batch displaying the phenotypes, number of cells within each sample
    displaying the phenotypes and minimizing number of cells without phenotypes.

    Parameters:
    -----------
    cell_name: str
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

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
      1-D numpy array with markers that must appear in optimal marker
      combinations.

    detection_method: 'auto' or int
      Method used to stop search for optimal marker combinations. If 'auto', use
      default algorithm relying on maximum number of phenotypes. If int, create
      a combination with exactly this number of markers.

    two_peak_threshold: float (default=3)
      Threshold to consider when determining whether a two peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three peaks marker is
      negative or low positive. Expression below this threshold means marker
      will be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three peaks marker is
      low_positive or positive. Expression above this threshold means marker
      will be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

    max_markers: int (default=15)
      Maximum number of relevant markers to select among total list of markers
      from total markers array. Must be less than or equal to 'len(markers)'.

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      'min_cellxsample' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of next parameter) in at least 50% of
      samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample in at least 50% of samples (see description of previous
      parameter) within a batch to be considered.

    processpool: None or pathos.pools.ProcessPool object
      If not None, ProcessPool object to use in downstream nested functions.

    Returns:
    --------
    nb_solution: int
      Number of optimal combinations found when checking and comparing all
      possible marker combinations.

    best_marker_comb: tuple() or list(tuple(str))
      Empty tuple or list of tuple(s) of strings with optimal combination(s)
      found during comparison process. Each tuple contains one combination.
      Number of tuples in 'best_marker_comb' is equal to 'nb_solution'.
    """

    logging.info(
        f"\t\t{cell_name}: Setting start parameters from detection method and markers of interest"
    )
    # Split markers of interest from representative markers
    markers_rep_only = markers_representative[
        ~np.isin(markers_representative, markers_interest)
    ]
    # Set marker counter and maximum combination length
    if detection_method == "auto":  # Default algorithm for combinations length
        nb_mk_interest = len(markers_interest)
        # Theoretical maximum number of markers in combination
        max_combination = (
            min(max_markers, len(markers_rep_only)) + nb_mk_interest
        )
        # Account for markers of interest
        marker_counter = nb_mk_interest if nb_mk_interest > 0 else 2
    else:  # Combinations with exactly 'detection_method' markers
        marker_counter = max_combination = detection_method
    logging.info(
        f"\t\t\t{cell_name}: Set marker_counter to {marker_counter} and max_combination to {max_combination}"
    )

    # Initialise counters and objects to store results. Note that by default, it
    # is assumed that minimum number of relevant markers is 2 (a single marker
    # cannot define a phenotype)
    enum_start = 0  # Combination enumeration start index
    max_nb_phntp_marker = 0  # Maximum nb of phenotypes for specific comb length
    max_nb_phntp_tot = -1  # Overall maximum number of phenotypes
    comb_dict = {}  # Dictionary to store best combinations
    best_comb_idx = np.empty(0)  # Indices of best marker combinations
    # Note: also used to avoid crashes when no combination is found

    # Go through all combinations until no better solution can be found: stop
    # while loop if maximum number of markers is reached or if possible solution
    # using more markers are worse than current best. If loop isn't stopped, it
    # means scores can still be improved
    logging.info(f"\t\t{cell_name}: Testing all combinations")
    while (marker_counter <= max_combination) and (
        max_nb_phntp_tot < max_nb_phntp_marker
    ):
        # Save new higher (or equal) maximum number of phenotypes
        max_nb_phntp_tot = max_nb_phntp_marker

        # Get all possible combinations containing 'marker_counter' markers
        poss_comb = get_poss_comb(
            marker_counter, markers_rep_only, markers_interest
        )

        # Create new range of indices
        indices = range(enum_start, enum_start + len(poss_comb))

        # For a given number of markers, check all possible combinations
        if not processpool:  # Use for loop to avoid creating new processes
            score_results_lst = []
            for idx, comb in zip(indices, poss_comb):
                comb_result_dict = evaluate_comb(
                    idx=idx,
                    comb=comb,
                    cell_name=cell_name,
                    mat_representative=mat_representative,
                    batches_label=batches_label,
                    samples_label=samples_label,
                    markers_representative=markers_representative,
                    two_peak_threshold=two_peak_threshold,
                    three_peak_markers=three_peak_markers,
                    three_peak_low=three_peak_low,
                    three_peak_high=three_peak_high,
                    min_samplesxbatch=min_samplesxbatch,
                    min_cellxsample=min_cellxsample,
                )
                score_results_lst.append(comb_result_dict)
        else:  # Use ProcessPool to parallelise combination testing
            score_results_lst = list(
                processpool.map(
                    partial(
                        evaluate_comb,
                        cell_name=cell_name,
                        mat_representative=mat_representative,
                        batches_label=batches_label,
                        samples_label=samples_label,
                        markers_representative=markers_representative,
                        two_peak_threshold=two_peak_threshold,
                        three_peak_markers=three_peak_markers,
                        three_peak_low=three_peak_low,
                        three_peak_high=three_peak_high,
                        min_samplesxbatch=min_samplesxbatch,
                        min_cellxsample=min_cellxsample,
                    ),
                    indices,
                    poss_comb,
                )
            )
            # Note: 'partial()' is used to iterate over 'indices' and
            # 'poss_comb' and keep other parameters constant

        # Remove combinations without solution and turn list into dict using
        # combination indices as keys
        score_results_dict = {
            dct.pop("idx"): dct for dct in score_results_lst if len(dct) > 1
        }

        # Increase marker counter; it doesn't matter whether a solution is found
        marker_counter += 1

        # Increment enumerate start to avoid overwriting data in next iteration
        enum_start += len(poss_comb)

        # Post-process results
        # If no combination is relevant, re-initialise counter of maximum number
        # of phenotype and skip to next iteration
        if not score_results_dict:
            max_nb_phntp_marker = 0
            continue

        # If at least one combination is relevant, get maximum number of
        # phenotypes with 'marker_counter' markers
        max_nb_phntp_marker = max(
            dct["nb_phntp"] for dct in score_results_dict.values()
        )

        # Only process better results: if 'm' and 'm + 1' markers give same
        # number of phenotypes, keep only solutions with 'm' markers
        if max_nb_phntp_marker > max_nb_phntp_tot:
            # Filter out combinations not reaching maximum number of phenotypes
            score_max_phntp = {
                indx: val
                for indx, val in score_results_dict.items()
                if val["nb_phntp"] == max_nb_phntp_marker
            }

            # Get minimum number of undefined cells
            min_nb_undef = min(
                dct["nb_undef_cells"] for dct in score_max_phntp.values()
            )
            # Filter out combinations not reaching minimum number of undefined
            # cells
            final_score = {
                indx: val
                for indx, val in score_max_phntp.items()
                if val["nb_undef_cells"] == min_nb_undef
            }

            # Save best results in general dictionaries and arrays
            final_values = list(final_score.values())
            comb_dict = {indx: v["comb"] for indx, v in final_score.items()}
            best_comb_idx = np.fromiter(final_score.keys(), dtype=int)
            best_nb_phntp = np.array(
                [dct["nb_phntp"] for dct in final_values], dtype=float
            )
            best_nb_undefined = np.array(
                [dct["nb_undef_cells"] for dct in final_values], dtype=float
            )

            # Free memory by deleting heavy objects
            del score_max_phntp, final_score, final_values

    logging.info(f"\t\t\t{cell_name}: All combinations checked")

    # Final post-processing of best results
    nb_comb = len(best_comb_idx)
    if nb_comb == 0:
        # No marker combination was found, stop now
        logging.info(f"\t\t{cell_name}: No optimal marker combination found")
        nb_solution = 0
        best_marker_comb = ()
    elif nb_comb == 1:
        # Only one combination, no need to further filter results
        logging.info(f"\t\t{cell_name}: 1 optimal marker combination found")
        nb_solution = 1
        best_marker_comb = list(comb_dict.values())
    else:
        # Several combinations, further filter results according to number of
        # phenotypes and number of undefined cells
        logging.info(
            f"\t\t{cell_name}: {nb_comb} optimal marker combinations found"
        )
        logging.info(
            f"\t\t\t{cell_name}: Filtering results to reduce number of combinations"
        )
        # Find combination(s) with maximum number of phenotypes
        final_idx = best_nb_phntp == best_nb_phntp.max()

        # If several combinations remain, filter by minimum number of undefined
        # cells
        if final_idx.sum() > 1:
            # Subset arrays to keep only max phenotype combinations
            filtered_comb_idx = best_comb_idx[final_idx]
            filtered_undefined = best_nb_undefined[final_idx]
            # Find combination(s) with minimum number of undefined cells
            min_undefined_mask = filtered_undefined == filtered_undefined.min()
            # Get final combination(s)
            nb_solution = min_undefined_mask.sum()
            best_marker_comb = [
                comb_dict[k] for k in filtered_comb_idx[min_undefined_mask]
            ]
        else:
            # Only one combination after maximum phenotype filtering
            nb_solution = 1
            best_marker_comb = [comb_dict[best_comb_idx[final_idx][0]]]

        str1 = "s" if nb_solution > 1 else ""
        logging.info(
            f"\t\t\t{cell_name}: {nb_solution} combination{str1} found"
        )

    return nb_solution, best_marker_comb
