"""
Function that determines best marker combinations representing a cell type by
maximizing number of phenotypes detected, proportion of samples within a batch
displaying the phenotypes, number of cells within each sample displaying the
phenotypes and minimizing number of cells without phenotypes.
"""


# Import utility modules
import itertools as ite
import numpy as np


# Import local functions
from score_marker_combinations import score_marker_combinations  # AT. Double-check path
# from cellxhaustive.score_marker_combinations import score_marker_combinations


# Convenience function to return specific dict values
def return_outputs(dict1, dict2, dict3, idx1, idx2):
    out1 = dict1[idx1[idx2[0]]]
    out2 = dict2[idx1[idx2[0]]]
    out3 = np.concatenate(dict3[idx1[idx2[0]]])
    return out1, out2, out3


# Function used in identify_phenotypes.py
def check_all_combinations(mat_representative, batches_label, samples_label,
                           markers_representative, three_peak_markers=[],
                           max_markers=15, min_annotations=3,
                           min_samplesxbatch=0.5, min_cellxsample=10):
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

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

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

    Returns:
    --------
    nb_solution: int
      Number of optimal combinations found when checking and comparing all possible
      marker combinations.

    best_marker_comb: tuple or list(tuple)
      Tuple or list of tuples with optimal combinations found during comparison
      process. Each tuple contains one combination. Number of tuples in
      'best_marker_comb' is equal to 'nb_solution'.

    cell_phntp_comb: array(str) or list(array(str))
      1-D numpy array of strings or list of 1-D numpy arrays of strings showing
      phenotype found for each cell using markers from associated 'best_marker_comb'
      tuple. Number of arrays in 'cell_phntp_comb' is equal to 'nb_solution'.

    best_phntp_comb: array(str) or list(array(str))
      1-D numpy array of strings or list of 1-D numpy arrays of strings showing
      representative phenotypes among all possible phenotypes from 'best_marker_comb'.
      Number of arrays in 'best_phntp_comb' is equal to 'nb_solution'.
    """

    # Create total space for each metrics ('samplesxbatch' and 'cellxsample')
    x_samplesxbatch_space = np.round(np.arange(min_samplesxbatch, 1.01, 0.01), 2)  # x-axis
    # Note: 'np.round()' is used to avoid floating point problem
    y_cellxsample_space = np.arange(min_cellxsample, 101)  # y-axis

    # Find theoretical maximum number of markers in combination
    max_combination = min(max_markers, len(markers_representative))

    # Initialise counters and objects to store results. Note that by default, it
    # is assumed that minimum number of relevant markers is 2 (only 1 marker can
    # not define a phenotype)
    marker_counter = 2
    comb_idx = 0
    comb_dict = {}
    cell_phntp_dict = {}
    phntp_list_dict = {}
    max_nb_phntp_marker = 0
    max_nb_phntp_tot = 0

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
    while ((marker_counter <= max_combination)
           and (max_nb_phntp_tot <= max_nb_phntp_marker)):

        # Save new higher (or equal) maximum number of phenotypes
        max_nb_phntp_tot = max_nb_phntp_marker

        # Re-initialise counter of maximum phenotypes for combinations with
        # 'marker_counter' elements
        max_nb_phntp_marker = 0

        # For a given number of markers, check all possible marker combinations
        for comb in ite.combinations(markers_representative, marker_counter):
            # AT. Opportunity to multiprocess? Or not because we need to test 2
            # markers first, then 3, then 4... And stop if nothing better is found

            # Slice data based on current marker combination 'comb'
            markers_comb = markers_representative[np.isin(markers_representative, np.asarray(comb))]
            mat_comb = mat_representative[:, np.isin(markers_representative, markers_comb)]

            # Find number of phenotypes and undefined cells for a given marker
            # combination 'comb' across 'samplesxbatch' and 'cellxsample' grid
            nb_phntp, phntp_to_keep, nb_undef_cells, phntp_per_cell = score_marker_combinations(
                mat_comb=mat_comb,
                batches_label=batches_label,
                samples_label=samples_label,
                markers_comb=markers_comb,
                three_peak_markers=three_peak_markers,
                x_samplesxbatch_space=x_samplesxbatch_space,
                y_cellxsample_space=y_cellxsample_space)

            # Constrain matrix given minimum number of phenotype conditions
            mask = (nb_phntp < min_annotations)
            nb_phntp = np.where(mask, np.nan, nb_phntp)
            nb_undef_cells = np.where(mask, np.nan, nb_undef_cells)

            # If there are possible good solutions, further process them
            if np.any(np.isfinite(nb_phntp)):
                # Calculate maximum number of phenotypes for marker combination
                # 'comb'. This counter is useful for two things. 1. Determine
                # maximum number of phenotypes found across all combinations with
                # 'marker_counter' elements. 2. Avoid processing and keeping
                # 'comb' with a poor score: if 'comb' max is worse than recorded
                # best overall (meaning all 'comb' previously analysed), it
                # isn't worth keeping
                max_nb_phntp = np.nanmax(nb_phntp)
                if ((max_nb_phntp >= max_nb_phntp_marker)
                        and (max_nb_phntp >= max_nb_phntp_tot)):
                    max_nb_phntp_marker = max_nb_phntp
                else:
                    continue

                # If there are interesting phenotypes, store marker combination
                # as well as phenotype per cell
                comb_dict[comb_idx] = comb
                cell_phntp_dict[comb_idx] = phntp_per_cell

                # Create metrics grid matching x and y space
                x_values, y_values = np.meshgrid(x_samplesxbatch_space,
                                                 y_cellxsample_space,
                                                 indexing='ij')

                # Constrain grid
                x_values = np.where(mask, np.nan, x_values)
                y_values = np.where(mask, np.nan, y_values)

                # Calculate minimum number of undefined cells for 'comb'
                min_nb_undefined = np.nanmin(nb_undef_cells)

                # Best solution has maximum number of new phenotypes...
                best_comb_idx = np.append(best_comb_idx, comb_idx)  # Index of comb to use in 'comb_dict'
                best_nb_phntp = np.append(best_nb_phntp, max_nb_phntp)

                # ... and minimum number of undefined cells
                nb_undef_cells[nb_phntp != max_nb_phntp] = np.nan
                best_nb_undefined = np.append(best_nb_undefined, min_nb_undefined)

                # ... and maximum percentage within batch
                x_values[nb_undef_cells != min_nb_undefined] = np.nan
                best_x_values = np.append(best_x_values, np.nanmax(x_values))

                # ... and maximum cells per sample
                y_values[x_values != np.nanmax(x_values)] = np.nan
                best_y_values = np.append(best_y_values, np.nanmax(y_values))

                # Keep best phenotypes list
                best_phntp_lst = phntp_to_keep[y_values == np.nanmax(y_values)]
                phntp_list_dict[comb_idx] = best_phntp_lst

            comb_idx += 1

        marker_counter += 1

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

    # Find combination(s) with maximum number of phenotypes
    max_phntp_idx = np.where(best_nb_phntp == np.max(best_nb_phntp))[0]

    # Most liely, there will be only one solution. This variable will be updated
    # if there are more solutions
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

                    # Note: 'np.concatenate' is used to convert an array of list
                    # into an array
                else:  # If there are several combinations, keep all of them
                    best_marker_comb = list(comb_dict.get(k) for k in best_comb_idx[max_yvalues_idx])
                    cell_phntp_comb = list(cell_phntp_dict.get(k) for k in best_comb_idx[max_yvalues_idx])
                    best_phntp_comb = list(np.concatenate(phntp_list_dict.get(k)) for k in best_comb_idx[max_yvalues_idx])

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
