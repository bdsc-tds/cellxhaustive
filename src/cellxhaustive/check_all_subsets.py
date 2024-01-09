"""
Function that determines the best marker combinations representing a cell type
by maximizing the number of phenotypes detected, the proportion of samples
within a batch containing the phenotypes, the number of cells within each sample
containing the phenotypes and minimizing the number of cells without phenotypes.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports local functions
from marker_combinations_scoring import marker_combinations_scoring  # AT. Double-check path
# from cellxhaustive.marker_combinations_scoring import marker_combinations_scoring


# Function used in identify_phenotypes.py  # AT. Update script name if needed
def check_all_combinations(mat_representative, batches_label, samples_label,
                           markers_representative, max_markers=15, min_annotations=3,
                           min_samplesxbatch=0.5, min_cellxsample=10):
    """
    Function that determines the best marker combinations representing a cell
    type by maximizing the number of phenotypes detected, the proportion of
    samples within a batch containing the phenotypes, the number of cells
    within each sample containing the phenotypes and minimizing the number of
    cells without phenotypes.

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

    max_markers: int (default=15)
      Maximum number of relevant markers to select among the total list of
      markers from the markers array. Must be less than or equal to 'len(markers)'.

    min_annotations: int (default=3)
      Minimum number of phenotypes for a combination of markers to be taken into
      account as a potential cell population. Must be in [2; len(markers)], but
      it is advised to choose a value in [3; len(markers) - 1].

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      'min_cellxsample' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of previous parameter) in at least 50% of
      the samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      10 cells/sample in at least 50% of the samples (see description of next
      parameter) within a batch to be considered.

    Returns:
    --------
      best_marker_comb: list(tuple)
        List of tuples containing the best marker combinations representing a
        cell type. Each tuple is a combination.
    """

    # Create total space for each metrics (samplesxbatch and cellxsample)
    x_samplesxbatch_space = np.round(np.arange(min_samplesxbatch, 1.01, 0.01), 2)  # x-axis
    # Note: we use np.round() to avoid floating point problem
    y_cellxsample_space = np.arange(min_cellxsample, 101)  # y-axis

    # Find theoretical maximum number of markers in combination
    max_combination = min(max_markers, len(markers_representative))

    # Initialize counters and objects to store results. Note that by default, we
    # assume that the minimum number of relevant markers is 2 (i.e. only 1 marker
    # can not define a phenotype)
    marker_counter = 2
    comb_idx = 0
    comb_dict = dict()
    max_nb_phntp = 0
    max_nb_phntp_comb = 0

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
    while (marker_counter <= max_combination) and (max_nb_phntp <= max_nb_phntp_comb):

        # Save new higher (or equal) maximum number of phenoytypes
        max_nb_phntp = max_nb_phntp_comb

        # For a given number of markers, check all possible marker combinations
        for comb in ite.combinations(markers_representative, marker_counter):
            # AT. Opportunity to multiprocess? Or not because we need to test 2
            # markers first, then 3, then 4... And stop if nothing better is found

            # Slice data based on current marker combination 'comb'
            markers_comb = markers_representative[np.isin(markers_representative, np.asarray(comb))]
            mat_comb = mat_representative[:, np.isin(markers_representative, markers_comb)]

            # Find number of phenotypes and undefined cells for a given marker
            # combination 'comb' across 'samplesxbatch' and 'cellxsample' grid
            nb_phntp, nb_undef_cells = marker_combinations_scoring(
                mat_comb=mat_comb,
                markers_comb=markers_comb,
                batches_label=batches_label,
                samples_label=samples_label,
                x_samplesxbatch_space=x_samplesxbatch_space,
                y_cellxsample_space=y_cellxsample_space,
                three_peak_markers=['CD4'])

            # Constrain matrix given minimum number of phenotype conditions
            mask = (nb_phntp < min_annotations)
            nb_phntp = np.where(mask, np.nan, nb_phntp)
            nb_undef_cells = np.where(mask, np.nan, nb_undef_cells)

            # If there are possible good solutions, store results
            if np.any(np.isfinite(nb_phntp)):
                # If there are interesting phenotypes, store marker combination
                comb_dict[comb_idx] = comb

                # Create parameter grid matching x and y space
                x_values, y_values = np.meshgrid(x_samplesxbatch_space,
                                                 y_cellxsample_space,
                                                 indexing='ij')

                # Constrain grid
                x_values = np.where(mask, np.nan, x_values)
                y_values = np.where(mask, np.nan, y_values)

                # Calculate maximum number of phenotypes for marker combination 'comb'
                max_nb_phntp_comb = np.nanmax(nb_phntp)

                # Calculate minimum number of undefined cells for marker combination 'comb'
                min_nb_undefined = np.nanmin(nb_undef_cells)

                # Best solution has maximum number of new phenotypes...
                best_comb_idx = np.append(best_comb_idx, comb_idx)  # Index of comb to use in 'comb_dict'
                best_nb_phntp = np.append(best_nb_phntp, max_nb_phntp_comb)

                # ... and minimum number of undefined cells
                nb_undef_cells[nb_phntp != max_nb_phntp_comb] = np.nan
                best_nb_undefined = np.append(best_nb_undefined, min_nb_undefined)

                # ... and maximum percentage within batch
                x_values[nb_undef_cells != min_nb_undefined] = np.nan
                best_x_values = np.append(best_x_values, np.nanmax(x_values))

                # ... and maximum cells per sample
                y_values[x_values != np.nanmax(x_values)] = np.nan
                best_y_values = np.append(best_y_values, np.nanmax(y_values))

            comb_idx += 1

        marker_counter += 1

    # Further refine possible best marker combination(s) according to metrics we
    # chose previously: number of phenotypes, number of undefined cells, x and y values

    # Find combination(s) with maximum number of phenotypes
    max_phntp_idx = np.where(best_nb_phntp == np.max(best_nb_phntp))[0]

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

                # Even if there are still several combinations, we keep all of them
                best_marker_comb = list(comb_dict.get(key) for key in best_comb_idx[max_yvalues_idx])

            else:  # Only one combination with maximum x value
                best_marker_comb = list(comb_dict[best_comb_idx[max_xvalues_idx[0]]])

        else:  # Only one combination with minimum number of undefined cells
            best_marker_comb = list(comb_dict[best_comb_idx[min_undefined_idx[0]]])

    elif len(max_phntp_idx) == 1:  # Only one combination with maximum number of phenotypes
        best_marker_comb = list(comb_dict[best_comb_idx[max_phntp_idx[0]]])

    else:  # No combination satisfying number of phenotypes condition was found
        best_marker_comb = []

    return best_marker_comb
