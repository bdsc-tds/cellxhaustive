"""
AT. Add general description here.
"""


# Imports utility modules
import copy
import itertools as ite
import numpy as np

# Imports local functions
from cell_subdivision_counts import cell_subdivision_counts
# from cellxhaustive.cell_subdivision_counts import cell_subdivision_counts  # AT. Double-check path


# AT. Add description
# AT. Check presence/absence of all parameters/variable
def check_all_subsets(mat_representative, batches_label, samples_label,
                      markers_representative, max_markers=15, min_annotations=3,
                      min_samplesxbatch=0.5, min_cellxsample=10):
    """
    # AT. Add function description

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
      Minimum number of markers used to define a cell population. Must be in
      [2; len(markers)], but it is advised to choose a value in [3; len(markers) - 1].
      # AT. Double check description



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
      # AT. Add what is returned by the function
    """

    # Create total grid of cellxsample and min_samplesxbatch axes on which
    # to compute metrics (number of cell types and number of undefined cells)

    # Create total space for each axis
    x_samplesxbatch_space = np.arange(min_samplesxbatch, 1.01, 0.01)  # x-axis
    y_cellxsample_space = np.arange(min_cellxsample, 101)  # y-axis

    # Create the grid
    XX, YY = np.meshgrid(x_samplesxbatch_space, y_cellxsample_space, indexing='ij')  # AT. Double-check what's the better indexing

    # # AT. Remove when finished testing
    # x_samplesxbatch_space = np.linspace(0, 1, 101)  # x-axis  # AT. Double-check
    # y_cellxsample_space = np.arange(101)  # y-axis  # AT. Double-check

    # XX = np.transpose(np.asarray([list(x_samplesxbatch_space) for ljh in range(len(y_cellxsample_space))])).astype(float)  # (101,101) matrix
    # YY = np.asarray([list(y_cellxsample_space) for ljh in range(len(x_samplesxbatch_space))]).astype(float)  # (101,101) matrix

    # # Slice the grid variables
    # XX = XX[:, y_cellxsample_space >= min_cellxsample]
    # XX = XX[x_samplesxbatch_space >= min_samplesxbatch, :]
    # YY = YY[:, y_cellxsample_space >= min_cellxsample]
    # YY = YY[x_samplesxbatch_space >= min_samplesxbatch, :]

    # Find theoretical maximum number of markers in the combination
    max_combination = min(max_markers, len(markers_representative))

    # Initialize counters and objects to store results
    # Note that by default, we assume that the minimum number of relevant
    # markers is 2 (i.e. only 1 markers can not define a phenotype)
    marker_counter = 2
    # marker_counter = min_annotations  # AT. Check this with Bernat
    index = 0  # AT. Double-check if interesting. If kept, rename it
    mset = dict()  # AT. Double-check if interesting. If kept, rename it

    # 'max_res' is a way to avoid going through all permutations, as if the
    # grouping is not good, further permutations will not be explored
    max_res = 0
    max_res_tmp = 0

    # Empty vectors to store results
    best_indices = []
    best_undefined = []
    best_cells = []
    best_matx = []
    best_maty = []

    # Go through all the combinations until no better solution can be found
    while True:

        # Stop the while loop if the maximum number of markers is reached or if
        # the possible solution using more markers is worse than the current
        # best. If the loop isn't stopped here, that means the scores can still
        # be improved.
        if (marker_counter > max_combination) or (max_res > max_res_tmp):
            break
        else:
            max_res = max_res_tmp

        # For a given number of markers, check all possible marker combinations
        for comb in ite.combinations(markers_representative, marker_counter):
            # AT. Opportunity to multiprocess?

            # Slice data based on the current marker combination 'comb'
            # markers_comb = markers[np.isin(markers, np.asarray(comb))]  # AT. Double-check
            markers_comb = markers_representative[np.isin(markers_representative, np.asarray(comb))]
            # mat_comb = mat_representative[:, np.isin(markers, markers_comb)]  # AT. Double-check
            mat_comb = mat_representative[:, np.isin(markers_representative, markers_comb)]

            # This variable is probably unnecessary but it's not a big deal to keep it
            mset[index] = comb

            # Find number of cell types and undefined cells for a given marker
            # combination 'comb' across cellxsample and min_samplesxbatch grid




            # AT. Bernat said it would be better to have cell_subdivision_counts return one number in results and one in undefined instead of matrices
            results, undefined = cell_subdivision_counts(
                mat_comb=mat_comb,
                batches_label=batches_label,
                samples_label=samples_label,
                markers_comb=markers_comb,
                x_samplesxbatch_space=x_samplesxbatch_space,  # AT. Might have to change parameter and variable names if I adapt function to take grid as input
                y_cellxsample_space=y_cellxsample_space,  # AT. Might have to change parameter and variable names if I adapt function to take grid as input
                three_markers=["CD4"])

            # AT. Need to change this function to take XX and YY as arguments, i.e. apply the function directly on the whole grid






            # AT. Slicing probably useless now that the constraints are included im the base grid?
            # Slice N phenotypes matrix given the conditions min_samplesxbatch and min_cellxsample and
            # calculate the maximum number of phenotypes for the marker combination g
            results = results[:, y_cellxsample_space >= min_cellxsample]
            results = results[x_samplesxbatch_space >= min_samplesxbatch, :]
            max_res_tmp = np.max(results)

            # AT. Slicing probably useless now that the constraints are included im the base grid?
            # Slice the undefined matrix as above
            undefined = undefined[:, y_cellxsample_space >= min_cellxsample]
            undefined = undefined[x_samplesxbatch_space >= min_samplesxbatch, :]

            # Deep copy of the grid variables (potential for efficientcy)
            # AT. Useless?
            matx = copy.deepcopy(XX)
            maty = copy.deepcopy(YY)

            # Further constrain matrix given the minimum number of phenotype conditions
            condi = results < min_annotations
            results[condi] = np.nan
            undefined[condi] = np.nan
            matx[condi] = np.nan
            maty[condi] = np.nan

            # If there are possible good solutions, store the results
            if np.any(np.isnan(results) == False):
                # The best solution will have the maximum number of new phenotypes...
                best_indices += [index]
                best_cells += [np.max(results[np.isnan(results) == False])]
                # ...and the minimum number of undefined cells...
                undefined[results != np.max(results[np.isnan(results) == False])] = np.nan
                best_undefined += [np.min(undefined[np.isnan(undefined) == False])]
                # and the maximum percentage within batch
                matx[undefined != np.min(undefined[np.isnan(undefined) == False])] = np.nan
                best_matx += [np.max(matx[np.isnan(matx) == False])]
                # and the maximum cells per sample
                maty[matx != np.max(matx[np.isnan(matx) == False])] = np.nan
                best_maty += [np.max(maty[np.isnan(maty) == False])]

            index += 1

        marker_counter += 1

    best_cells = np.asarray(best_cells)
    best_undefined = np.asarray(best_undefined)
    best_indices = np.asarray(best_indices)
    best_matx = np.asarray(best_matx)
    best_maty = np.asarray(best_maty)

    if len(best_cells) > 0:
        # Filter based on total number of new phenotypes
        best_indices = best_indices[best_cells == np.max(best_cells)]
        best_matx = best_matx[best_cells == np.max(best_cells)]
        best_maty = best_maty[best_cells == np.max(best_cells)]
        best_undefined = best_undefined[best_cells == np.max(best_cells)]
        best_cells = best_cells[best_cells == np.max(best_cells)]

    # AT. Add if/else block description
    if len(best_indices) > 1:
        i = np.where(best_undefined == np.min(best_undefined))[0]

        if len(i) > 1:
            best_indices = best_indices[i]
            best_matx = best_matx[i]
            j = np.where(best_matx == np.max(best_matx))[0]
            return list(mset[best_indices[j[0]]])
        else:
            return list(mset[best_indices[i[0]]])
    elif len(best_indices) == 0:
        return []
    else:
        return list(mset[best_indices[0]])
