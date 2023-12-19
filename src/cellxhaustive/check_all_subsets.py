"""
AT. Add general description here.
"""


# Imports utility modules
import copy
import itertools as ite
import numpy as np

# Imports local functions
from cell_subdivision_counts import cell_subdivision_counts  # AT. Double-check path
# from cellxhaustive.cell_subdivision_counts import cell_subdivision_counts


# AT. Add description
# AT. Check presence/absence of all parameters/variable
def check_all_subsets(max_markers, x_p, y_ns,
                      mat_subset,
                      markers, markers_representative,
                      batches, samples, min_cellxsample, percent_samplesxbatch,
                      min_cells):
    """
    # AT. Add function description

    Parameters:
    -----------
    max_markers:
      # AT. Add parameter description (and default?)

    x_p:
      # AT. Add parameter description (and default?)

    y_ns:
      # AT. Add parameter description (and default?)



    mat_subset: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains data only
      on the required cell label and batch.



    markers:
      # AT. Add parameter description (and default?)

    markers_representative:
      # AT. Add parameter description (and default?)

    batches:
      # AT. Add parameter description (and default?)

    samples:
      # AT. Add parameter description (and default?)



    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in percent_samplesxbatch % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      10 cells/sample in at least 50% of the samples (see description of next
      parameter) within a batch to be considered.

    percent_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      min_cellxsample cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of previous parameter) in at least 50% of
      the samples within a batch to be considered.



    min_cells:
      # AT. Add parameter description (and default?)

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Find which should be the maximum number of markers
    max_markers = np.min([max_markers, len(markers_representative)])

    # Start indices. Notice that here we assume that the minimum number of
    # relevant markers is 2 (i.e. one marker alone can not define phenotypes)
    counter = 2
    index = 0
    mset = dict()

    # Resmax variable is a way to avoid going through all permutations, as if
    # the grouping is not good, we shouldn't explore further permutations
    resmax = 0
    resmax_ = 0

    # Empty vectors to store results
    best_indices = []
    best_undefined = []
    best_cells = []
    best_matx = []
    best_maty = []
    maty_ = np.asarray([list(y_ns) for ljh in range(len(x_p))]).astype(float)
    matx_ = np.transpose(np.asarray([list(x_p) for ljh in range(len(y_ns))])).astype(float)

    # Slice the grid variables
    matx_ = matx_[:, y_ns >= min_cellxsample]
    matx_ = matx_[x_p >= percent_samplesxbatch, :]
    maty_ = maty_[:, y_ns >= min_cellxsample]
    maty_ = maty_[x_p >= percent_samplesxbatch, :]

    # We keep permuting until no better solution can be found
    while True:

        # Cut the while loop if we reached the maximum number of clusters or if
        # the possible solution using more markers will necessarily be worse
        # than the current best
        if counter > max_markers or resmax > resmax_:
            break
        else:
            resmax = resmax_

        # For a given number of markers "counter", look at all possible marker combinations
        for g in ite.combinations(markers_representative, counter):

            # Slice data based on the current marker combination g
            markers_subset = np.asarray(list(g))
            mat_subset = mat_[:, np.isin(markers, markers_subset)]
            markers_subset = markers[np.isin(markers, markers_subset)]

            # This variable is probably unnecessary but it's not a big deal to keep it
            mset[index] = g

            # Find number of cells and undefined cells for a given marker
            # combination across the p and ns grid
            results, undefined = cell_subdivision_counts(
                mat=mat_,
                mat_representative=mat_subset,
                markers=markers,
                markers_representative=markers_subset,
                batches=batches,
                samples=samples,
                marker_order=list(marker_order),
                three_marker=["CD4"],
                p_min=x_p,
                s_min=y_ns)

            # Deep copy of the grid variables (potential for efficientcy)
            matx = copy.deepcopy(matx_)
            maty = copy.deepcopy(maty_)

            # Slice N phenotypes matrix given the conditions percent_samplesxbatch and min_cellxsample and
            # calculate the maximum number of phenotypes for the marker combination g
            results = results[:, y_ns >= min_cellxsample]
            results = results[x_p >= percent_samplesxbatch, :]
            resmax_ = np.max(results)

            # Slice the undefined matrix as above
            undefined = undefined[:, y_ns >= min_cellxsample]
            undefined = undefined[x_p >= percent_samplesxbatch, :]

            # Further constrain matrix given the minimum number of phenotype conditions
            condi = results < min_cells
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

        counter += 1

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
