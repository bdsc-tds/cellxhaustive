"""
AT. Add general description here.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports local functions
from cellxhaustive.select_cells import select_cells


# Permute across positive and negative expression of the relevant markers
# and identify new cell types
# AT. Check presence/absence of all parameters/variable
def cell_subdivision_counts(mat, mat_representative,
                            markers, markers_representative,
                            marker_order, batches, samples, three_markers=[],
                            p_min=np.array([0.1, 0.2]), s_min=np.array([5, 10])):
    """
    Cell line subdivision.
    # AT. Add function description (use the one before?)

    Parameters:
    -----------
    mat: ndarray
      2-D array expression matrix.

    mat_representative: ndarray
      2-D array expression matrix of the representative markers.

    markers: array
      1-D array with the markers in mat corresponding to each column.

    markers_representative: ndarray
      1-D array with the relevant markers in mat_representative corresponding
      to each column.

    marker_order: list(str)
      List of markers used in the gating strategy ordered accordingly.

    batches:
      # AT. Update parameter description.

    samples:
      # AT. Update parameter description.

    three_markers: list(str)
      List of markers with potentially three peaks.
      # AT. Add (default=[])

    p_min: list(int) (default=)
      Minimum proportion of samples for annotation to be considered.
      # AT. Check if int, list, or else and update/remove default

    s_min: list(int) (default=)
      # AT. Update parameter description.
      # AT. Check if int, list, or else and update/remove default

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    undefined = np.zeros((len(p_min), len(s_min)))
    results = np.zeros((len(p_min), len(s_min)))

    # Find different groups of positive cells
    three_markers_ = np.asarray(three_markers)[np.isin(three_markers, markers_representative)]
    three_markers_low = [x + "_low" for x in three_markers_]
    three_markers = list(three_markers_)
    markers_representative_ = list(markers_representative) + list(three_markers_low)
    groups = [tuple(np.unique(i)) for i in ite.combinations_with_replacement(
        markers_representative_, len(markers_representative_))]
    groups = [tuple([])] + groups

    # Loop over unique groups
    for i in set(groups):
        iteration = np.unique(i)
        if np.any(np.isin(three_markers_[np.isin(three_markers_low, iteration)], iteration)):
            continue
        else:
            # The groups are defined by positive markers
            positives = np.asarray(markers_representative)[np.isin(markers_representative, iteration)]

            # The negatives should be defined easily
            negatives = np.asarray(markers_representative)[np.isin(markers_representative, iteration) == False]

            # The low positives are those markers that appear in three_markers_low
            lowpositives = np.asarray(three_markers_)[np.isin(three_markers_low, iteration)]

        # Figure out which cells fulfill these rules
        # AT. select_cells heavily relies on ADTnorm DEFAULT values for pos and neg peaks --> Make a note in documentation
        cells = select_cells(mat_representative,
                             markers_representative,
                             positive=positives,
                             negative=negatives,
                             lowpositive=lowpositives,
                             three_markers=three_markers)

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = np.zeros(np.shape(results)) == 0
        for b in np.unique(batches):
            cells_ = cells[batches == b]
            samples_ = samples[batches == b]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_] == x) for x in np.unique(samples_)])
            keep_cell_type_ = keep_cell_type_[:, np.newaxis] > s_min
            keep_cell_type_ = (np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_))))[np.newaxis, :]
            keep_cell_type_ = keep_cell_type_ > p_min[:, np.newaxis]
            keep_cell_type = np.logical_and(keep_cell_type, keep_cell_type_)

        results += keep_cell_type * 1
        undefined += (keep_cell_type == False) * np.sum(cells)

    return results, undefined
