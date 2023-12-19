"""
AT. Add general description here.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports local functions
from select_cells import select_cells  # AT. Double-check path
# from cellxhaustive.select_cells import select_cells


# Permute across positive and negative expression of the relevant markers
# and identify new cell types
# AT. Check presence/absence of all parameters/variable
def cell_subdivision_counts(mat_representative,
                            markers_representative,
                            batches, samples,
                            percent_samplesxbatch=np.array([0.1, 0.2]),
                            min_cellxsample=np.array([5, 10]),
                            three_markers=[]):
    """
    Cell line subdivision.
    # AT. Add function description (use the one before?)

    Parameters:
    -----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    mat_representative: array(float)
      2-D numpy array expression matrix of the representative markers, with
      cells in D0 and markers in D1. In other words, rows contain cells and
      columns contain markers.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of mat_representative.




    batches: array(str)
      1-D numpy array with batch names of each cell of mat. Useful for defining
      the thresholds for the new annotations.

    samples: array(str)
      1-D numpy array with sample names of each cell of mat. Useful for defining
      the thresholds for the new annotations.

    min_cellxsample: array(float) (default=[5, 10])
      Minimum number of cells within each sample in percent_samplesxbatch % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      5 or 10 cells/sample in at least 10 or 20% of the samples (see
      description of next parameter) within a batch to be considered.

    percent_samplesxbatch: array(float) (default=[0.1, 0.2])
      Minimum proportion of samples within each batch with at least
      min_cellxsample cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 5 or 10
      cells/sample (see description of previous parameter) in at least 10 or 20%
      of the samples within a batch to be considered.



    three_markers: list(str) (default=[])
      List of markers with potentially three peaks.
      # AT. Improve description? Do we even keep it? Or set it as an option?

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    undefined = np.zeros((len(percent_samplesxbatch), len(min_cellxsample)))
    results = np.zeros((len(percent_samplesxbatch), len(min_cellxsample)))

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
            keep_cell_type_ = keep_cell_type_[:, np.newaxis] > min_cellxsample
            keep_cell_type_ = (np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_))))[np.newaxis, :]
            keep_cell_type_ = keep_cell_type_ > percent_samplesxbatch[:, np.newaxis]
            keep_cell_type = np.logical_and(keep_cell_type, keep_cell_type_)

        results += keep_cell_type * 1
        undefined += (keep_cell_type == False) * np.sum(cells)

    return results, undefined
