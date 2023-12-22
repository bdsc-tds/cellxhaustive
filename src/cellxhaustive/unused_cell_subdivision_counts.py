"""
AT. Add general description here.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports local functions
from cellxhaustive.determine_marker_status import determine_marker_status  # AT. Double-check path
from select_cells import select_cells  # AT. Double-check path
# from determine_marker_status import determine_marker_status
# from cellxhaustive.select_cells import select_cells


# Permute across positive and negative expression of the relevant markers
# and identify new cell types
# AT. Check presence/absence of all parameters/variable
def cell_subdivision_counts(mat_comb, batches_label, samples_label, markers_comb,
                            x_samplesxbatch_space=np.array([0.1, 0.2]),  # AT. Might have to change parameter and variable names if I adapt function to take grid as input
                            y_cellxsample_space=np.array([5, 10]),  # AT. Might have to change parameter and variable names if I adapt function to take grid as input
                            three_peak_markers=[]):
    """
    Cell line subdivision.
    # AT. Add function description (use the one before?)

    Parameters:
    -----------
    mat_comb: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and a specific combination of
      representative markers.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_comb'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_comb'.

    markers_comb: array(str)
      1-D numpy array with markers matching each column of 'mat_comb'.



    x_samplesxbatch_space: array(float) (default=[0.1, 0.2])
      Minimum proportion of samples within each batch with at least
      'y_cellxsample_space' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 5 or 10
      cells/sample (see description of previous parameter) in at least 10 or 20%
      of the samples within a batch to be considered.
      # AT. Update description and default if I adapt function to take grid as input

    y_cellxsample_space: array(float) (default=[5, 10])
      Minimum number of cells within each sample in 'x_samplesxbatch_space' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      5 or 10 cells/sample in at least 10 or 20% of the samples (see
      description of next parameter) within a batch to be considered.
      # AT. Update description and default if I adapt function to take grid as input



    three_peak_markers: list(str) (default=[])
      List of markers with potentially three peaks.
      # AT. Improve description? Do we even keep it? Or set it as an option?

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Initialize matrices to store results
    undefined = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    results = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))

    # Split positive peaks of markers with 3 peaks into normal and '_low'
    three_markers_comb = np.asarray(three_peak_markers)[np.isin(three_peak_markers, markers_comb)]
    three_markers_low = np.char.add(three_markers_comb, '_low')
    markers_comb_with3 = np.concatenate((markers_comb, three_markers_low))

    # Find different combinations of positive cells
    pos_cells = [tuple(np.unique(group))
                 for group in ite.combinations_with_replacement(markers_comb_with3, len(markers_comb_with3))]
    pos_cells = set([tuple()] + pos_cells)  # tuple(): case with only negative markers


# AT. Rework this to have both positive and negative here ?

# AT. Why do we skip iteration containing markers with a low positive peak??

"""
I think that I am generating the different groupings with itertools because it’s
faster. When you have three peaks, however, I consider marker A and marker A_low
as two independent markers (one reflecting the low peak and the other the high one).
Now, the itertools function will give me groupings like (A, D) or (A_low, B). The
first example reflects a phenotype with A and D as positive peaks, while the second
example represents a phenotype with A middle peak and B positive peak. The problem
with the itertools function that I have is that it might also give me a grouping
like (A, A_low, B). This does not make sense because A can’t be positive AND middle
peak. So I am trying to correct for that. The truth is that there is probably a better
way to generate the grouping in the first place to avoid that, but I couldn’t find a
good solution using itertools.
"""

# AT. How are identified negatively expressed markers???
# AT. Negative marker = not in pos or low_pos ??

    # Loop over unique groups
    for pos_comb in pos_cells:
        if np.any(np.isin(three_markers_comb[np.isin(three_markers_low, pos_comb)], pos_comb)):
            # AT. np.isin(three_markers_low, pos_comb): True if pos_comb contains at least one marker_low
            continue
        else:
            # The groups are defined by positive markers
            positives = markers_comb[np.isin(markers_comb, pos_comb)]

            # The negatives should be defined easily
            negatives = markers_comb[np.isin(markers_comb, pos_comb, invert=True)]

            # The low positives are those markers that appear in three_markers_low
            lowpositives = three_markers_comb[np.isin(three_markers_low, pos_comb)]

        # Figure out which cells fulfill these rules
        # AT. select_cells heavily relies on ADTnorm DEFAULT values for pos and neg peaks --> Make a note in documentation
        # AT. Determine which cells have 'pos_comb' combination of positive and negative markers
        cells = select_cells(mat_comb=mat_comb,
                             markers_comb=markers_comb,
                             positives=positives,
                             negatives=negatives,
                             lowpositives=lowpositives,
                             three_markers_comb=three_markers_comb)  # AT. Double-check if directly inputing an array works

        # AT. Cell type attribution happens here
        # AT. x_samplesxbatch_space and y_cellxsample_space are also impacting here

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = np.zeros(np.shape(results)) == 0
        for b in np.unique(batches_label):
            cells_ = cells[batches_label == b]
            samples_ = samples_label[batches_label == b]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_] == x) for x in np.unique(samples_)])
            keep_cell_type_ = keep_cell_type_[:, np.newaxis] > y_cellxsample_space
            keep_cell_type_ = (np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_))))[np.newaxis, :]
            keep_cell_type_ = keep_cell_type_ > x_samplesxbatch_space[:, np.newaxis]
            keep_cell_type = np.logical_and(keep_cell_type, keep_cell_type_)

        results += keep_cell_type * 1
        undefined += (keep_cell_type == False) * np.sum(cells)

    return results, undefined

# Change to return only cells that are fulfilling the conditions??





        # AT. Could return an array of 'cell' rows * markers_comb columns if it's easier
        # AT. What is undefined??? Everything negative?
        # np.unique(cell_type, return_counts=True)


        # AT. Cell type attribution happens here
        # AT. x_samplesxbatch_space and y_cellxsample_space are also impacting here

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = np.zeros(np.shape(results)) == 0
        for b in np.unique(batches_label):
            cells_ = cells[batches_label == b]
            samples_ = samples_label[batches_label == b]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_] == x) for x in np.unique(samples_)])
            keep_cell_type_ = keep_cell_type_[:, np.newaxis] > y_cellxsample_space
            keep_cell_type_ = (np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_))))[np.newaxis, :]
            keep_cell_type_ = keep_cell_type_ > x_samplesxbatch_space[:, np.newaxis]
            keep_cell_type = np.logical_and(keep_cell_type, keep_cell_type_)

        results += keep_cell_type * 1
        undefined += (keep_cell_type == False) * np.sum(cells)

    return results, undefined
