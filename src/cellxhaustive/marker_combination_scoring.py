"""
AT. Add general description here.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports local functions
from determine_marker_status import determine_marker_status  # AT. Double-check path
# from cellxhaustive.determine_marker_status import cellxhaustive.determine_marker_status


# Permute across positive and negative expression of the relevant markers
# and identify new cell types
# AT. Check presence/absence of all parameters/variable
# Function used in
# AT. Improve description
def marker_combinations_scoring(mat_comb, markers_comb,
                                batches_label, samples_label,
                                x_samplesxbatch_space, y_cellxsample_space,
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

    markers_comb: array(str)
      1-D numpy array with markers matching each column of 'mat_comb'.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_comb'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_comb'.

    x_samplesxbatch_space: array(float)
      Minimum proportion of samples within each batch with at least
      'y_cellxsample_space' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 5 or 10
      cells/sample (see description of previous parameter) in at least 10 or 20%
      of the samples within a batch to be considered.
      # AT. Update description and default if I adapt function to take grid as input

    y_cellxsample_space: array(float)
      Minimum number of cells within each sample in 'x_samplesxbatch_space' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      5 or 10 cells/sample in at least 10 or 20% of the samples (see
      description of next parameter) within a batch to be considered.
      # AT. Update description and default if I adapt function to take grid as input

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Determine markers status of 'markers_comb' using expression data
    cell_types = determine_marker_status(mat_comb=mat_comb,
                                         markers_comb=markers_comb,
                                         three_peak_markers=three_peak_markers,
                                         two_peak_threshold=3,
                                         three_peak_lower_threshold=2,
                                         three_peak_upper_threshold=4)
    # AT. Could return an array of 'cell' rows * 'markers_comb' columns if it's easier
    # AT. Or an array of array instead of an array of lists?



    # Count each cell type
    # all_cell_type, cell_type_count = np.unique(cell_type, return_counts=True)  # AT. Use this after?

    # Initialize matrices to store results
    undefined = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    results = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))







# AT. Do we actually need the minimums ?

    # Extract minimum thresholds for x and y
    # min_x = np.min(x_samplesxbatch_space)
    # min_y = np.min(y_cellxsample_space)

    # Keep only cells passing both minimum thresholds
    keep_cell_type = np.logical_and(PLACEHOLDER > min_samplesxbatch, cell_type_count > min_cellxsample)

    all_cell_type[cell_type_count > min_y]


    # x_samplesxbatch_space = np.arange(min_samplesxbatch, 1.01, 0.01)  # x-axis
    # y_cellxsample_space = np.arange(min_cellxsample, 101)  # y-axis

    # # Create the grid
    # XX, YY = np.meshgrid(x_samplesxbatch_space, y_cellxsample_space, indexing='ij')  # AT. Double-check what's the better indexing



"""
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

      For y:
        - Condition of number of cells / sample (10/sample)

        - Condition of number of samples / batch (50%)


        mat_comb, batches_label, samples_label,
"""

for batch in np.unique(batches_label):
    mat_comb_batch = mat_comb[batches_label == batch]  # Split expression data according to batch
    mat_comb_samples = samples_label[batches_label == batch]  # Split samples data according to batch
    cell_type_batch = cell_type[batches_label == batch]  # Split cell type data according to batch

    # Count each cell type
    cell_type_batch_all, cell_type_count = np.unique(cell_type_batch, return_counts=True)

#
for type_batch, type_count in ite.zip_longest(cell_type_batch_all, cell_type_count):








# Don't forget to increase undefined



# 1. Split by sample
# 2. Compare split matrix to number of cells (which varies --> XX here?)

# Starts with 0, adds +1 if iteration on marker sub-combination satisfies conditions

        # AT. undefined is what doesn't pass the 2 thresholds


        # AT. Cell type attribution happens here
        # AT. x_samplesxbatch_space and y_cellxsample_space are also impacting here

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = (np.zeros(np.shape(results)) == 0)

        # for batch in np.unique(batches_label):
            # cells_ = cells[batches_label == batch]
            # samples_ = samples_label[batches_label == batch]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_] == x) for x in np.unique(samples_)])
            keep_cell_type_ = keep_cell_type_[:, np.newaxis] > y_cellxsample_space
            keep_cell_type_ = (np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_))))[np.newaxis, :]
            keep_cell_type_ = keep_cell_type_ > x_samplesxbatch_space[:, np.newaxis]
            keep_cell_type = np.logical_and(keep_cell_type, keep_cell_type_)

        results += keep_cell_type * 1
        undefined += (keep_cell_type == False) * np.sum(cells)

    return results, undefined


"""
GOAL: RETURN 2 MATRICES
- 1 WITH THE CELL TYPE RESULTS ACROSS THE 2 PARAMETER VARIATIONS
- 1 WITH THE NUMBER OF UNDEFINED CELLS (OR % OF UNDEFINED CELLS) ACROSS THE 2 PARAMETER VARIATIONS
"""



# OR RETUN BEST CELL TYPE FOR ALL THE MATRIX ?
