"""
AT. Add general description here.
"""


# Imports utility modules
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
                                x_samplesxbatch_space=np.round(np.arange(0.5, 1.01, 0.1), 1),
                                y_cellxsample_space=np.arange(10, 101, 10),
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




# AT. Create a default matrix filled with False/True and update it along the way?

    # Initialize matrices to store results
    nb_cell_types = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    nb_undefined_cells = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))

    # Process marker status combinations returned by previous function
    # 'determine_marker_status()' and check whether they are worth keeping
    for cell_type in np.unique(cell_types):
        # AT. Multithread/process here? Conflict between batches?

        ###### AT. Discuss with Bernat about the importance of 'cell_type' presence in all batches (np.logical_and vs np.logical_or)
        # Initialise temporary array to store 'cell_type' results
        keep_cell_type = np.full(np.shape(nb_cell_types), False)

        ##### AT. Former code
        # # Initialise temporary array to store 'cell_type' results
        # keep_cell_type = np.full(np.shape(nb_cell_types), True)

        # Process batches separately
        for batch in np.unique(batches_label):
            # AT. Multithread/process here?

            # Split cell type data according to batch
            cell_types_batch = cell_types[batches_label == batch]

            # Split sample data, first according to batch and then cell type
            cell_type_samples = samples_label[batches_label == batch][cell_types_batch == cell_type]  # Split samples data  # AT. Needed???

            # If there are no cells of type 'cell_type' in 'batch', skip
            if cell_type_samples.size == 0:
                continue

            # Calculate number of different samples in current batch and cell type
            samples_nb = float(len(np.unique(cell_type_samples)))

            # Count number of cells with 'cell_type' marker combination in each sample
            cell_count_sample = np.asarray([np.sum(cell_type_samples == smpl)
                                            for smpl in np.unique(cell_type_samples)])

            # Check whether cell counts satisfy y threshold
            keep_cell_type_batch = cell_count_sample[:, np.newaxis] >= y_cellxsample_space
            # Note: we use np.newaxis to add a dimension to work on different
            # samples concurrently

            # Calculate proportion of samples in current batch satisfying the
            # 'y_cellxsample_space' condition
            keep_cell_type_batch = (np.sum(keep_cell_type_batch, axis=0) / samples_nb)
            # Notes:
            # - keep_cell_type_batch is a boolean array, so we can calculate its sum
            # - np.sum(keep_cell_type_batch, axis=0) calculates the number of samples
            # satisfying y condition for a given y in the grid

            # Check whether sample proportions satisfy x threshold
            keep_cell_type_batch = keep_cell_type_batch >= x_samplesxbatch_space[:, np.newaxis]
            # Note: [:, np.newaxis] is used to transpose the 1-D array into a 2D
            # array to allow the comparison


A = np.zeros(np.shape(results)) == 0

A = np.logical_and(A, keep_cell_type)


cell_type_samples = samples_label[batches_label == batch][cell_types_batch == cell_type]  # Split samples data  # AT. Needed???
cell_type_samples_test = np.append(cell_type_samples, np.array('toto'))
samples_nb_test = float(len(np.unique(cell_type_samples_test)))
cell_count_sample_test = np.array([4, 1])
keep_cell_type_test = cell_count_sample_test[:, np.newaxis] >= y_cellxsample_space
keep_cell_type_test[0, 0:2] = True
keep_cell_type_test[1, 0] = True
# keep_cell_type_test = (np.sum(keep_cell_type_test, axis=0) / samples_nb_test)
keep_cell_type_test = (np.sum(keep_cell_type_test, axis=0) / samples_nb_test)[np.newaxis, :]
keep_cell_type_test = keep_cell_type_test >= x_samplesxbatch_space[:, np.newaxis]
A = np.zeros(np.shape(results)) == 0
A = np.logical_and(A, keep_cell_type_test)


np.logical_and(keep_cell_type_test)


A = np.logical_and(A, keep_cell_type_test)
            # array to allow comparison

            ###### AT. Discuss with Bernat about the importance of 'cell_type' presence in all batches (np.logical_and vs np.logical_or)
            # Add nummber of undefined cells to counter
            nb_undefined_cells += (keep_cell_type_batch == False) * np.sum(cell_count_sample)

            # Intersect batch results with general results
            keep_cell_type = np.logical_or(keep_cell_type, keep_cell_type_batch)

        # Add cell_type presence to cell type counter
        nb_cell_types += keep_cell_type * 1

    return nb_cell_types, nb_undefined_cells

        # AT. Loop over cell type then over batch? Or batch then cell type?
        # for cell_type, cell_type_count in ite.zip_longest(cell_types_batch_lst, cell_types_batch_count):

            # AT. 2nd way (new)
            # # Test if 'cell_type' passes the y_cellxsample_space threshold
            # y_test = (cell_count_sample[:, np.newaxis] >= y_cellxsample_space)

            # # Test if 'cell_type' passes the y_cellxsample_space threshold
            # x_test = (PLACEHOLDER >= y_cellxsample_space)


            # Test if 'cell_type' passes the x_samplesxbatch_space threshold
            # float(len(np.unique(cell_type_samples))): number of different samples in 'batch'
            np.sum(keep_cell_type, axis=0) / float(len(np.unique(cell_type_samples)))
            (np.sum(keep_cell_type, axis=0) / float(len(np.unique(cell_type_samples))))[np.newaxis, :]

            # Compare results of both tests
            np.logical_and(x_test, y_test)

            # Test if 'cell_type' passes the y_cellxsample_space threshold
            if np.any(cell_type_count >= y_cellxsample_space):
                # If it does, test if 'cell_type' passes the x_samplesxbatch_space threshold
                if:
            else:
            # If it doesn't, change status to False and skip to next 'cell_type'



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
