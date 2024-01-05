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
                                three_peak_markers=[],
                                x_samplesxbatch_space=np.round(np.arange(0.5, 1.01, 0.1), 1),
                                y_cellxsample_space=np.arange(10, 101, 10)):
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

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    x_samplesxbatch_space: array(float) (default=[0.5, 0.6, 0.7, ..., 1])
      Minimum proportion of samples within each batch with at least
      'y_cellxsample_space' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10, 20...
      100 cells/sample (see description of next parameter) in at least 50%, 60%...
      100% of samples within a batch to be considered.

    y_cellxsample_space: array(float) (default=[10, 20, 30, ..., 100])
      Minimum number of cells within each sample in 'x_samplesxbatch_space' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10, 20...
      100 cells/sample in at least 50%, 60%... 100% of samples (see description
      of previous parameter) within a batch to be considered.

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Determine markers status of 'markers_comb' using expression data
    phenotypes = determine_marker_status(mat_comb=mat_comb,
                                         markers_comb=markers_comb,
                                         three_peak_markers=three_peak_markers,
                                         two_peak_threshold=3,
                                         three_peak_lower_threshold=2,
                                         three_peak_upper_threshold=4)
    # AT. Could return an array of 'cell' rows * 'markers_comb' columns if it's easier
    # AT. Or an array of array instead of an array of lists?

    # Initialize arrays to store results
    nb_phenotypes = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    nb_undefined_cells = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))

    # Process marker status combinations returned by 'determine_marker_status()'
    # and check whether they are worth keeping
    for phenotype in np.unique(phenotypes):
        # AT. Multithread/process here? Conflict between batches?

        ###### AT. Discuss with Bernat about the importance of 'phenotype' presence in all batches (np.logical_and vs np.logical_or)
        # Initialise temporary array to store 'phenotype' results
        keep_phenotype = np.full(np.shape(nb_phenotypes), False)

        ##### AT. Former code
        # # Initialise temporary array to store 'phenotype' results
        # keep_phenotype = np.full(np.shape(nb_phenotypes), True)

        # Process batches separately
        for batch in np.unique(batches_label):
            # AT. Multithread/process here?

            # Split cell type data according to batch
            phenotypes_batch = phenotypes[batches_label == batch]

            # Split sample data, first according to batch and then cell type
            phenotype_samples = samples_label[batches_label == batch][phenotypes_batch == phenotype]

            # If there are no cells of type 'phenotype' in 'batch', skip
            if phenotype_samples.size == 0:
                continue

            # Calculate number of different samples in current batch and cell type
            samples_nb = float(len(np.unique(phenotype_samples)))

            # Count number of cells with 'phenotype' marker combination in each sample
            cell_count_sample = np.asarray([np.sum(phenotype_samples == smpl)
                                            for smpl in np.unique(phenotype_samples)])

            # Check whether cell counts satisfy y threshold
            keep_phenotype_batch = cell_count_sample[:, np.newaxis] >= y_cellxsample_space
            # Note: we use np.newaxis to add a dimension to work on different
            # samples concurrently

            # Calculate proportion of samples in current batch satisfying
            # 'y_cellxsample_space' condition
            keep_phenotype_batch = (np.sum(keep_phenotype_batch, axis=0) / samples_nb)
            # Notes:
            # - keep_phenotype_batch is a boolean array, so we can calculate its sum
            # - np.sum(keep_phenotype_batch, axis=0) calculates the number of samples
            # satisfying y condition for a given y in the grid

            # Check whether sample proportions satisfy x threshold
            keep_phenotype_batch = keep_phenotype_batch >= x_samplesxbatch_space[:, np.newaxis]
            # Note: [:, np.newaxis] is used to transpose the 1-D array into a 2D
            # array to allow comparison

            ###### AT. Discuss with Bernat about the importance of 'phenotype' presence in all batches (np.logical_and vs np.logical_or)
            # Add number of undefined cells to counter
            nb_undefined_cells += (keep_phenotype_batch == False) * np.sum(cell_count_sample)

            # Intersect batch results with general results
            keep_phenotype = np.logical_or(keep_phenotype, keep_phenotype_batch)

        # Add 'phenotype' presence/absence to cell type counter
        nb_phenotypes += keep_phenotype * 1

    return nb_phenotypes, nb_undefined_cells

    #         ###### AT. Former code
    #         # Intersect batch results with general results
    #         keep_phenotype = np.logical_and(keep_phenotype, keep_phenotype_batch)

    #     nb_phenotypes += keep_phenotype * 1
    #     nb_undefined_cells += (keep_phenotype == False) * np.sum(cell_count_sample)  # AT. Rework this to count 'phenotype' cells in all batches

    # return nb_phenotypes, nb_undefined_cells








"""
GOAL: RETURN 2 MATRICES
- 1 WITH THE CELL TYPE RESULTS ACROSS THE 2 PARAMETER VARIATIONS
- 1 WITH THE NUMBER OF UNDEFINED CELLS (OR % OF UNDEFINED CELLS) ACROSS THE 2 PARAMETER VARIATIONS
"""


# OR RETUN BEST CELL TYPE FOR ALL THE MATRIX ?
