"""
Function that determines number of unique cell phenotypes (combination of positive
and negative markers) and number of cells without phenotype in an expression matrix
across different parameters thresholds.
"""


# Import utility modules
import numpy as np


# Import local functions
from determine_marker_status import determine_marker_status  # AT. Double-check path
# from cellxhaustive.determine_marker_status import cellxhaustive.determine_marker_status


# Function used in check_all_subsets.py  # AT. Update script name if needed
def score_marker_combinations(mat_comb, batches_label, samples_label,
                              markers_comb,
                              three_peak_markers=[],
                              x_samplesxbatch_space=np.round(np.arange(0.5, 1.01, 0.1), 1),
                              y_cellxsample_space=np.arange(10, 101, 10)):
    """
    Function that determines number of unique cell phenotypes (combination of
    positive and negative markers) and number of cells without phenotype in an
    expression matrix across different parameters thresholds.

    Parameters:
    -----------
    mat_comb: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_comb'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_comb'.

    markers_comb: array(str)
      1-D numpy array with markers matching each column of 'mat_comb'.

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
    nb_phntp: array(float)
      2-D numpy array showing number of unique cell phenotypes (combinations of
      positive and negative markers from 'markers_comb') identified in 'mat_comb'
      across grid composed of parameters 'x_samplesxbatch_space' in D0 and
      'y_cellxsample_space' in D1.

    phntp_to_keep: array(list(str))
      2-D numpy array showing lists of significant phenotypes across grid composed
      of parameters 'x_samplesxbatch_space' in D0 and 'y_cellxsample_space' in D1.
      In each element, len(phntp_to_keep[i, j]) = nb_phntp[i, j].

    nb_undef_cells: array(float)
      2-D numpy array showing number of undefined cells (cells without a phenotype)
      in 'mat_comb' across grid composed of parameters 'x_samplesxbatch_space' in
      D0 and 'y_cellxsample_space' in D1.

    phntp_per_cell: array(str)
      1-D numpy array showing best phenotype determined for each cell using markers
      from 'markers_comb'.
    """

    # Determine markers status of 'markers_comb' using expression data
    phntp_per_cell = determine_marker_status(
        mat_comb=mat_comb,
        markers_comb=markers_comb,
        three_peak_markers=three_peak_markers,
        two_peak_threshold=3,
        three_peak_lower_threshold=2,
        three_peak_upper_threshold=4)

    # Initialise arrays to store results
    nb_phntp = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    nb_undef_cells = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    phntp_to_keep = np.empty(len(x_samplesxbatch_space) * len(y_cellxsample_space),
                             dtype=object)
    phntp_to_keep[...] = [[] for _ in range(len(phntp_to_keep))]
    phntp_to_keep = np.reshape(phntp_to_keep, (len(x_samplesxbatch_space),
                                               len(y_cellxsample_space)))
    # Note: using 'dtype=list' fills array with None, hence this trick

    # Wrapper function of 'list.append()'
    def append_wrapper(lst, elt):
        return lst.append(elt)

    # Vectorized version of append wrapper
    array_appending = np.vectorize(append_wrapper, otypes=[str])

    # Process marker phenotypes returned by 'determine_marker_status()' and
    # check whether they are worth keeping
    for phenotype in np.unique(phntp_per_cell):
        # AT. Multithread/process here? Conflict between batches?

        # Initialise temporary array to store 'phenotype' results
        keep_phenotype = np.full(nb_phntp.shape, True)

        # Process batches separately
        for batch in np.unique(batches_label):
            # AT. Multithread/process here?

            # Split cell type data according to batch
            phenotypes_batch = phntp_per_cell[batches_label == batch]

            # Split sample data, first according to batch and then cell type
            phenotype_samples = samples_label[batches_label == batch][phenotypes_batch == phenotype]

            # If there are no cells of type 'phenotype' in 'batch', that means
            # 'phenotype' will not be present in all batches, so stop now
            if phenotype_samples.size == 0:
                keep_phenotype = np.logical_and(keep_phenotype, False)
                break

            # Calculate number of different samples in current batch and cell type
            samples_nb = float(len(np.unique(phenotype_samples)))

            # Count number of cells per phenotype in each sample
            cell_count_sample = np.asarray([np.sum(phenotype_samples == smpl)
                                            for smpl in np.unique(phenotype_samples)])

            # Check whether cell counts satisfy y threshold
            keep_phenotype_batch = cell_count_sample[:, np.newaxis] >= y_cellxsample_space
            # Note: np.newaxis is used to add a dimension to work on different
            # samples concurrently

            # Calculate proportion of samples in current batch satisfying
            # 'y_cellxsample_space' condition
            keep_phenotype_batch = (np.sum(keep_phenotype_batch, axis=0) / samples_nb)
            # Notes:
            # - 'keep_phenotype_batch' is a boolean array, so it can be summed
            # - 'np.sum(keep_phenotype_batch, axis=0)' calculates number of samples
            # satisfying y condition for a given y in grid

            # Check whether sample proportions satisfy x threshold
            keep_phenotype_batch = keep_phenotype_batch >= x_samplesxbatch_space[:, np.newaxis]
            # Note: '[:, np.newaxis]' is used to transpose 1-D array into a 2-D
            # array to allow comparison

            # Intersect batch results with general results
            keep_phenotype = np.logical_and(keep_phenotype, keep_phenotype_batch)
            # Note: for consistency, phenotypes have to be present in all batches,
            # hence usage of 'np.logical_and()'

        # Add 'phenotype' presence/absence to cell type counter
        nb_phntp += keep_phenotype * 1

        # Add number of undefined cells to counter
        nb_undef_cells += np.logical_not(keep_phenotype) * np.sum(phntp_per_cell == phenotype)

        # Add 'phenotype' to elements passing thresholds if there are any
        if np.any(keep_phenotype):
            _ = array_appending(phntp_to_keep[keep_phenotype], phenotype)
        # '_' is used to avoid 'array_appending' printing something to stdout

    return nb_phntp, phntp_to_keep, nb_undef_cells, phntp_per_cell
