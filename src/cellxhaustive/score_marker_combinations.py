"""
Script that determines number of unique cell phenotypes (combination of positive
and negative markers) and number of cells without phenotype in an expression matrix
across different metrics thresholds.
"""


# Import utility modules
import logging
import numpy as np


# Import local functions
from cellxhaustive.determine_marker_status import determine_marker_status


# Function used in check_all_combinations.py
def score_marker_combinations(mat_comb, batches_label, samples_label,
                              markers_comb, two_peak_threshold,
                              three_peak_markers, three_peak_low, three_peak_high,
                              x_samplesxbatch_space, y_cellxsample_space):
    """
    Function that determines number of unique cell phenotypes (combination of
    positive and negative markers) and number of cells without phenotype in an
    expression matrix across different metrics thresholds.

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

    two_peak_threshold: float (default=3)
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means marker will
      be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
      low_positive or positive. Expression above this threshold means marker will
      be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

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
      2-D numpy array with number of unique cell phenotypes (combinations of
      positive and negative markers from 'markers_comb') identified in 'mat_comb'
      across grid composed of metrics 'x_samplesxbatch_space' in D0 and
      'y_cellxsample_space' in D1.

    nb_undef_cells: array(float)
      2-D numpy array with number of undefined cells (cells without a phenotype)
      in 'mat_comb' across grid composed of metrics 'x_samplesxbatch_space' in
      D0 and 'y_cellxsample_space' in D1.
    """

    # Determine markers status of 'markers_comb' using expression data
    logging.debug('\t\t\t\t\t\tDetermining marker status for each cell')
    phntp_per_cell = determine_marker_status(
        mat_comb=mat_comb,
        markers_comb=markers_comb,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high)

    # Initialise arrays to store results
    nb_phntp = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    nb_undef_cells = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))

    # Process marker phenotypes returned by 'determine_marker_status()' and
    # check whether they are worth keeping
    logging.debug('\t\t\t\t\t\tChecking which phenotypes are passing thresholds')
    for phenotype in np.unique(phntp_per_cell):

        # Initialise temporary array to store 'phenotype' results
        keep_phenotype = np.full(nb_phntp.shape, True)

        # Process batches separately
        for batch in np.unique(batches_label):
            # Split phenotype data according to batch
            phenotypes_batch = phntp_per_cell[batches_label == batch]

            # Split sample data, first according to batch and then phenotype
            phenotype_samples = samples_label[batches_label == batch][phenotypes_batch == phenotype]

            # If there are no 'phenotype' cells in 'batch', that means it cannot
            # be present in all batches, so stop now
            if phenotype_samples.size == 0:
                keep_phenotype = np.logical_and(keep_phenotype, False)
                break

            # Calculate number of unique samples in current batch and phenotype
            samples_nb = float(len(np.unique(phenotype_samples)))

            # Count number of cells per phenotype in each sample
            cell_count_sample = np.asarray([np.sum(phenotype_samples == smpl)
                                            for smpl in np.unique(phenotype_samples)])

            # Check whether previous counts satisfy cell/sample threshold
            keep_phenotype_batch = cell_count_sample[:, np.newaxis] >= y_cellxsample_space
            # Note: np.newaxis is used to add a dimension to work on different
            # samples concurrently

            # Calculate proportion of samples in current batch satisfying
            # cell/sample threshold
            keep_phenotype_batch = (np.sum(keep_phenotype_batch, axis=0) / samples_nb)
            # Notes:
            # - 'keep_phenotype_batch' is a boolean array, so it can be summed
            # - 'np.sum(keep_phenotype_batch, axis=0)' calculates number of samples
            # satisfying cell/sample threshold for a given y in grid

            # Check whether previous proportions satisfy sample/batch threshold
            keep_phenotype_batch = keep_phenotype_batch >= x_samplesxbatch_space[:, np.newaxis]
            # Note: '[:, np.newaxis]' is used to transpose 1-D array into a 2-D
            # array to allow comparison

            # Intersect batch results with general results
            keep_phenotype = np.logical_and(keep_phenotype, keep_phenotype_batch)
            # Note: for consistency, phenotypes have to be present in all batches,
            # hence usage of 'np.logical_and()'

        # Add 'phenotype' presence/absence to phenotype counter
        nb_phntp += keep_phenotype * 1

        # Add number of undefined cells to counter
        nb_undef_cells += np.logical_not(keep_phenotype) * np.sum(phntp_per_cell == phenotype)

    logging.debug('\t\t\t\t\t\tFinished check')

    return nb_phntp, nb_undef_cells
    # Note: 'phntp_per_cell' is not returned to avoid memory cost of storing and
    # dragging it across several functions and will be recalculated when needed.
