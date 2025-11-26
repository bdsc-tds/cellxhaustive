"""
Script that determines number of unique cell phenotypes (combination of positive
and negative markers) and number of cells without phenotype in an expression
matrix across different metrics thresholds.
"""

# Import utility modules
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Import local functions
from determine_marker_status import determine_marker_status  # AT. Double-check path
# from cellxhaustive.determine_marker_status import determine_marker_status


# Function used in check_all_combinations.py
def score_marker_combinations(
    cell_name,
    comb_name,
    mat_comb,
    batches_label,
    samples_label,
    markers_comb,
    two_peak_threshold,
    three_peak_markers,
    three_peak_low,
    three_peak_high,
    min_samplesxbatch,
    min_cellxsample,
):
    """
    Function that determines number of unique cell phenotypes (combinations of
    positive and negative markers) and number of cells without phenotype in an
    expression matrix across different metrics thresholds.

    Parameters:
    -----------
    cell_name: str
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

    comb_name: str
      Combination name.

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
      Threshold to consider when determining whether a two peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three peaks marker is
      negative or low positive. Expression below this threshold means marker
      will be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three peaks marker is
      low_positive or positive. Expression above this threshold means marker
      will be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      'min_cellxsample' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of next parameter) in at least 50% of
      samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample in at least 50% of samples (see description of previous
      parameter) within a batch to be considered.

    Returns:
    --------
    nb_phntp: int
      Number of unique cell phenotypes (combinations of positive and negative
      markers from 'markers_comb') identified in 'mat_comb'.

    nb_undef_cells: int
      Number of undefined cells (cells without a phenotype) in 'mat_comb'.
    """

    # Determine markers status of 'markers_comb' using expression data
    logging.debug(
        f"\t\t\t\t\t{cell_name} - ({comb_name}): Determining marker status for each cell"
    )
    phntp_per_cell = determine_marker_status(
        mat_comb=mat_comb,
        markers_comb=markers_comb,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
    )

    # Initialise counters to store number of phenotypes and undefined cells
    nb_phntp = 0
    nb_undef_cells = 0

    # Process marker phenotypes returned by 'determine_marker_status()' and
    # check whether they are worth keeping
    logging.debug(
        f"\t\t\t\t\t{cell_name} - ({comb_name}): Checking which phenotypes are passing thresholds"
    )

    # Create categorical integer codes for faster operations
    # Note: map strings to integers "phenotype0" -> 0, "phenotype1" -> 1...
    # phntp_cat.codes = array of integers
    # phntp_cat.categories = array of original strings
    phntp_cat = pd.Categorical(phntp_per_cell)
    batch_cat = pd.Categorical(batches_label)
    sample_cat = pd.Categorical(samples_label)

    # Create sparse matrix with number of cells for each phenotype/sample pair
    # with the following structure:
    # - Shape: (nb_phenotypes, nb_samples)
    # - Rows: phenotypes
    # - Columns: samples
    # - phenotype_sample_mtx[phenotype_code, sample_code] = number of cells
    # Example: if cell 0 is phenotype 5 in sample 3, add 1 to mtx[5, 3]
    phenotype_sample_mtx = csr_matrix(
        (np.ones(len(phntp_per_cell)), (phntp_cat.codes, sample_cat.codes)),
        shape=(len(phntp_cat.categories), len(sample_cat.categories)),
    )

    # Map samples to batches
    # Example: {0: 0, 1: 0, 2: 1, ...} means samples 0, 1 are in batch 0
    sample_to_batch_map = (
        pd.Series(batch_cat.codes, index=sample_cat.codes)
        .groupby(level=0)
        .first()
    )

    # Initialise array to store 'phenotype' results
    keep_phenotypes = np.ones(len(phntp_cat.categories), dtype=bool)

    # Loop through batches
    for batch_idx in range(len(batch_cat.categories)):
        # Get all sample indices for current batch
        samples_in_batch = sample_to_batch_map[
            sample_to_batch_map == batch_idx
        ].index.values

        # Slice matrix according to batch: keep all phenotypes (rows) but only
        # samples present in current batch (cols)
        batch_matrix = phenotype_sample_mtx[:, samples_in_batch]

        # Loop through phenotypes and decide whether to keep them
        for phntp_idx in np.where(keep_phenotypes)[0]:
            # Get current phenotype counts across all samples in current batch
            # Note: .toarray() converts sparse to dense, .ravel() flattens to 1D
            # Example: [0, 5, 12] means 5 cells in sample 1, 12 in sample 2...
            cell_count_phntp_batch = batch_matrix[phntp_idx].toarray().ravel()

            # If there are no 'phenotype' cells in 'batch', that means it cannot
            # be present in all batches, so skip rest of checks
            if cell_count_phntp_batch.sum() == 0:
                keep_phenotypes[phntp_idx] = False
                continue

            # Count how many samples satisfy cell/sample threshold
            samples_above_threshold = np.sum(
                cell_count_phntp_batch >= min_cellxsample
            )

            # Count how many samples have any cells of this phenotype
            samples_in_batch_total = np.sum(cell_count_phntp_batch > 0)

            # Proportion of (samples with phenotype) that meet threshold

            # Calculate proportion of sample/batch passing cell/sample threshold
            sample_batch_prop = samples_above_threshold / samples_in_batch_total

            # If proportion is below threshold, discard phenotype
            if sample_batch_prop < min_samplesxbatch:
                keep_phenotypes[phntp_idx] = False

    # Count total number of phenotypes and undefined cells
    nb_phntp = np.sum(keep_phenotypes)
    nb_undef_cells = np.sum(
        np.isin(phntp_cat.codes, np.where(~keep_phenotypes)[0])
    )

    logging.debug(f"\t\t\t\t\t{cell_name} - ({comb_name}): Finished check")

    return nb_phntp, nb_undef_cells
    # Note: 'phntp_per_cell' is not returned to avoid memory cost of storing and
    # dragging it across several functions and will be recalculated when needed
