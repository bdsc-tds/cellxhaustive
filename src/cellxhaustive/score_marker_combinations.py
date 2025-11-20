"""
Script that determines number of unique cell phenotypes (combination of positive
and negative markers) and number of cells without phenotype in an expression
matrix across different metrics thresholds.
"""

# Import utility modules
import logging
import numpy as np


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
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means marker
      will be considered negative. See description of 'three_peak_high' for
      more information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
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
    for phenotype in np.unique(phntp_per_cell):
        # Initialise boolean marker deciding whether to keep 'phenotype'
        keep_phenotype = True

        # Process batches separately
        for batch in np.unique(batches_label):
            # Split phenotype data according to batch
            phenotype_batches = phntp_per_cell[batches_label == batch]

            # Split sample data, first according to batch and then phenotype
            phenotype_samples = samples_label[batches_label == batch][
                phenotype_batches == phenotype
            ]

            # If there are no 'phenotype' cells in 'batch', then it cannot be
            # present in all batches, so stop now and don't keep this phenotype
            if phenotype_samples.size == 0:
                keep_phenotype = False
                break

            # Calculate number of unique samples in current batch and phenotype
            samples_nb = float(len(np.unique(phenotype_samples)))

            # Count number of cells per phenotype in each sample
            cell_count_samples = np.asarray([
                np.sum(phenotype_samples == smpl)
                for smpl in np.unique(phenotype_samples)
            ])

            # Check whether counts satisfy cell/sample threshold
            cell_sample_bool = cell_count_samples >= min_cellxsample

            # Calculate proportion of samples in current batch satisfying
            # cell/sample threshold
            sample_batch_prop = np.sum(cell_sample_bool) / samples_nb
            # Note: 'cell_sample_bool' is a boolean array, so it can be summed

            # Check whether proportion satisfies sample/batch threshold
            keep_phenotype_batch = sample_batch_prop >= min_samplesxbatch

            # Intersect batch results with general results
            keep_phenotype = keep_phenotype and keep_phenotype_batch
            # Note: phenotypes should be present in all batches for consistency

        # If 'phenotype' is kept, increase phenotype counter
        if keep_phenotype:
            nb_phntp += 1
        else:
            # If 'phenotype' is rejected, increase undefined cells counter
            nb_undef_cells += np.sum(phntp_per_cell == phenotype)

    logging.debug(f"\t\t\t\t\t{cell_name} - ({comb_name}): Finished check")

    return nb_phntp, nb_undef_cells
    # Note: 'phntp_per_cell' is not returned to avoid memory cost of storing and
    # dragging it across several functions and will be recalculated when needed
