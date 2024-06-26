"""
Script that determines number of unique cell phenotypes (combination of positive
and negative markers) and number of cells without phenotype in an expression matrix
across different metrics thresholds.
"""


# Import utility modules
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# Import local functions
from cellxhaustive.determine_marker_status import determine_marker_status
from utils import get_chunksize


# Wrapper function of 'list.append()' for vectorised use in numpy
# Function used in score_marker_combinations()
def append_wrapper(lst, elt):
    return lst.append(elt)


# Function used in score_marker_combinations()
def keep_relevant_phntp(batch, batches_label, samples_label, phntp_per_cell,
                        phntp, x_samplesxbatch_space, y_cellxsample_space):
    """
    Function that determines whether a phenotype is relevant, depending on
    samplesxbatch and cellxsample thresholds.

    Parameters:
    -----------
    batch: str
      Value of batch to process.

    batches_label: array(str)
      1-D numpy array with batch names of each cell.

    samples_label: array(str)
      1-D numpy array with sample names of each cell.

    phntp_per_cell: array(str)
      1-D numpy array with best phenotype determined for each cell.

    phntp: str
      Value of phenotype to process.

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
    keep_phntp_batch: array(bool)
      2-D numpy array with booleans indicating whether 'phenotype' is relevant
      in 'batch' across grid composed of metrics 'x_samplesxbatch_space' in D0
      and 'y_cellxsample_space' in D1.
    """

    # Split cell type data according to batch
    phntp_batch = phntp_per_cell[batches_label == batch]

    # Split sample data, first according to batch and then cell type
    phntp_smpl = samples_label[batches_label == batch][phntp_batch == phntp]

    # Calculate number of different samples in current batch and cell type
    smpl_nb = float(len(np.unique(phntp_smpl)))

    # If there are no cells of type 'phntp' in 'batch', there is no need to go
    # further, so return a False array with right dimensions
    if smpl_nb == 0:
        array_dim = (len(x_samplesxbatch_space), len(y_cellxsample_space))
        keep_phntp_batch = np.full(array_dim, False)
        return keep_phntp_batch

    # Count number of cells per phntp in each sample
    cell_count_smpl = np.asarray([np.sum(phntp_smpl == smpl)
                                  for smpl in np.unique(phntp_smpl)])

    # Check whether cell counts satisfy y threshold
    keep_phntp_batch = (cell_count_smpl[:, np.newaxis] >= y_cellxsample_space)
    # Note: np.newaxis is used to add a dimension to work on different
    # samples concurrently

    # Calculate proportion of samples in current batch satisfying
    # 'y_cellxsample_space' condition
    keep_phntp_batch = (np.sum(keep_phntp_batch, axis=0) / smpl_nb)
    # Notes:
    # - 'keep_phntp_batch' is a boolean array, so it can be summed
    # - 'np.sum(keep_phntp_batch, axis=0)' calculates number of samples
    # satisfying y condition for a given y in grid

    # Check whether proportion satisfy x threshold
    keep_phntp_batch = (keep_phntp_batch >= x_samplesxbatch_space[:, np.newaxis])
    # Note: '[:, np.newaxis]' is used to transpose 1-D array into a 2-D
    # array to allow comparison

    return keep_phntp_batch


# Function used in check_all_combinations.py
def score_marker_combinations(mat_comb, batches_label, samples_label,
                              markers_comb, two_peak_threshold,
                              three_peak_markers, three_peak_low, three_peak_high,
                              x_samplesxbatch_space, y_cellxsample_space, nb_cpu_keep):
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

    nb_cpu_keep: int (default=1)
      Integer to set up CPU numbers in downstream nested functions.

    Returns:
    --------
    nb_phntp: array(float)
      2-D numpy array with number of unique cell phenotypes (combinations of
      positive and negative markers from 'markers_comb') identified in 'mat_comb'
      across grid composed of metrics 'x_samplesxbatch_space' in D0 and
      'y_cellxsample_space' in D1.

    phntp_to_keep: array(list(str))
      2-D numpy array with lists of significant phenotypes across grid composed
      of metrics 'x_samplesxbatch_space' in D0 and 'y_cellxsample_space' in D1.
      In each element, len(phntp_to_keep[i, j]) = nb_phntp[i, j].

    nb_undef_cells: array(float)
      2-D numpy array with number of undefined cells (cells without a phenotype)
      in 'mat_comb' across grid composed of metrics 'x_samplesxbatch_space' in
      D0 and 'y_cellxsample_space' in D1.

    phntp_per_cell: array(str)
      1-D numpy array with best phenotype determined for each cell using markers
      from 'markers_comb'.
    """

    # Determine markers status of 'markers_comb' using expression data
    logging.debug(f'\t\t\t\t\t\tDetermining marker status for each cell')
    phntp_per_cell = determine_marker_status(
        mat_comb=mat_comb,
        markers_comb=markers_comb,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
        nb_cpu_keep=nb_cpu_keep)

    # Initialise arrays to store results
    nb_phntp = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    nb_undef_cells = np.zeros((len(x_samplesxbatch_space), len(y_cellxsample_space)))
    phntp_to_keep = np.empty(len(x_samplesxbatch_space) * len(y_cellxsample_space),
                             dtype=object)
    phntp_to_keep[...] = [[] for _ in range(len(phntp_to_keep))]
    phntp_to_keep = np.reshape(phntp_to_keep, (len(x_samplesxbatch_space),
                                               len(y_cellxsample_space)))
    # Note: using 'dtype=list' fills array with None, hence this trick

    # Vectorized version of append wrapper
    array_appending = np.vectorize(append_wrapper, otypes=[str])

    # Get array of unique batches
    uniq_batches = np.unique(batches_label)

    # Process marker phenotypes returned by 'determine_marker_status()' and
    # check whether they are worth keeping
    logging.debug(f'\t\t\t\t\t\tChecking which phenotypes are passing thresholds')
    for phenotype in np.unique(phntp_per_cell):
        # Process batches using multiprocessing
        chunksize = get_chunksize(uniq_batches, nb_cpu_keep)
        with ProcessPoolExecutor(max_workers=nb_cpu_keep) as executor:
            keep_phntp_lst = list(executor.map(partial(keep_relevant_phntp,
                                                       batches_label=batches_label,
                                                       samples_label=samples_label,
                                                       phntp_per_cell=phntp_per_cell,
                                                       phntp=phenotype,
                                                       x_samplesxbatch_space=x_samplesxbatch_space,
                                                       y_cellxsample_space=y_cellxsample_space),
                                               uniq_batches,
                                               chunksize=chunksize))
        # Notes: only 'uniq_batches' is iterated over, hence the use of
        # 'partial()' to keep the other parameters constant

        # Intersect all batch results to retain phenotypes present in all
        keep_phenotype = np.logical_and.reduce(keep_phntp_lst)
        # Note: for consistency, phntps have to be present in all batches,
        # hence usage of 'np.logical_and()'

        # Add 'phenotype' presence/absence to cell type counter
        nb_phntp += keep_phenotype * 1

        # Add number of undefined cells to counter
        nb_undef_cells += np.logical_not(keep_phenotype) * np.sum(phntp_per_cell == phenotype)

        # Add 'phenotype' to elements passing thresholds if there are any
        if np.any(keep_phenotype):
            _ = array_appending(phntp_to_keep[keep_phenotype], phenotype)
        # '_' is used to avoid 'array_appending' printing something to stdout
    logging.debug(f'\t\t\t\t\t\tFinished check')

    return nb_phntp, phntp_to_keep, nb_undef_cells, phntp_per_cell
