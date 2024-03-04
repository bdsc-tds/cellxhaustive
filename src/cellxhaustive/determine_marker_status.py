"""
Script that determines status of markers (positive, negative or low_positive
in a case of a marker with three peaks, positive or negative in other cases)
depending on their expression.
"""


# Import utility modules
import numpy as np
from multiprocessing import Pool
from functools import partial


# Function used in determine_marker_status()
def get_marker_status(expression_array, markers_array, tp, tpm, tpl, tph):
    """
    Function that determines status of markers (positive, negative or low_positive
    in a case of a marker with three peaks, positive or negative in other cases)
    depending on their expression.

    Parameters:
    -----------
    expression_array: array(float)
      1-D numpy array expression matrix containing marker expression for one
      cell. This matrix is a subset of the general expression matrix and contains
      sliced data matching cell label, batch, and representative markers.

    markers_array: array(str)
      1-D numpy array with markers matching each column of 'expression_array'.

    tp: float (default=3)
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means marker will be
      considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    tpm: list(str) (default=[])
      List of markers that have three peaks.

    tpl: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means marker
      will be considered negative. See description of 'tph' for more information
      on low_positive markers.

    tph: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
      low_positive or positive. Expression above this threshold means marker
      will be considered positive. Expression between 'tpl' and 'tph' means
      marker will be considered low_positive.

    Returns:
    --------
    status: str
      String containing cell marker status ('+', '-', 'low'), with markers order
      matching that of 'markers_array'.
    """

    status = ''
    # Iterate over markers
    for idx, marker in enumerate(markers_array):
        expression = expression_array[idx]
        # Determine marker type: positive, negative (or low_positive)
        if marker in tpm:  # Marker has 3 peaks
            if expression >= tph:  # Marker is positive
                marker_type = '+'
            elif expression < tpl:  # Marker is negative
                marker_type = '-'
            else:  # Marker is in-between, i.e low_positive
                marker_type = 'low'

        else:  # Marker has 2 peaks
            if expression >= tp:  # Marker is positive
                marker_type = '+'
            else:  # Marker is negative
                marker_type = '-'

        # Append current marker type to string of marker types of corresponding
        # cell. Separate markers with a '/' if required
        if not status:  # String is empty
            status = f'{marker}{marker_type}'
        else:  # String is not empty, so markers are separated with '/'
            status += f'/{marker}{marker_type}'

    return status


# Function used in score_marker_combinations.py
def determine_marker_status(mat_comb, markers_comb, two_peak_threshold,  # AT. CPU param?
                            three_peak_markers, three_peak_low, three_peak_high):
    """
    Function that multiprocesses marker status computing.

    Parameters:
    -----------
    mat_comb: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

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

    Returns:
    --------
    phntp_per_cell: array(str)
      1-D numpy array of strings matching order of cells in 'mat_comb'. Each
      string contains cell marker status ('+', '-', 'low'), with markers order
      matching that of 'markers_comb'.
    """

    # Turn 2D 'mat_comb' into 1D iterable of float arrays. Each array contains
    # marker expression for one cell
    total_expression = np.fromiter(mat_comb, dtype=list)

    # Compute marker status using multiprocessing
    with Pool() as pool:  # AT. CPU param?
        status_results_lst = pool.map(partial(get_marker_status,
                                              markers_array=markers_comb,
                                              tp=two_peak_threshold,
                                              tpm=three_peak_markers,
                                              tpl=three_peak_low,
                                              tph=three_peak_high),
                                      total_expression)
    # Notes:
    #   - More efficient to parallelise by cell than marker
    #   - Only 'total_expression' is iterated over, hence the use of 'partial()'
    #   to keep the other parameters constant

    # Convert results back to str array
    phntp_per_cell = np.array(status_results_lst, dtype=str)

    return phntp_per_cell
