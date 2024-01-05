"""
Function that determines status of markers (positive, negative or low_positive
in a case of a marker with three peaks, positive or negative in other cases)
depending on their expression.
"""


# Imports utility modules
import numpy as np


# Function used in marker_combinations_scoring.py
def determine_marker_status(mat_comb, markers_comb, three_peak_markers=[],
                            two_peak_threshold=3,
                            three_peak_lower_threshold=2,
                            three_peak_upper_threshold=4):
    """
    Function that determines status of markers (positive, negative or
    low_positive in a case of a marker with three peaks, positive or negative
    in other cases) depending on their expression.

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

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    two_peak_threshold: float (default=3)
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means the marker
      will be considered negative. Conversely, expression above this threshold
      means the marker will be considered positive.

    three_peak_lower_threshold: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means the marker
      will be considered negative. See description of 'three_peak_upper_threshold'
      for more information on low_positive markers.

    three_peak_upper_threshold: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
      low_positive or positive. Expression above this threshold means the marker
      will be considered positive. Expression between 'three_peak_lower_threshold'
      and 'three_peak_upper_threshold' means the marker will be considered low_positive.

    Returns:
    --------
    cell_types: array(str)
      1-D numpy array of strings matching order of cells in 'mat_comb'. Each
      string contains the cell marker types ('+', '-', 'low'), with markers
      order matching the one of 'markers_comb'.
    """

    # Create an array containing empty strings to store cell types
    cell_types = np.empty(mat_comb.shape[0], dtype=np.dtype('U100'))

    # Create a special array iterator
    iterator = np.nditer(mat_comb, flags=['multi_index'], order='C')

    # Iterate over expression matrix with the 'multi_index' flag to retrieve
    # current position in array
    for expression in iterator:

        # Extract row and col indices
        cell, marker_pos = iterator.multi_index

        # Determine marker type: positive, negative (or low_positive)
        if markers_comb[marker_pos] in three_peak_markers:  # Marker has 3 peaks
            if expression >= three_peak_upper_threshold:  # Marker is positive
                marker_type = '+'
            elif expression < three_peak_lower_threshold:  # Marker is negative
                marker_type = '-'
            else:  # Marker is in-between, i.e low_positive
                marker_type = 'low'

        else:  # Marker has 2 peaks
            if expression >= two_peak_threshold:  # Marker is positive
                marker_type = '+'
            else:  # Marker is negative
                marker_type = '-'

        # Append current marker type to string of marker types of corresponding
        # cell. Separate markers with a '/' if required
        if not cell_types[cell]:
            cell_types[cell] += f'{markers_comb[marker_pos]}{marker_type}'
        else:
            cell_types[cell] += f'/{markers_comb[marker_pos]}{marker_type}'

    return cell_types
