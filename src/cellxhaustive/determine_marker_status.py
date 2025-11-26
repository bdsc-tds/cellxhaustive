"""
Script that determines status of markers (positive, negative or low_positive
in a case of a marker with three peaks, positive or negative in other cases)
depending on their expression.
"""

# Import utility modules
import numpy as np


# Function used in score_marker_combinations.py
def determine_marker_status(
    mat_comb,
    markers_comb,
    two_peak_threshold,
    three_peak_markers,
    three_peak_low,
    three_peak_high,
):
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

    Returns:
    --------
    phntp_per_cell: array(str)
      1-D numpy array of strings matching order of cells in 'mat_comb'. Each
      string contains cell marker status ('+', '-', 'low'), with markers order
      matching that of 'markers_comb'.
    """

    # Create three peaks markers mask
    is_three = np.isin(markers_comb, three_peak_markers)

    # Initialize marker status array
    # Use dtype=object to allow storing strings of different lengths
    mat_comb_obj = np.empty_like(mat_comb, dtype=object)

    # Find marker status
    if np.any(is_three):
        # With three peaks markers. Find their status first
        mat_comb_obj[:, is_three] = np.select(
            [
                mat_comb[:, is_three] >= three_peak_high,
                mat_comb[:, is_three] < three_peak_low,
            ],
            ["+", "-"],
            default="low",
        )
        # Find status of two peaks markers
        mat_comb_obj[:, ~is_three] = np.where(
            mat_comb[:, ~is_three] < two_peak_threshold, "-", "+"
        )
    else:
        # Only two peaks markers
        mat_comb_obj = np.where(mat_comb < two_peak_threshold, "-", "+")

    # Concatenate markers and status
    markers_comb_expanded = np.tile(markers_comb, (mat_comb_obj.shape[0], 1))
    mat_comb_obj = np.char.add(markers_comb_expanded, mat_comb_obj)

    # Concatenate all columns into one
    phntp_per_cell = np.array(
        ["/".join(row) for row in mat_comb_obj], dtype=str
    )

    return phntp_per_cell
