"""
AT. Add general description here.
"""


# Import modules
import numpy as np


# Brute force way to select cells given normalized data
# AT. Check presence/absence of all parameters/variable
def select_cells(mat_comb, markers_comb,
                 positive=[], negative=[], lowpositive=[],
                 two_peak_threshold=3,
                 three_markers_comb=[],
                 three_peak_lower_threshold=2, three_peak_upper_threshold=4):
    """
    Main gating strategy. AT. Improve description

    Parameters:
    -----------
    mat_comb: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    markers_comb: array(str)
      1-D numpy array with markers matching each column of mat_comb.



    positive: list(str)
      List of markers that are positively expressed.

    negative: list(str)
      List of markers that are negatively expressed.

    lowpositive: list(str)
      List of markers that are express a middle peak.

    three_markers_comb: list(str)
      List of markers that have three peaks.

    two_peak_threshold: int (default=3)
      # AT. Add parameter description

    three_peak_lower_threshold: int (default=2)
      # AT. Add parameter description

    three_peak_upper_threshold: int (default=4)
      # AT. Add parameter description

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Start a vector that will define the filtering
    pos = True

    # Find markers with two peaks and markers with three
    # three_markers_comb = np.array(three_markers_comb)  # AT. Double-check if directly inputing an array in cell_subdivision_counts works
    two_markers = markers_comb[np.isin(markers_comb, three_markers_comb) == False]

    # Start by selecting on markers with three peaks
    if len(three_markers_comb) > 0:
        # If positive, find values above the corresponding threshold
        for i in three_markers_comb[np.isin(three_markers_comb, positive)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == i)[0][0]] >= three_peak_upper_threshold)
        # If middle peak, find values above and below the corresponding threshold
        for i in three_markers_comb[np.isin(three_markers_comb, lowpositive)]:
            pos = np.logical_and(pos,
                                 np.logical_and(mat_comb[:, np.where(markers_comb == i)[0][0]] < three_peak_upper_threshold,
                                                mat_comb[:, np.where(markers_comb == i)[0][0]] >= three_peak_lower_threshold)
                                 )
        # If negative, find values below the corresponding threshold
        for i in three_markers_comb[np.isin(three_markers_comb, negative)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == i)[0][0]] < three_peak_lower_threshold)

    # Process markers with two peaks if there are any
    if len(two_markers) > 0:
        # If positive, find values above the corresponding threshold
        for i in two_markers[np.isin(two_markers, positive)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == i)[0][0]] >= two_peak_threshold)
        # If negative, find values below the corresponding threshold
        for i in two_markers[np.isin(two_markers, negative)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == i)[0][0]] < two_peak_threshold)

    return pos
