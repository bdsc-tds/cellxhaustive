"""
AT. Add general description here.
"""


# Import modules
import numpy as np


# Brute force way to select cells given normalized data
# AT. Check presence/absence of all parameters/variable
def select_cells(mat, markers, positive=[], negative=[], lowpositive=[],
                 three_markers=[], two_peak_threshold=3,
                 three_peak_lower_threshold=2, three_peak_upper_threshold=4):
    """
    Main gating strategy. AT. Improve description

    Parameters:
    -----------
    mat: ndarray
      A 2-D array normalized expression matrix.

    markers: array
      A 1-D array with the markers in mat corresponding to each column.

    positive: list(str)
      List of markers that are positively expressed.

    negative: list(str)
      List of markers that are negatively expressed.

    lowpositive: list(str)
      List of markers that are express a middle peak.

    three_markers: list(str)
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
    three_markers = np.array(three_markers)
    two_markers = markers[np.isin(markers, three_markers) == False]

    # Start by selecting on markers with three peaks
    if len(three_markers) > 0:
        # If positive, find values above the corresponding threshold
        for i in three_markers[np.isin(three_markers, positive)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] >= three_peak_upper_threshold)
        # If middle peak, find values above and below the corresponding threshold
        for i in three_markers[np.isin(three_markers, lowpositive)]:
            pos = np.logical_and(pos,
                                 np.logical_and(mat[:, np.where(markers == i)[0][0]] < three_peak_upper_threshold,
                                                mat[:, np.where(markers == i)[0][0]] >= three_peak_lower_threshold)
                                 )
        # If negative, find values below the corresponding threshold
        for i in three_markers[np.isin(three_markers, negative)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] < three_peak_lower_threshold)

    # Process markers with two peaks if there are any
    if len(two_markers) > 0:
        # If positive, find values above the corresponding threshold
        for i in two_markers[np.isin(two_markers, positive)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] >= two_peak_threshold)
        # If negative, find values below the corresponding threshold
        for i in two_markers[np.isin(two_markers, negative)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] < two_peak_threshold)

    return pos
