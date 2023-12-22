"""
AT. Add general description here.
"""


# Import modules
import numpy as np


# Brute force way to select cells given normalized data
# AT. Check presence/absence of all parameters/variable
def select_cells(mat_comb, markers_comb,
                 positives=[], negatives=[], lowpositives=[],
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



    positives: list(str)
      List of markers that are positively expressed.

    negatives: list(str)
      List of markers that are negatively expressed.

    lowpositives: list(str)
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
    two_markers = markers_comb[np.isin(markers_comb, three_markers_comb, invert=True)]

    # Start by selecting on markers with three peaks
    if len(three_markers_comb) > 0:
        # If positive, find values above the corresponding threshold
        for marker in three_markers_comb[np.isin(three_markers_comb, positives)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == marker)[0][0]] >= three_peak_upper_threshold)
            # AT. np.where(markers_comb == marker)[0][0] selects the position index of 'marker' in 'markers_comb'
            # AT. mat_comb[:, np.where(markers_comb == marker)[0][0] selects the expression value for 'marker'
            # AT. mat_comb[:, np.where(markers_comb == marker)[0][0]] >= three_peak_upper_threshold selects cells with expression value passing 'three_peak_upper_threshold' threshold
            # AT. np.logical_and is used to replace cells fitting this pattern in 'pos'
        # If middle peak, find values above and below the corresponding threshold
        for marker in three_markers_comb[np.isin(three_markers_comb, lowpositives)]:
            pos = np.logical_and(pos,
                                 np.logical_and(mat_comb[:, np.where(markers_comb == marker)[0][0]] < three_peak_upper_threshold,
                                                mat_comb[:, np.where(markers_comb == marker)[0][0]] >= three_peak_lower_threshold)
                                 )
            # AT. Same idea than before, but selecting cells with expression between 'three_peak_upper_threshold' and 'three_peak_lower_threshold'
        # If negative, find values below the corresponding threshold
        for marker in three_markers_comb[np.isin(three_markers_comb, negatives)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == marker)[0][0]] < three_peak_lower_threshold)
            # AT. Same idea than before, but selecting cells with expression lower than 'three_peak_lower_threshold'

    # Process markers with two peaks if there are any
    if len(two_markers) > 0:
        # If positive, find values above the corresponding threshold
        for marker in two_markers[np.isin(two_markers, positives)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == marker)[0][0]] >= two_peak_threshold)
            # AT. Same idea than before, but selecting cells with expression higher than 'two_peak_threshold'
        # If negative, find values below the corresponding threshold
        for marker in two_markers[np.isin(two_markers, negatives)]:
            pos = np.logical_and(pos, mat_comb[:, np.where(markers_comb == marker)[0][0]] < two_peak_threshold)
            # AT. Same idea than before, but selecting cells with expression lower than 'two_peak_threshold'
            # AT. Could be replaced by taking the opposite of positive values, as there is only two states in a two-peaks marker: positive or negative

    return pos
