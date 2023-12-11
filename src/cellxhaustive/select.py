import numpy as np

# Brute force way to select cells given normalized data
def select_cells(mat, markers, positive=[], negative=[], lowpositive=[], three_marker=[], two_peak_threshold=3, three_peak_lower_threshold = 2, three_peak_upper_threshold = 4):
    """Main gating strategy

       Parameters
       ----------
       mat : ndarray
         A 2-D array normalized expression matrix

       markers : array
         A 1-D array with the markers in `mat` corresponding to each column

       positive : list(str)
         List of markers that are positively expressed.
         
       negative : list(str)
         List of markers that are negatively expressed.
         
       lowpositive : list(str)
         List of markers that are express a middle peak.

       three_marker : list(str)
         List of markers that have three peaks.
    """
    
    # Start a vector that will define the filtering
    pos = True
    
    # Find markers with two peaks and markers with three
    three_marker = np.array(three_marker)
    two_marker = markers[np.isin(markers, three_marker) == False]

    # Start by selecting on markers with three peaks
    if len(three_marker) > 0:
        # If positive, find values above the corresponding threshold
        for i in three_marker[np.isin(three_marker, positive)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] >= three_peak_upper_threshold)
        # If middle peak, find values above and below the corresponding threshold
        
        for i in three_marker[np.isin(three_marker, lowpositive)]:
            pos = np.logical_and(pos,
                                 np.logical_and(mat[:, np.where(markers == i)[0][0]] < three_peak_upper_threshold,
                                                mat[:, np.where(markers == i)[0][0]] >= three_peak_lower_threshold)
                                 )

        # If negative, find values below the corresponding threshold
        for i in three_marker[np.isin(three_marker, negative)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] < three_peak_lower_threshold)

    if len(two_marker) > 0:
        # If positive, find values above the corresponding threshold
        for i in two_marker[np.isin(two_marker, positive)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] >= two_peak_threshold)

        # If negative, find values below the corresponding threshold
        for i in two_marker[np.isin(two_marker, negative)]:
            pos = np.logical_and(pos, mat[:, np.where(markers == i)[0][0]] < two_peak_threshold)

    return pos
