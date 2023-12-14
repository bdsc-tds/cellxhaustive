"""
AT. Add general description here.
"""


# Import modules
import numpy as np


# Utility function to rename cell types
def find_set_differences(cell_groups_renaming, baseline_name='baseline'):
    """
    # AT. Double-check description in doc-string
    Find key differences across dictionaries. Provided a dictionary where every
    key corresponds to a given cell line, and every value is the set of
    positive/negative markers for that same cell line, the function will provide
    a dictionary with the key differences between cell types.

    Parameters:
    -----------
    cell_groups_renaming : dict(set(str))
      Imagine that you have three cell lines (CD4 T, CD8 T and DCs) that are
      characterized by three markers: CD3, CD4, CD8. cell_groups_renaming is a
      dictionary such as:
      cell_groups_renaming['CD4 T'] = set(['CD3+', 'CD4+', 'CD8-'])

    baseline_name : str (default='baseline')
      Name given to the baseline cell type picked.

    Returns:
    --------
      Dictionary with key differences between cell types. Following the
    example above, the function will return the following dictionary:
    {'CD4 T': 'CD4+', 'CD8 T': 'CD8+', 'DCs': 'CD3-'}
    """

    markers, cnts = np.unique([y for x in cell_groups_renaming.keys()
                               for y in list(cell_groups_renaming[x])],
                              return_counts=True)
    common_markers = markers[np.flip(np.argsort(cnts))]

    # Identify what are the key markers that distinguish the different groups
    # and define the baseline based on the shortest combination
    keep_markers = []
    common_first = set()
    for marker in common_markers:
        marker_cleaned = marker.replace('-', '').replace('+', '')
        if marker_cleaned not in common_first:
            common_first.add(marker_cleaned)
            keep_markers.append(marker)

    for cell_pop in cell_groups_renaming.keys():
        cell_groups_renaming[cell_pop] = ' '.join(np.sort(list(cell_groups_renaming[cell_pop] - set(keep_markers))))
        if cell_groups_renaming[cell_pop] == '' and cell_pop != -1:
            cell_groups_renaming[cell_pop] = baseline_name
        elif cell_pop == -1:
            cell_groups_renaming[cell_pop] = 'undefined'

    return cell_groups_renaming
