"""
Function that computes new names for lists of markers based on differences with
a main name. It identifies key markers distinguishing groups of markers from main
name and creates new names by minimising name differences.
"""


# Function used in assign_cell_types.py  # AT. Update script name if needed
def find_name_difference(base_comb, base_name, best_phntp):
    """
    Function that computes new names for lists of markers based on differences
    with a main name. It identifies key markers distinguishing groups of markers
    from main name and creates new names by minimising name differences.

    Parameters:
    -----------
    base_comb: str or list(str)
      String or list of strings containing marker combination(s) used to define
      base name(s) of cells groups.

    base_name: str or list(str)
      String or list of strings containing base(s) name for cell types (e.g.
      CD4 T-cells for 'CD4T').

    best_phntp: array(str)
      1-D numpy array containing strings made of concatenated lists of marker
      combinations. Each string is a different marker combination.

    Returns:
    --------
    names_conv: dict {str: str}
      Dictionary containing a name mapping between phenotypes (keys) and updated
      names (values) to annotate cells.
    """

    # Split 'base_comb' string into list of markers
    base_comb_lst = base_comb.split('/')

    # Initialise empty dict to store name conversion
    names_conv = {}

    # Fill dict with new names
    if isinstance(base_comb, str):  # 0 or 1 exact match, variables are strings
        for comb_str in best_phntp:
            if comb_str == base_comb:  # No need to find differences with 'base_comb'
                names_conv[comb_str] = f'Main {base_name} ({comb_str})'
            else:  # Find differences between 'comb_str' and 'base_comb'
                # Convert strings to lists to facilitate comparisons
                comb_str_lst = comb_str.split('/')
                # Find different markers
                diff_markers = sorted(list(set(comb_str_lst) - set(base_comb_lst)))
                # Add new name to dict
                names_conv[comb_str] = f"{base_name} ({', '.join(diff_markers)})"
    else:  # Several exact matches, variables are lists of strings
        base_comb_lst = [comb.split('/') for comb in base_comb]
        for comb_str in best_phntp:
            overlap = []
            if comb_str in base_comb:  # No need to find differences with 'base_comb'
                names_conv[comb_str] = f'Main {base_name} ({comb_str})'
            else:
                comb_str_lst = comb_str.split('/')
                # Calculate overlap between 'comb_str' and exact matches
                for comb_lst in base_comb_lst:
                    overlap.append(len(list(set(comb_lst) & set(comb_str_lst))))
                # Select maximum overlap
                max_overlap_idx = overlap.index(max(overlap))
                # Find different markers
                diff_markers = sorted(list(set(comb_str_lst) - set(base_comb_lst[max_overlap_idx])))
                # Add new name to dict
                names_conv[comb_str] = f"{base_name[max_overlap_idx]} ({', '.join(diff_markers)})"

    return names_conv
