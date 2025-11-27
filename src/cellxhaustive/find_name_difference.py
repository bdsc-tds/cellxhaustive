"""
Function that computes new names for lists of markers based on differences with
a main name. It identifies key markers distinguishing groups of markers from
main name and creates new names by minimising name differences.
"""


# Function used in assign_cell_types.py
def find_name_difference(base_comb, base_name, best_phntp):
    """
    Function that computes new names for lists of markers based on differences
    with a main name. It identifies key markers distinguishing groups of markers
    from main name and creates new names by minimising name differences.

    Parameters:
    -----------
    base_comb: str or list(str)
      String or list of strings with marker combination(s) used to define base
      name(s) of cells groups.

    base_name: str or list(str)
      String or list of strings with base(s) name for cell types (e.g.
      CD4 T-cells for 'CD4T').

    best_phntp: array(str)
      1-D numpy array with strings made of concatenated lists of marker
      combinations. Each string is a different marker combination.

    Returns:
    --------
    names_conv: dict({str: str})
      Dictionary with name mapping between phenotypes (keys) and updated names
      (values) to annotate cells.
    """

    # Initialise empty dictionary to store name conversion
    names_conv = {}

    # Fill dictionary with new names
    if isinstance(base_comb, str):  # 0 or 1 exact match, variables are strings
        # Split 'base_comb' string into set of markers
        base_set = set(base_comb.split("/"))
        # Loop through all phenotypes
        for comb_str in best_phntp:
            # 'comb_str' and 'base_comb' are identical, no need to look for
            # differences
            if comb_str == base_comb:
                names_conv[comb_str] = base_name
            else:  # Find differences between 'comb_str' and 'base_comb'
                # Split 'comb_str' string into set of markers
                comb_set = set(comb_str.split("/"))
                # Find different markers
                diff_markers = sorted(comb_set - base_set)
                # Add new name to dictionary
                names_conv[comb_str] = (
                    f"{base_name} ({', '.join(diff_markers)})"
                )
    else:  # Several exact matches, variables are lists of strings
        # Split 'base_comb' string list into list of set of markers
        base_sets = [set(comb.split("/")) for comb in base_comb]
        base_comb_set = set(base_comb)  # For fast membership check
        # Loop through all phenotypes
        for comb_str in best_phntp:
            # 'comb_str' exists in 'base_comb_set', no need to look for
            # differences
            if comb_str in base_comb_set:
                names_conv[comb_str] = base_name[base_comb.index(comb_str)]
            else:
                # Split 'comb_str' string into set of markers
                comb_set = set(comb_str.split("/"))

                # Find maximum overlap using set intersection
                overlaps = [len(base_set & comb_set) for base_set in base_sets]
                max_overlap_idx = overlaps.index(max(overlaps))

                # Find different markers
                diff_markers = sorted(comb_set - base_sets[max_overlap_idx])
                names_conv[comb_str] = (
                    f"{base_name[max_overlap_idx]} ({', '.join(diff_markers)})"
                )

    return names_conv
