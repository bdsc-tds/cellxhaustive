"""
Function that searches for matches between combinations of markers and a list of
markers-defined cell types (i.e: cell type 1 is A+, B-, C-, cell type 2 is A-,
C+, D+...). If one or several match(es) is(are) found, marker combination(s) will
be assigned corresponding cell type(s) from list and other combinations will be
assigned names derived from exact match(es). If no match is found, combination
with most cells will be used as base name and other combinations will be assigned
names derived from this base name.
"""

# Import utility modules
import logging
import numpy as np


# Import local functions
from cellxhaustive.find_name_difference import find_name_difference


# Function used in identify_phenotypes.py
def assign_cell_types(
    cell_name,
    best_comb_name,
    mat_representative,
    batches_label,
    samples_label,
    markers_representative,
    cell_types_dict,
    cell_phntp,
    best_phntp,
):
    """
    Function that searches for matches between combinations of markers and a
    list of markers-defined cell types (i.e: cell type 1 is A+, B-, C-, cell
    type 2 is A-, C+, D+...). If one or several match(es) is(are) found, marker
    combination(s) will be assigned corresponding cell type(s) from list and
    other combinations will be assigned names derived from exact match(es). If
    no match is found, combination with most cells will be used as base name and
    other combinations will be assigned names derived from this base name.

    Parameters:
    -----------
    cell_name: str
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

    best_comb_name: str
      Best combination name.

    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_representative'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_representative'.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of 'mat_representative'.

    cell_types_dict: dict({str: list()})
      Dictionary with cell types as keys and list of cell type defining markers
      as values.

    cell_phntp_comb: array(str)
      1-D numpy array of strings with phenotype found for each cell using
      markers from associated 'markers_representative' tuple.

    best_phntp_comb: array(str)
      1-D numpy array of strings with representative phenotypes among all
      possible phenotypes from 'markers_representative'.

    Returns:
    --------
    new_labels: array(str)
      1-D numpy array of strings with new names for each cell of
      'mat_representative'.

    names_conv: dict({str: str})
      Dictionary with name mapping between phenotypes (keys) and updated names
      (values) to annotate cells.
    """

    # Trim down cell classification to remove any marker that is not present
    # in 'markers_representative'
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Trimming cell classification to keep only relevant markers"
    )
    markers_set = set(sorted(markers_representative))
    cell_types_filtered = {}
    for cell_type, cell_mkers in cell_types_dict.items():
        # Keep markers present in markers_representative
        mkers_lst_clean = [
            mker for mker in cell_mkers if mker[:-1] in markers_set
        ]
        if mkers_lst_clean:
            cell_types_filtered[cell_type] = mkers_lst_clean

    # Reduce 'cell_types_filtered' redundancy
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Reducing redundancy in new classification"
    )
    nb_of_prot = {k: len(v) for k, v in cell_types_dict.items()}

    # Group by marker list to find duplicates
    markers_to_types = {}
    for cell_type, cell_mkers in cell_types_filtered.items():
        key = tuple(cell_mkers)
        if key not in markers_to_types:
            markers_to_types[key] = []
        markers_to_types[key].append(cell_type)

    # Keep only minimum cell type for each unique marker list
    cell_types_clean = {}
    for marker_list, cell_types in markers_to_types.items():
        if len(cell_types) == 1:
            cell_types_clean[cell_types[0]] = sorted(marker_list)
        else:
            # Keep the one with minimum original marker count
            min_type = min(cell_types, key=lambda x: nb_of_prot[x])
            cell_types_clean[min_type] = sorted(marker_list)

    # Determine number of exact matches between phenotypes from 'best_phntp'
    # and marker lists from 'cell_types_clean'
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Determining exact matches between phenotypes and cell classification"
    )

    # Build reverse mapping for faster lookup
    markers_to_cell_type = {tuple(v): k for k, v in cell_types_clean.items()}

    # Match phenotypes and cell types
    phntp_match = []
    cell_types_match = []
    for phntp_str in best_phntp:
        phntp = phntp_str.split("/")
        phntp_tuple = tuple(phntp)
        if phntp_tuple in markers_to_cell_type:
            phntp_match.append(phntp_str)
            cell_types_match.append(markers_to_cell_type[phntp_tuple])

    # Determine base name(s) for all phenotypes
    n_matches = len(phntp_match)
    if n_matches == 0:  # No exact match
        # Most present phenotype will be used as base name
        mask = np.isin(cell_phntp, best_phntp)
        uniq_phntp, phntp_count = np.unique(
            cell_phntp[mask], return_counts=True
        )
        base_comb = uniq_phntp[phntp_count.argmax()]
        base_name = cell_name
        logging.info(
            f"\t\t\t\t\t\t{cell_name} - ({best_comb_name}): No exact match between phenotypes and cell classification"
        )
    elif n_matches == 1:  # One exact match that was already determined
        base_comb = phntp_match[0]
        base_name = cell_types_match[0]
        logging.info(
            f"\t\t\t\t\t\t{cell_name} - ({best_comb_name}): 1 exact match between phenotypes and cell classification"
        )
    else:  # Several exact matches that were already determined
        base_comb = phntp_match
        base_name = cell_types_match
        logging.info(
            f"\t\t\t\t\t\t{cell_name} - ({best_comb_name}): Found {n_matches} exact matches between phenotypes and cell classification"
        )

    # Get mapping dictionary to convert names
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Building dictionary to convert cell type names"
    )
    names_conv = find_name_difference(
        base_comb=base_comb, base_name=base_name, best_phntp=best_phntp
    )

    # Convert phenotypes to new names
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Renaming cell types"
    )
    new_labels = np.vectorize(names_conv.get, otypes=[str])(
        cell_phntp, f"Unannotated {cell_name}"
    )
    # Note: with dict.get method, non-representative phenotypes (missing from
    # 'names_conv') are automatically converted to 'Unannotated {cell_name}'

    return new_labels, names_conv
