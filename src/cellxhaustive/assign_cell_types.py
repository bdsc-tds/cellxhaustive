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
from find_name_difference import find_name_difference  # AT. Double-check path
# from cellxhaustive.find_name_difference import find_name_difference


# Function used in identify_phenotypes.py
def assign_cell_types(mat_representative,
                      batches_label,
                      samples_label,
                      markers_representative,
                      cell_types_dict,
                      cell_name,
                      cell_phntp,
                      best_phntp):
    """
    Function that searches for matches between combinations of markers and a list
    of markers-defined cell types (i.e: cell type 1 is A+, B-, C-, cell type 2
    is A-, C+, D+...). If one or several match(es) is(are) found, marker
    combination(s) will be assigned corresponding cell type(s) from list and other
    combinations will be assigned names derived from exact match(es). If no match
    is found, combination with most cells will be used as base name and other
    combinations will be assigned names derived from this base name.

    Parameters:
    -----------
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

    cell_types_dict: dict {str: list()}
      Dictionary with cell types as keys and list of cell-type defining markers
      as values.

    cell_name: str or None
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

    cell_phntp_comb: array(str)
      1-D numpy array of strings with phenotype found for each cell using markers
      from associated 'markers_representative' tuple.

    best_phntp_comb: array(str)
      1-D numpy array of strings with representative phenotypes among all possible
      phenotypes from 'markers_representative'.

    Returns:
    --------
      new_labels: array(str)
        1-D numpy array of strings with new names for each cell of 'mat_representative'.
    """

    # Trim down cell classification to remove any marker that is not present
    # in 'markers_representative'
    logging.info('\t\t\t\t\tTrimming cell classification to keep only relevant markers')
    markers_representative = np.asarray(sorted(markers_representative))
    cell_types_filtered = {}
    for cell_type, cell_mkers in cell_types_dict.items():
        # Separate and protein marker and sign
        mkers_list = [mkers[:-1] for mkers in cell_mkers]
        signs_list = [mkers[-1] for mkers in cell_mkers]
        # Test if markers belong to 'markers_representative' and rebuild list
        mkers_list_clean = [f'{mker}{sign}'
                            for mker, sign in zip(mkers_list, signs_list)
                            if mker in markers_representative]
        # If 'mkers_list_clean' is not empty, keep it
        if mkers_list_clean:
            cell_types_filtered[cell_type] = mkers_list_clean
        else:
            cell_types_filtered[cell_type] = []

    # Reduce 'cell_types_filtered' redundancy
    logging.info('\t\t\t\t\tReducing redundancy in new classification')
    nb_of_prot = np.array([len(mkers) for mkers in cell_types_dict.values()])
    cell_types_clean = {}
    for cell_type, cell_mkers in cell_types_filtered.items():
        # Find which cell types have identical markers
        comparison = np.array([[(j == cell_mkers) * 1, i]
                               for i, j in cell_types_filtered.items()])
        condi = comparison[:, 0].astype(bool)
        # 'cell_mkers' is either unique or current minimum
        if ((np.sum(condi) == 1)
                or (comparison[condi, 1][np.argmin(nb_of_prot[condi])] == cell_type)):
            cell_types_clean[cell_type] = cell_mkers

    # Remove keys with empty lists of markers and sort marker lists
    cell_types_clean = {k: sorted(v)
                        for k, v in cell_types_clean.items() if v}
    # Note: it does not matter if cell_types_clean is empty

    # Determine number of exact matches between phenotypes from 'best_phntp'
    # and marker lists from 'cell_types_clean'
    logging.info('\t\t\t\t\tDetermining exact matches between phenotypes and cell classification')
    phntp_match = []
    cell_types_match = []
    for phntp in np.char.split(best_phntp, sep='/'):
        if not cell_types_clean:
            # If 'cell_types_clean' is empty, there can be no exact match
            break
        if phntp in cell_types_clean.values():
            phntp_match.append('/'.join(phntp))  # Get matching phntp
            cell_type = [k for k, v in cell_types_clean.items() if v == phntp]
            cell_types_match.append(cell_type[0])  # Get matching cell type

    # Determine base name(s) for all phenotypes
    if len(phntp_match) == 0:  # No exact match
        # Most present phenotype (i.e. phenotype present in highest number
        # of cells) will be used as base name
        uniq_phntp, phntp_count = np.unique(cell_phntp[np.isin(cell_phntp, best_phntp)],
                                            return_counts=True)
        # Note: only representative phenotypes need to be considered, hence
        # filtering with 'best_phntp'
        base_comb = uniq_phntp[np.argmax(phntp_count)]
        base_name = cell_name
        logging.info('\t\t\t\t\t\tNo exact match between phenotypes and cell classification')

    elif len(phntp_match) == 1:  # One exact match that was already determined
        base_comb = phntp_match[0]
        base_name = cell_types_match[0]
        logging.info('\t\t\t\t\t\t1 exact match between phenotypes and cell classification')

    else:  # Several exact matches that were already determined
        base_comb = phntp_match
        base_name = cell_types_match
        logging.info(f'\t\t\t\t\t{len(base_name)} exact match between phenotypes and cell classification')

    # Get mapping dictionary to convert names
    logging.info('\t\t\t\t\tBuilding dictionary to convert cell type names')
    names_conv = find_name_difference(
        base_comb=base_comb,
        base_name=base_name,
        best_phntp=best_phntp)

    # Convert phenotypes to new names
    logging.info('\t\t\t\t\tRenaming cell types')
    new_labels = np.vectorize(names_conv.get)(cell_phntp, f'Other {cell_name}')
    # Note: with dict.get method, non-representative phenotypes (missing from
    # 'names_conv') are automatically converted to 'Other {cell_name}'

    return new_labels
