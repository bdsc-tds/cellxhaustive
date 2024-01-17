"""
Function that searches for matches between combinations of markers and a list of
markers-defined cell types (i.e: cell type 1 is A+, B-, C-, cell type 2 is A-,
C+, D+...). If one or several match(es) is(are) found, marker combination(s) will
be assigned corresponding cell type(s) from list and other combinations will be
assigned names derived from exact match(es). If no match is found, combination
with most cells will be used as base name and other combinations will be assigned
names derived from this base name.
"""


# Imports utility modules
import numpy as np


# Imports local functions
from find_set_differences import find_set_differences  # AT. Double-check path
# from cellxhaustive.find_set_differences import find_set_differences


# AT. Check presence/absence of all parameters/variable
# Function used in identify_phenotypes.py  # AT. Update script name if needed
def assign_cell_types(mat_representative,
                      batches_label,
                      samples_label,
                      markers_representative,
                      cell_phntp,
                      best_phntp,
                      cell_types_dict,
                      cell_name=None):
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

    cell_phntp_comb: array or list(array)
      1-D numpy array or list of 1-D numpy arrays showing phenotype found for
      each cell using markers from associated 'markers_representative' tuple.

    best_phntp_comb: array or list(array)
      1-D numpy array or list of 1-D numpy arrays showing representative phenotypes
      among all possible phenotypes from 'markers_representative'.

    cell_types_dict: {str: list()}
      Dictionary with cell types as keys and list of cell-type defining markers
      as values.

    cell_name: str or None (default=None)
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').


















    cell_types_dict_ = dict()
    for i in cell_types_dict:
        cell_types_dict_[i] = dict()
        for j in cell_types_dict[i]:
            cell_types_dict_[i][j] = list(np.asarray(cell_types_dict[i][j])[np.isin(cell_types_dict[i][j], markers)])

    # Clean up redundancies in cell_types_dict_
    number_of_proteins = np.array([len(v['positive']) + len(v['negative'])
                                   for v in cell_types_dict.values()])
    cell_types_dict_clean = dict()
    for k, v in cell_types_dict_.items():
        comparison = np.array([[(j == v) * 1, i]
                               for i, j in cell_types_dict_.items()])
        condi = comparison[:, 0].astype(bool)
        if np.sum(condi) == 1:
            cell_types_dict_clean[k] = v
        elif comparison[condi, 1][np.argmin(number_of_proteins[condi])] == k:
            cell_types_dict_clean[k] = v
        else:
            continue

    number_of_proteins = {k: (len(v['positive']) + len(v['negative']))
                          for k, v in cell_types_dict.items()}
    # AT. Check if it is used after

    # Find different groups of positive cells
    # AT. CD4 has 3 peaks, so we split low and high positive. But right now it's the only one we have
    # AT. Sort this at the beginning to make it easier. When doing permutations, we need to take into account 3 peaks instead of 2!
    three_markers_ = np.asarray(three_peak_markers)[np.isin(three_peak_markers, markers_representative)]
    three_markers_low = [x + "_low" for x in three_markers_]
    three_peak_markers = list(three_markers_)
    markers_representative_ = list(markers_representative) + list(three_markers_low)
    groups = [tuple(np.unique(i)) for i in ite.combinations_with_replacement(
        markers_representative_, len(markers_representative_))]
    groups = [tuple([])] + groups
    # AT. The combinations only contain positive markers, which is why we need to add the empty tuple for the case with only negative markers

    # Start dictionaries to store results
    cell_groups_renaming = {}
    cell_groups = {}
    cell_groups_ = {}
    repeated_names = {}

    # An array to define the clustering groups
    clustering_labels = np.zeros(np.shape(mat_representative)[0]) - 1
    # AT. We start with all cells 'undefined'

    # Initial values
    gdx = 0
    cell_groups[-1] = f'other {cell_name}'
    cell_groups_[-1] = f'other {cell_name}'
    unidentified = 0  # Count number of undefined cells
    cell_groups_renaming[-1] = set([])

    # Loop over unique groups
    for i in set(groups):
        iteration = np.unique(i)
        if np.any(np.isin(three_markers_[np.isin(three_markers_low, iteration)], iteration)):
            continue
        else:
            # The groups are defined by positive markers
            positives = np.asarray(markers_representative)[np.isin(markers_representative, iteration)]
            # The negatives should be defined easily
            negatives = np.asarray(markers_representative)[np.isin(markers_representative, iteration, invert=True)]  # AT. Check it this works
            # negatives = np.asarray(markers_representative)[np.isin(markers_representative, iteration) == False]
            # The low positives are those markers that appear in three_markers_low
            lowpositives = np.asarray(three_markers_)[np.isin(three_markers_low, iteration)]

        # Figure out which cells fulfill these rules
        cells = select_cells(mat=mat_representative,
                             markers=markers_representative,
                             positive=positives,
                             negative=negatives,
                             lowpositive=lowpositives,
                             three_peak_markers=three_peak_markers)

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = True
        for b in np.unique(batches):
            cells_ = cells[batches == b]
            samples_ = samples[batches == b]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_] == x) >= min_cellxsample for x in np.unique(samples_)])
            keep_cell_type_ = np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_)))
            keep_cell_type = keep_cell_type and keep_cell_type_ >= min_samplesxbatch

        if keep_cell_type:
            # To store it, let's find a name for it
            mat_avg = np.mean(mat[cells, :], axis=0)
            positives_ = markers[mat_avg >= 3]
            negatives_ = markers[mat_avg < 3]

            condi = [np.all(np.isin(x['negative'], list(negatives_)))
                     and np.all(np.isin(x['positive'], list(positives_)))
                     for x in cell_types_dict_clean.values()]
            potential_name = np.asarray(list(cell_types_dict_clean.keys()))[condi]

            if len(potential_name) == 0:
                potential_name = cell_name
            elif len(potential_name) == 1:
                potential_name = potential_name[0]
            else:
                potential_name = potential_name[np.argmax(np.array([number_of_proteins[x] for x in potential_name]))]

            try:
                x = repeated_names[potential_name]
                repeated_names[potential_name] += 1
            except KeyError:
                repeated_names[potential_name] = 0

            x = repeated_names[potential_name]
            clustering_labels[cells] = int(gdx)
            if len(three_markers_) == 0:
                cell_groups_renaming[gdx] = set(list(map(lambda ls: ls + "+", positives)) + list(map(lambda ls: ls + "-", negatives)))
            else:
                cell_groups_renaming[gdx] = set(list(map(lambda ls: ls + "++", positives)) + list(map(lambda ls: ls + "-", negatives)) + list(map(lambda ls: ls + "+", lowpositives)))

            cell_groups_[gdx] = potential_name
            cell_groups[gdx] = f'{potential_name} {str(int(x))}'  # AT. Use _ instead of space?
            cell_groups[gdx] = f'{cell_groups[gdx]} ({np.sum(cells)} cells)'
            gdx += 1
        else:
            unidentified += np.sum(cells)

    # It is useful to generate a dictionary with the clustering index and the
    # different positive/negative sequence of markers
    cell_groups[-1] = f'{cell_groups[-1]} ({unidentified} cells)'
    # mat_average = mat[:, np.isin(markers, marker_order + list(markers_representative))]
    # markers_average = markers[np.isin(markers, marker_order + list(markers_representative))]

    for repeated_x in repeated_names:
        x = find_set_differences({k: cell_groups_renaming[k]
                                  for k, v in cell_groups_.items() if v == repeated_x},
                                 baseline_name='')
        for k in x.keys():
            cell_groups_[k] = (x[k] + " " + cell_groups_[k]).strip()

    cell_groups_renaming = find_set_differences(cell_groups_renaming)

    for x in cell_groups_renaming.keys():
        cell_groups_renaming[x] += " (" + cell_groups[x].split(" (")[1]
        cell_groups_[x] += " (" + cell_groups[x].split(" (")[1]

    if all(clustering_labels == -1):
        cell_groups_[-1] = cell_name

    return cell_groups_renaming, cell_groups_, clustering_labels
    # return cell_groups_renaming, cell_groups_, clustering_labels, mat_average, markers_average  # AT. What was returned before
