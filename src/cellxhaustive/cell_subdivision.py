"""
AT. Add general description here.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports local functions
from find_set_differences import find_set_differences  # AT. Double-check path
from select_cells import select_cells  # AT. Double-check path
# from cellxhaustive.find_set_differences import find_set_differences
# from cellxhaustive.select_cells import select_cells


# Permute across positive and negative expression of the relevant markers, and
# identify new cell types
# AT. Check presence/absence of all parameters/variable
def cell_subdivision(mat, mat_representative,
                     markers, markers_representative, marker_order,
                     batches, samples,
                     min_cellxsample=10, min_samplesxbatch=0.4,
                     three_peak_markers=[], cell_name=None):
    # AT. min_samplesxbatch is different from the usual default value. Is it on purpose
    """
    Cell line subdivision.
    # AT. Add function description (use the one before?)
    # We need the representative matrix but also the full matrix with the proper ontology
    # Take cell-naming outside from this function

    Parameters:
    -----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    mat_representative: array(float)
      2-D numpy array expression matrix of the representative markers, with
      cells in D0 and markers in D1. In other words, rows contain cells and
      columns contain markers.

    markers: array(str)
      1-D numpy array with markers matching each column of mat.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of mat_representative.



    marker_order: list(str)
      List of markers used in the gating strategy ordered accordingly.
      # AT. Check parameter description (and default?). Do we even keep it? Or set it as an option?



    batches: array(str)
      1-D numpy array with batch names of each cell of mat.

    samples: array(str)
      1-D numpy array with sample names of each cell of mat.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in min_samplesxbatch % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      10 cells/sample in at least 50% of the samples (see description of next
      parameter) within a batch to be considered.

    min_samplesxbatch: float (default=0.4)
      Minimum proportion of samples within each batch with at least
      min_cellxsample cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of previous parameter) in at least 50% of
      the samples within a batch to be considered.



    three_peak_markers: list(str) (default=[])
      List of markers with potentially three peaks.
      # AT. Improve description? Do we even keep it? Or set it as an option?

    cell_name: str or None (default=None)
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').
      # AT. None is automatically converted to str and always appears in f-string

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Trim down the cell classification to remove any protein that is not
    # present (major cell types to renaming)
    # AT. Remove cell ontologies that have no markers we can use
    # AT. Take cell type with the fewer number of markers (example with 2 cell type with same markers +1 that is missing)
    # AT. Add cell ontology names here ?

    major_cell_types_ = dict()
    for i in major_cell_types:
        major_cell_types_[i] = dict()
        for j in major_cell_types[i]:
            major_cell_types_[i][j] = list(np.asarray(major_cell_types[i][j])[np.isin(major_cell_types[i][j], markers)])

    # Clean up redundancies in major_cell_types_
    number_of_proteins = np.array([len(v['positive']) + len(v['negative'])
                                   for v in major_cell_types.values()])
    major_cell_types_clean = dict()
    for k, v in major_cell_types_.items():
        comparison = np.array([[(j == v) * 1, i]
                               for i, j in major_cell_types_.items()])
        condi = comparison[:, 0].astype(bool)
        if np.sum(condi) == 1:
            major_cell_types_clean[k] = v
        elif comparison[condi, 1][np.argmin(number_of_proteins[condi])] == k:
            major_cell_types_clean[k] = v
        else:
            continue

    number_of_proteins = {k: (len(v['positive']) + len(v['negative']))
                          for k, v in major_cell_types.items()}
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
    # AT. This only names with positive markers, which is why we need to add the empty tuple for the case with only negative markers

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
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_] == x) > min_cellxsample for x in np.unique(samples_)])
            keep_cell_type_ = np.sum(keep_cell_type_, axis=0) / float(len(np.unique(samples_)))
            keep_cell_type = keep_cell_type and keep_cell_type_ > min_samplesxbatch

        if keep_cell_type:
            # To store it, let's find a name for it
            mat_avg = np.mean(mat[cells, :], axis=0)
            positives_ = markers[mat_avg >= 3]
            negatives_ = markers[mat_avg < 3]

            condi = [np.all(np.isin(x['negative'], list(negatives_)))
                     and np.all(np.isin(x['positive'], list(positives_)))
                     for x in major_cell_types_clean.values()]
            potential_name = np.asarray(list(major_cell_types_clean.keys()))[condi]

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
    mat_average = mat[:, np.isin(markers, marker_order + list(markers_representative))]
    markers_average = markers[np.isin(markers, marker_order + list(markers_representative))]

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

    return cell_groups_renaming, cell_groups_, clustering_labels, mat_average, markers_average
