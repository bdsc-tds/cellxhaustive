"""
# Function that identifies the cell type of
AT. Add general description here.
"""


# Import utility modules
import numpy as np


# Import local functions
from assign_cell_types import assign_cell_types  # AT. Double-check path
from check_all_combinations import check_all_combinations  # AT. Double-check path
from knn_classifier import knn_classifier  # AT. Double-check path
# from cellxhaustive.assign_cell_types import assign_cell_types  # AT. Double-check path
# from cellxhaustive.check_all_combinations import check_all_combinations
# from cellxhaustive.knn_classifier import knn_classifier


# Function used in cellxhaustive.py
def identify_phenotypes(mat, batches, samples, markers, is_label, cell_types_dict,
                        cell_name, three_peak_markers=[],
                        max_markers=15, min_annotations=3,
                        min_samplesxbatch=0.5, min_cellxsample=10,
                        knn_refine=True, knn_min_probability=0.5):
    """
    Pipeline for automated gating, feature selection, and clustering to
    generate new annotations.
    # AT. Update description

    Parameters
    ----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    batches: array(str)
      1-D numpy array with batch names of each cell of 'mat'.

    samples: array(str)
      1-D numpy array with sample names of each cell of 'mat'.

    markers: array(str)
      1-D numpy array with markers matching each column of 'mat'.

    is_label: array(bool)
      1-D numpy array with booleans to indicate cells matching current cell type.

    cell_types_dict: {str: list()}
      Dictionary with cell types as keys and list of cell-type defining markers
      as values.

    cell_name: str or None
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    max_markers: int (default=15)
      Maximum number of relevant markers to select among total list of markers
      from total markers array. Must be less than or equal to 'len(markers)'.

    min_annotations: int (default=3)
      Minimum number of phenotypes for a combination of markers to be taken into
      account as a potential cell population. Must be in '[2; len(markers)]',
      but it is advised to choose a value in '[3; len(markers) - 1]'.

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least 'min_cellxsample'
      cells for a new annotation to be considered. In other words, by default, an
      annotation needs to be assigned to at least 10 cells/sample (see description
      of next parameter) in at least 50% of samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of samples
      within each batch for a new annotation to be considered. In other words, by
      default, an annotation needs to be assigned to at least 10 cells/sample in at
      least 50% of samples (see description of previous parameter) within a batch
      to be considered.

    knn_refine: bool (default=True)
      If True, clustering done via permutations of relevant markers will be
      refined using a KNN-classifier.

    knn_min_probability: float (default=0.5)
      Confidence threshold for KNN-classifier to reassign a new cell type
      to previously undefined cells.

    Returns:
    --------
      # AT. Add what is returned by the function
      new_labels
      cell_phntp_comb
    """

    # Main gating: select relevant markers for cell population 'label' in each batch

    # Perform gating for every batch independently
    # AT. Do we need it? Multithread/process here?
    for batch in np.unique(batches):

        # Create boolean array to select cells matching current 'batch'
        is_batch = (batches == batch)

        # Create boolean array to select cells matching current 'label' and 'batch'
        is_label_batch = np.logical_and(is_batch, is_label)

        # Subset expression matrix to cells of current 'batch' and 'label' only
        mat_subset = mat[is_label_batch, :]

        # Check bimodality of markers and select best ones
        marker_center_values = -np.abs(np.mean(mat_subset, axis=0) - 3)
        marker_threshold_value = np.sort(marker_center_values)[::-1][max_markers]  # Select max_markers-th center value (in descending order)
        is_center_greater = (marker_center_values > marker_threshold_value)
        markers_rep = markers[is_center_greater]  # Select markers with center higher than max_markers-th center

        # Store list of relevant markers for every batch
        # Note: try/except avoids an error if dict doesn't exist yet
        try:
            markers_rep_batches[batch] = list(markers_rep)
        except NameError:
            markers_rep_batches = {}
            markers_rep_batches[batch] = list(markers_rep)

    # Marker selection and matrix slicing: select relevant markers shared across
    # all batches and extract related data

    # Extract markers present in all batches
    markers_rep_all = set.intersection(*map(set, markers_rep_batches.values()))
    markers_rep_all = np.array(list(markers_rep_all))  # Converts format back to array

    # Extract expression, batch and sample information across all batches for
    # cell population 'label'
    mat_subset_label = mat[is_label, :]
    batches_label = batches[is_label]
    samples_label = samples[is_label]

    # Slice matrix to keep only expression of relevant markers
    markers_rep_all = markers[np.isin(markers, markers_rep_all)]  # Reorders markers
    mat_subset_rep_markers = mat_subset_label[:, np.isin(markers, markers_rep_all)]

    # Evaluate combinations of markers: go over every combination and find all
    # possible best combinations, phenotypes of all 'mat_subset_rep_markers'
    # cells and among those, phenotypes passing various thresholds (see function
    # definition for more information on those)
    nb_solution, best_marker_comb, cell_phntp_comb, best_phntp_comb = check_all_combinations(
        mat_representative=mat_subset_rep_markers,
        batches_label=batches_label,
        samples_label=samples_label,
        markers_representative=markers_rep_all,
        three_peak_markers=three_peak_markers,
        max_markers=max_markers,
        min_annotations=min_annotations,
        min_samplesxbatch=min_samplesxbatch,
        min_cellxsample=min_cellxsample)

    if nb_solution == 0:
        # 'best_marker_comb' is empty, which means that no marker combination
        # was found to properly represent cell type 'label' (from annotate()
        # function), so keep original annotation
        new_labels = np.full(cell_phntp_comb.shape, f'Other {cell_name}')
        results_dict = {'new_labels': new_labels,
                        'cell_phntp_comb': cell_phntp_comb}

    elif nb_solution == 1:
        # Slice matrix to keep only expression of best combination
        markers_rep_comb = markers[np.isin(markers, best_marker_comb)]
        mat_subset_rep_markers_comb = mat_subset_label[:, np.isin(markers, best_marker_comb)]

        # Assign cell type using only markers from 'best_marker_comb'
        new_labels = assign_cell_types(
            mat_representative=mat_subset_rep_markers_comb,
            batches_label=batches_label,
            samples_label=samples_label,
            markers_representative=markers_rep_comb,
            cell_types_dict=cell_types_dict,
            cell_name=cell_name,
            cell_phntp=cell_phntp_comb,
            best_phntp=best_phntp_comb)

        results_dict = {'new_labels': new_labels,
                        'cell_phntp_comb': cell_phntp_comb}

        # Check if conditions to run KNN-classifier are fulfilled
        is_undef = (new_labels == f'Other {cell_name}')  # Get number of undefined cells
        if (knn_refine  # Decided by user
                and ((np.sum(is_undef) > 1))  # At least 2 undefined cells
                and (len(np.unique(new_labels)) > 2)):  # At least 2 cell types different from 'Other'
            # If so, run it
            reannotated_labels, reannotation_proba = knn_classifier(
                mat_representative=mat_subset_rep_markers_comb,
                new_labels=new_labels,
                is_undef=is_undef,
                knn_min_probability=knn_min_probability)

            results_dict = {'reannotated_labels': reannotated_labels,
                            'reannotation_proba': reannotation_proba}

    else:  # Several solutions
        pass
        # AT. Do for loop to try and select best
        # AT. Use a parameter in argparse for the maximum number of solution to evaluate





    return results_dict


# nb_solution = float
# best_marker_comb = tuple or list of tuples
# cell_phntp_comb = array or list of arrays
# best_phntp_comb = array or list of arrays

# What do we do when there are several combinations?


# AT. Problem with 3 peaks markers --> How to deal with them in major_cell_type dict?
# AT. Checked presence of null variables until here


    """
    AT. In assign_cell_types(), in case of Array vs list of arrays, adapt definitions of:
    - cell_phntp_comb
    - best_phntp_comb
    - new_labels

    best_phntp --> array(str)
    phntp_match --> list(str)
    cell_types_clean --> key (str), val (list)
    base_name --> str
    base_comb --> str
    Change to else (instead of elif/else) and adapt assign_cell_types to deal with lists?
    """
