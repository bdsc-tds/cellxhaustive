"""
Function that identifies most probable cell type and phenotype for a group of cells
using expression of its most relevant markers.
"""


# Import utility modules
import numpy as np
from collections import defaultdict


# Import local functions
from assign_cell_types import assign_cell_types  # AT. Double-check path
from check_all_combinations import check_all_combinations  # AT. Double-check path
from knn_classifier import knn_classifier  # AT. Double-check path
# from cellxhaustive.assign_cell_types import assign_cell_types  # AT. Double-check path
# from cellxhaustive.check_all_combinations import check_all_combinations  # AT. Double-check path
# from cellxhaustive.knn_classifier import knn_classifier  # AT. Double-check path


# Function used in cellxhaustive.py
def identify_phenotypes(mat, batches, samples, markers, is_label, cell_types_dict,
                        cell_name, three_peak_markers=[],
                        max_markers=15, min_annotations=3,
                        min_samplesxbatch=0.5, min_cellxsample=10,
                        knn_refine=True, knn_min_probability=0.5):
    """
    Function that identifies most probable cell type and phenotype for a group of
    cells using expression of its most relevant markers.

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
    results_dict: dict {str: list(array(str, float, np.nan))}
      Dictionary with 2 mandatory keys and 2 optional keys:
        - 'new_labels' (mandatory): list of 1-D numpy arrays with cell type for each
          cell of 'mat[is_label]'. 1 array corresponds to 1 optimal marker combination.
        - 'cell_phntp_comb' (mandatory): list of 1-D numpy arrays with full phenotype
          for each cell of 'mat[is_label]'. 1 array corresponds to 1 optimal marker
          combination.
        - 'reannotated_labels' (optional): list of 1-D numpy arrays with cell type
          for each cell of 'mat[is_label]'. 1 array corresponds to 1 optimal marker
          combination. Undefined cell types were reannotated using a KNN-classifier.
          Available only if 'knn_refine=True'.
        - 'reannotation_proba' (optional): list of 1-D numpy arrays with reannotation
          prediction probability determined by a KNN-classifier for each undefined cell
          of 'mat[is_label]'. 1 array corresponds to 1 optimal marker combination.
          Available only if 'knn_refine=True'.
    """

    # Main gating: select relevant markers for cell population 'label' in each batch
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

    # Initialise result dictionary with empty lists
    # Note: even if lists end up with only 1 element, it makes processing results
    # in cellxhaustive.py easier
    results_dict = defaultdict(list)

    if nb_solution == 0:
        # 'best_marker_comb' is empty, which means that no marker combination
        # was found to properly represent cell type 'label' (from annotate()
        # function), so keep original annotation
        new_labels = np.full(cell_phntp_comb.shape, f'Other {cell_name}')

        # Append results to dictionary
        results_dict['new_labels'].append(new_labels)
        results_dict['cell_phntp_comb'].append(cell_phntp_comb)

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

        # Append results to dictionary
        results_dict['new_labels'].append(new_labels)
        results_dict['cell_phntp_comb'].append(cell_phntp_comb)

        if knn_refine:
            # Reannotate only if conditions to run KNN-classifier are met
            is_undef = (new_labels == f'Other {cell_name}')  # Get number of undefined cells

            # At least 2 undefined cells and 2 cell types different from 'Other'
            if ((np.sum(is_undef) > 1) and (len(np.unique(new_labels)) > 2)):
                # Reannotate cells
                reannotated_labels, reannotation_proba = knn_classifier(
                    mat_representative=mat_subset_rep_markers_comb,
                    new_labels=new_labels,
                    is_undef=is_undef,
                    knn_min_probability=knn_min_probability)

                # Append results to dictionary
                results_dict['reannotated_labels'].append(reannotated_labels)
                results_dict['reannotation_proba'].append(reannotation_proba)

            else:  # If conditions are not met, no reannotation
                # Use default arrays as placeholders for reannotation results
                reannotated_labels = np.full(new_labels.shape[0], 'No_reannotation')
                reannotation_proba = np.full(new_labels.shape[0], np.nan)
                results_dict['reannotated_labels'].append(reannotated_labels)
                results_dict['reannotation_proba'].append(reannotation_proba)

    else:  # Several solutions

        # Initialise counter of undefined cells
        undef_counter = []

        for i in range(nb_solution):
            # Slice matrix to keep only expression of best combination
            markers_rep_comb = markers[np.isin(markers, best_marker_comb[i])]
            mat_subset_rep_markers_comb = mat_subset_label[:, np.isin(markers, best_marker_comb[i])]

            # Assign cell type using only markers from 'best_marker_comb[i]'
            new_labels = assign_cell_types(
                mat_representative=mat_subset_rep_markers_comb,
                batches_label=batches_label,
                samples_label=samples_label,
                markers_representative=markers_rep_comb,
                cell_types_dict=cell_types_dict,
                cell_name=cell_name,
                cell_phntp=cell_phntp_comb[i],
                best_phntp=best_phntp_comb[i])

            # Append results to dictionary
            results_dict['new_labels'].append(new_labels)
            results_dict['cell_phntp_comb'].append(cell_phntp_comb[i])

            if knn_refine:
                # Reannotate only if conditions to run KNN-classifier are met
                is_undef = (new_labels == f'Other {cell_name}')  # Get number of undefined cells

                # At least 2 undefined cells and 2 cell types different from 'Other'
                if ((np.sum(is_undef) > 1) and (len(np.unique(new_labels)) > 2)):
                    # Reannotate cells
                    reannotated_labels, reannotation_proba = knn_classifier(
                        mat_representative=mat_subset_rep_markers_comb,
                        new_labels=new_labels,
                        is_undef=is_undef,
                        knn_min_probability=knn_min_probability)

                    # Append results to dictionary
                    results_dict['reannotated_labels'].append(reannotated_labels)
                    results_dict['reannotation_proba'].append(reannotation_proba)

                    # Record number of undefined cells after reannotation
                    nb_undef = np.sum(reannotated_labels == f'Other {cell_name}')
                    undef_counter.append(nb_undef)

                else:  # If conditions are not met, no reannotation
                    # Use default arrays as placeholders for reannotation results
                    reannotated_labels = np.full(new_labels.shape[0], 'No_reannotation')
                    reannotation_proba = np.full(new_labels.shape[0], np.nan)
                    results_dict['reannotated_labels'].append(reannotated_labels)
                    results_dict['reannotation_proba'].append(reannotation_proba)

                    # Record number of undefined cells after reannotation
                    nb_undef = np.sum(is_undef)
                    undef_counter.append(nb_undef)

        # Get index of undefined cells minimum
        min_undef_idx = [i for i, x in enumerate(undef_counter) if x == min(undef_counter)]

        # Filter results using previous indices
        for val in results_dict.values():
            val = [i for idx, i in enumerate(val) if idx in set(min_undef_idx)]
            # Note: set() is used to increase speed

    return results_dict
