"""
AT. Add general description here.
"""


# Imports utility modules
import numpy as np

# Imports local functions
from cell_subdivision import cell_subdivision  # AT. Double-check path
from check_all_subsets import check_all_subsets  # AT. Double-check path
from knn_classifier import knn_classifier  # AT. Double-check path
# from cellxhaustive.cell_subdivision import cell_subdivision
# from cellxhaustive.check_all_subsets import check_all_subsets
# from cellxhaustive.knn_classifier import knn_classifier


# Function to identify the cell type of
# AT. Update description
def identify_phenotypes(mat, markers, batches, samples, is_label,
                        max_markers=15, min_annotations=3,  # bimodality_selection_method='midpoint',  # AT. Remove parameter if we decide to remove DBSCAN
                        min_samplesxbatch=0.5, min_cellxsample=10,
                        knn_refine=True, knn_min_probability=0.5,
                        cell_name=None):
    """
    Pipeline for automated gating, feature selection, and clustering to
    generate new annotations.
    # AT. Update description

    Parameters
    ----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    markers: array(str)
      1-D numpy array with markers matching each column of 'mat'.

    batches: array(str)
      1-D numpy array with batch names of each cell of 'mat'.

    samples: array(str)
      1-D numpy array with sample names of each cell of 'mat'.

    is_label: array(bool)
      1-D numpy array with booleans to indicate cells matching current cell type.

    max_markers: int (default=15)
      Maximum number of relevant markers to select among the total list of
      markers from the markers array. Must be less than or equal to 'len(markers)'.

    min_annotations: int (default=3)
      Minimum number of markers used to define a cell population. Must be in
      [2; len(markers)], but it is advised to choose a value in [3; len(markers) - 1].

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      'min_cellxsample' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of previous parameter) in at least 50% of
      the samples within a batch to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      10 cells/sample in at least 50% of the samples (see description of next
      parameter) within a batch to be considered.

    knn_refine: bool (default=True)
      If True, the clustering done via permutations of relevant markers will be
      refined using a KNN classifier.

    knn_min_probability: float (default=0.5)
      Confidence threshold for the KNN classifier to reassign a new cell type
      to previously undefined cells.



    cell_name: str or None (default=None)
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').
      # AT. None is automatically converted to str and always appears in f-string

    Returns:
    --------
      # AT. Add what is returned by the function
      # AT. Double-check if we need everything
      With best set/Without best set:
        is_label:
        cell_groups_name/{-1: cell_name}:
        clustering_labels/np.zeros(np.sum(is_label)) - 1:
        markers_rep_batches:
        markers_rep_all/[]:
    """

    # Main gating: select relevant markers for cell population 'label' in each batch

    # Perform gating for every batch independently
    # AT. Might be a bit of an overkill. Do we need it?
    for batch in np.unique(batches):

        # Create boolean array to select cells matching current batch
        is_batch = (batches == batch)

        # Create boolean array to select cells matching current label and batch
        is_label_batch = np.logical_and(is_batch, is_label)

        # Subset expression matrix to cells of current batch and label only
        mat_subset = mat[is_label_batch, :]

        # Check bimodality of the markers and select the best ones
        marker_center_values = -np.abs(np.mean(mat_subset, axis=0) - 3)
        marker_threshold_value = np.sort(marker_center_values)[::-1][max_markers]  # Select max_markers-th center value (in descending order)
        is_center_greater = (marker_center_values > marker_threshold_value)
        markers_rep = markers[is_center_greater]  # Select markers with center higher than max_markers-th center

        # Store list of relevant markers for every batch
        try:  # To avoid problem if dict doesn't exist yet
            markers_rep_batches[batch] = list(markers_rep)
        except NameError:
            markers_rep_batches = dict()
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

    # Evaluate subsets of markers: go over every combination of markers and
    # calculate the resulting number of cell types and unidentified cells

    # AT. Remove these matrices??
    x_p = np.linspace(0, 1, 101)
    y_ns = np.arange(101)


# AT. WHICH MATRIX SHOULD WE USE IN check_all_subsets()???
# mat_subset_label OR mat_subset_rep_markers???


# AT. Update variable name
    markers_rep_all_ = check_all_subsets(mat_subset=mat_subset_label,
                                         # mat_representative=mat_subset_rep_markers,  # AT. The sliced matrix isn't used in the former code...
                                         batches=batches_label,
                                         samples=samples_label,
                                         markers=markers,
                                         markers_representative=markers_rep_all,
                                         max_markers=max_markers,
                                         min_annotations=min_annotations,
                                         x_p=x_p,
                                         y_ns=y_ns,
                                         min_cellxsample=min_cellxsample,
                                         percent_samplesxbatch=percent_samplesxbatch)









    if len(markers_rep_all_) > 0:
        markers_rep_all = markers[np.isin(markers, markers_rep_all_)]
        mat_subset_rep_markers = mat_subset_label[:, np.isin(markers, markers_rep_all_)]
        # AT. Redundant?

        # Now let's figure out which groups of markers form relevant cell types
        # AT. Assign the annotations to the cells using the best set of markers defined before
        cell_groups, cell_groups_name, clustering_labels, mat_average, markers_average = cell_subdivision(
            # AT. CHANGE THIS IN FUNCTION
            mat=mat_subset_label,
            mat_representative=mat_subset_rep_markers,
            markers=markers,
            markers_representative=markers_rep_all,
            marker_order=marker_order,
            batches=batches_label,
            samples=samples_label,
            min_cellxsample=min_cellxsample,
            min_samplesxbatch=min_samplesxbatch,
            three_markers=three_markers,
            cell_name=cell_name)
        # AT. In annotate(), only cell_groups_name and clustering_labels are used, so do we actually need to return the other elements?

        # Try to classify undefined cells using a KNN classifier
        if knn_refine:
            clustering_labels = knn_classifier(mat_representative=mat_subset_rep_markers,
                                               clustering_labels=clustering_labels,
                                               knn_min_probability=knn_min_probability)

        return is_label, cell_groups_name, clustering_labels, markers_rep_batches, markers_rep_all

    else:
        # AT. If there is no 'best set' and more specialized annotation, keep the parent one (more general)
        # AT. DIRTY!
        return is_label, {-1: cell_name}, np.zeros(np.sum(is_label)) - 1, markers_rep_batches, []

# AT. In annotate(), only cell_groups_name and clustering_labels are used, so do we actually need to return the other elements?
