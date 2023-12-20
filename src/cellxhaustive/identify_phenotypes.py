"""
AT. Add general description here.
"""


# Imports utility modules
import itertools as ite
import numpy as np

# Imports ML modules
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

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
                        min_cellxsample=10, percent_samplesxbatch=0.5,
                        knn_refine=True, knn_min_probability=0.5,
                        cell_name=None):
    """
    Pipeline for automated gating, feature selection, and clustering to
    generate new annotations.

    Parameters
    ----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    markers: array(str)
      1-D numpy array with markers matching each column of 'mat'.

    batches: array(str)
      1-D numpy array with batch names of each cell of 'mat'.
      the thresholds for the new annotations.

    samples: array(str)
      1-D numpy array with sample names of each cell of 'mat'.
      the thresholds for the new annotations.

    is_label: array(bool)
      1-D numpy array with booleans to indicate cells matching current cell type.

    max_markers: int (default=15)
      Maximum number of relevant markers to select among the total list of
      markers from the markers array. Must be less than or equal to 'len(markers)'.

    min_annotations: int (default=3)
      Minimum number of markers used to define a cell population. Must be in
      [2; len(markers)], but it is advised to choose a value in [3; len(markers) - 1].

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'percent_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      10 cells/sample in at least 50% of the samples (see description of next
      parameter) within a batch to be considered.

    percent_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      'min_cellxsample' cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of previous parameter) in at least 50% of
      the samples within a batch to be considered.

    bimodality_selection_method: str (default = 'midpoint')
      Two possible methods: 'DBSCAN', which uses the clustering method with the
      same name, and 'midpoint', which uses markers closer to the normalized matrix
      # AT. Remove description if we decide to remove DBSCAN

    knn_refine: bool (default=False)
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

        # if bimodality_selection_method.upper() == 'DBSCAN':

        #     # Subset expression matrix to cells of the current batch only
        #     mat_subset = mat[is_batch, :]

        #     eps_marker_clustering = np.sqrt((pairwise_distances(mat_subset.transpose(), metric='euclidean') ** 2) / float(np.sum(is_batch)))
        #     # Notes:
        #     # - Pairwise distances are computed between rows
        #     # - Matrix is transposed to have scores per marker and not cell
        #     eps_marker_clustering = np.quantile(eps_marker_clustering[np.triu_indices(np.shape(eps_marker_clustering)[0], k=1)], q=0.05)
        #     # Notes:
        #     # - k=1 offsets the diagonal by 1 on the right, not to take the
        #     # 0-diagonal into account
        #     # - np.triu_indices() returns two arrays, one containing
        #     # x-coordinates, the other containing y-coordinates
        #     # Overall, this selects the 5th quantile of the upper-right triangle
        #     # of the transposed expression matrix of the whole batch

        #     # Create boolean array to select cells matching current label and batch
        #     is_label_batch = is_label[is_batch]

        #     # Subset expression matrix to cells of current batch and label only
        #     mat_subset = mat_subset[is_label_batch, :]

        #     # AT. Only reviewed until here

        #     # Transpose matrix to perform marker clustering
        #     X = mat_subset.transpose()

        #     # Calculate normalized pairwise distance
        #     eps = np.sqrt((pairwise_distances(X, metric='euclidean') ** 2) / float(np.shape(X)[1]))

        #     # Run DBSCAN
        #     km = DBSCAN(min_samples=1, eps=eps_marker_clustering, metric='precomputed', leaf_size=2).fit(eps)
        #     labels_ = km.labels_

        #     # Find the cluster center and generate an array of cluster by cell.
        #     cluster_centers_ = np.zeros((len(np.unique(labels_)), np.shape(mat_subset)[0]))
        #     for i in range(len(np.unique(labels_))):
        #         cluster_centers_[i, :] = np.mean(mat_subset[:, labels_ == i], axis=1)

        #     # Generate marker expression per cluster matrix
        #     mark_exprr = np.zeros((len(np.unique(labels_)), len(markers)))
        #     for idx, i in enumerate(labels_):
        #         mark_exprr[i, idx] += 1

        #     # Remove known invariants and associated clustered markers
        #     known_invariants = markers[np.sum(mark_exprr[np.any(mark_exprr[:, np.isin(markers, marker_order)], axis=1), :], axis=0) > 0]

        #     # Remove invariant markers based on the known invariants
        #     vt = VarianceThreshold().fit(mat_subset)
        #     invariants = markers[vt.variances_ <= np.max(vt.variances_[np.isin(markers, marker_order)])]
        #     invariants = np.unique(list(invariants) + list(known_invariants))

        #     # For the variable markers, find closer to the cluster centroids. Slice matrix subsequently
        #     markers_rep = []
        #     for idx, i in enumerate(np.sum(mark_exprr, axis=1)):
        #         m_step = markers[mark_exprr[idx, :] > 0]
        #         if i == 1:
        #             markers_rep += [m_step[0]]
        #         else:
        #             closest, _ = pairwise_distances_argmin_min(cluster_centers_[idx, :][np.newaxis, :],
        #                                                        X[np.isin(markers, m_step), :])
        #             markers_rep += [markers[np.isin(markers, m_step)][closest][0]]

        #     markers_rep = np.asarray(markers_rep)[np.isin(markers_rep, invariants) == False]
        #     markers_rep = markers[np.isin(markers, markers_rep)]

        #     # Generate dictionary for relevant markers
        #     markers_rep_dict = dict()
        #     for i in markers_rep:
        #         markers_rep_dict[i] = markers[mark_exprr[labels_[markers == i][0], :] > 0]

        # else:

        # Create boolean array to select cells matching current label and batch
        is_label_batch = np.logical_and(is_batch, is_label)

        # Subset expression matrix to cells of current batch and label only
        mat_subset = mat[is_label_batch, :]

        # Check bimodality of the markers and select the best ones
        marker_center_values = -np.abs(np.mean(mat_subset, axis=0) - 3)
        marker_midpoint_value = np.sort(marker_center_values)[::-1][max_midpoint_preselection]  # Select the max_midpoint_preselection-th center value in descending order
        # AT. max_midpoint_preselection: it looks a bit weird to always select the 15th value to test for max. Is it on purpose? Or should take something like the median value? Or something else?
        # AT. Change name of variable max_midpoint_preselection?
            # AT. Change max_midpoint_preselection for max_markers??
        # AT. Change name of variable marker_midpoint_value?
        is_center_greater = (marker_center_values > marker_midpoint_value)
        # markers_rep = markers[np.isin(markers, markers[is_center_greater])]  # AT. Use shorter version below?
        markers_rep = markers[is_center_greater]  # Select markers with center higher than max_midpoint_preselection-th center

        # Remove invariant markers based on the known invariants
        vt = VarianceThreshold().fit(mat_subset)
        # AT. We could use a threshold directly in VarianceThreshold()



        invariants = markers[vt.variances_ <= np.max(vt.variances_[np.isin(markers, marker_order)])]
        # invariants = markers[vt.variances_ <= np.max(vt.variances_[np.isin(markers, markers_rep)])]
        # AT. Problem here with marker_order missing because it was used during main gating
            # AT. Use markers_rep instead?
        # AT. If selecting variances smaller than the max variance, it will always select everything...

        markers_rep = np.asarray(markers_rep)[np.isin(markers_rep, invariants) == False]

        markers_rep = markers[np.isin(markers, markers_rep)]




        # Generate dictionary for relevant markers
        markers_rep_dict = dict([(mark_rep, mark_rep) for mark_rep in markers_rep])

        # Store dictionary for every batch
        try:  # To avoid problem if dict doesn't exist yet
            markers_rep_batches[batch] = markers_rep_dict
        except NameError:
            markers_rep_batches = dict()
            markers_rep_batches[batch] = markers_rep_dict
        # AT. Check usage of this dict



    # Select relevant markers shared across batches and slice data accordingly
    # AT. Check part about data slicing

    # Filter markers with appearences in all batches
    # mark_uniq, mark_uniq_count = np.unique(list(ite.chain.from_iterable(
    #     [list(markers_rep_batches[i].keys()) for idx, i in enumerate(markers_rep_batches.keys())])),
    #     return_counts=True)

    # Extract markers present in all batches
    mark_uniq, mark_uniq_count = np.unique(list(
        ite.chain.from_iterable(
            [list(markers_rep_batches[i].keys()) for i in markers_rep_batches.keys()]
        )
    ),
        return_counts=True)

    markers_rep_all = mark_uniq[mark_uniq_count == len(markers_rep_batches.keys())]

    # Merge all batches together and extract main cell type
    # Merge all batches together and extract main cell type
    # AT. Select samples for this population (decided in is_label)
    # AT. It
    mat_subset_label = mat[is_label, :]
    batches_label = batches[is_label]
    samples_label = samples[is_label]

    # Slice matrix and markers using the selected markers through the main gating
    # AT. Matrix with relevant markers only (15 markers)
    mat_representative = mat_subset_label[:, np.isin(markers, markers_rep_all)]
    markers_rep_all = markers[np.isin(markers, markers_rep_all)]

    # STUDY SUBSETS OF MARKERS: Go over every combination of markers and
    # understand the resulting number of cell types and unidentified cells
    x_p = np.linspace(0, 1, 101)
    y_ns = np.arange(101)

    markers_rep_all_ = check_all_subsets(max_markers=max_markers,
                                                x_p=x_p,
                                                y_ns=y_ns,
                                                mat_subset=mat_subset_label,
                                                markers=markers,
                                                markers_representative=markers_rep_all,
                                                marker_order=marker_order,
                                                batches=batches_label,
                                                samples=samples_label,
                                                min_cellxsample=min_cellxsample,
                                                percent_samplesxbatch=percent_samplesxbatch,
                                                min_cells=min_annotations)

    if len(markers_rep_all_) > 0:
        markers_rep_all = markers[np.isin(markers, markers_rep_all_)]
        mat_representative = mat_subset_label[:, np.isin(markers, markers_rep_all_)]
        # AT. Redundant?

        # Now let's figure out which groups of markers form relevant cell types
        # AT. Assign the annotations to the cells using the best set of markers defined before
        cell_groups, cell_groups_name, clustering_labels, mat_average, markers_average = cell_subdivision(
            # AT. CHANGE THIS IN FUNCTION
            mat=mat_subset_label,
            mat_representative=mat_representative,
            markers=markers,
            markers_representative=markers_rep_all,
            marker_order=marker_order,
            batches=batches_label,
            samples=samples_label,
            min_cellxsample=min_cellxsample,
            percent_samplesxbatch=percent_samplesxbatch,
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
