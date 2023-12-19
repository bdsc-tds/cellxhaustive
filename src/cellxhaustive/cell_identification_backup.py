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
from cellxhaustive.cell_subdivision import cell_subdivision
from cellxhaustive.check_all_subsets import check_all_subsets
from cellxhaustive.knn_classifier import knn_classifier
from cellxhaustive.main_gating import gaussian_gating


# Full pipeline
# AT- Check annotation
def cell_identification(mat, markers, marker_order, batches, samples, positive,
                        three_markers=[], min_cellxsample=10, percent_samplesxbatch=0.5,
                        max_midpoint_preselection=15, max_markers=15,
                        min_annotations=3, bimodality_selection_method='midpoint',
                        mat_raw=None, knn_refine=True, knn_min_probability=0.5,
                        cell_name='None', random_state=42):
    """
    Pipeline for automated gating, feature selection, and clustering to
    generate new annotations.

    Parameters:
    -----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    markers: array(str)
      1-D numpy array with markers matching each column of mat.



    marker_order: list(str)
      List of markers used in the gating strategy, ordered accordingly.



    batches: array(str)
      1-D numpy array with batch names of each cell of mat. Useful for defining
      the thresholds for the new annotations.

    samples: array(str)
      1-D numpy array with sample names of each cell of mat. Useful for defining
      the thresholds for the new annotations.



    positive: list(bool)
      List indicating whether the markers in marker_order are positively (True)
      or negatively (False) expressed.

    three_markers: list(str)
      List of markers with potentially three peaks.



    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in percent_samplesxbatch % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least
      10 cells/sample in at least 50% of the samples (see description of next
      parameter) within a batch to be considered.

    percent_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least
      min_cellxsample cells for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample (see description of previous parameter) in at least 50% of
      the samples within a batch to be considered.

    max_midpoint_preselection: int (default=15)
      Number of representative markers to select among the total list of
      markers from the markers array. Must be less than or equal to len(markers).



    max_markers: int or None (default=15)
      Maximum number of relevant markers selected.
      # AT. Merge with max_midpoint_preselection?

    min_annotations: int (default=3)
      Maximum number of relevant markers selected.
      # AT. Check if None is valid

    bimodality_selection_method: str (default = 'midpoint')
      Two possible methods: 'DBSCAN', which uses the clustering method with the
      same name, and 'midpoint', which uses markers closer to the normalized matrix

    mat_raw: ndarray or None (default=None)
      2-D array raw expression matrix.



    knn_refine: bool (default=False)
      If True, the clustering done via permutations of relevant markers will be
      refined using a KNN classifier.

    knn_min_probability: float (default=0.5)
      Confidence threshold for the KNN classifier to reassign a new cell type
      to previously undefined cells.

    cell_name: str or None (default=None)
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').
      # AT. None is automatically converted to str and always appears in f-string

    random_state: int or None (default=42)
      Random seed.

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # MAIN GATING: we perform the gating for every batch independently,
    # but that might be a bit of an overkill...
    # AT. Adapt if we don't perform gating independently

    markers_representative_batches = dict()
    is_label = np.zeros(np.shape(mat)[0]) != 0

    for kdx, k in enumerate(np.unique(batches)):
        # AT. kdx is not used, so remove it along with enumerate?
        batch = (batches == k)

        # if a raw matrix is provided, we use this for the main gating
        if mat_raw is None:
            is_label_b = gaussian_gating(mat[batch, :],
                                         markers,
                                         marker_order=marker_order,
                                         positive=positive,
                                         random_state=random_state
                                         )
        else:
            is_label_b = gaussian_gating(mat_raw[batch, :],
                                         markers,
                                         marker_order=marker_order,
                                         positive=positive,
                                         random_state=random_state
                                         )

        # Subset batch
        is_label[batch] = is_label_b
        mat_ = mat[batch, :]

        if bimodality_selection_method == 'DBSCAN':  # AT. upper/lower to avoid case problems?

            eps_marker_clustering = np.sqrt((pairwise_distances(mat_.transpose(), metric='euclidean') ** 2) / float(np.sum(batch)))
            eps_marker_clustering = np.quantile(eps_marker_clustering[np.triu_indices(np.shape(eps_marker_clustering)[0], k=1)], q=0.05)

            # Subset cell types
            mat_ = mat_[is_label_b, :]

            # Transpose matrix to perform marker clustering
            X = mat_.transpose()

            # Calculate normalized pairwise distance
            eps = np.sqrt((pairwise_distances(X, metric='euclidean') ** 2) / float(np.shape(X)[1]))

            # Calculate PCA space
            Xt = PCA(n_components=2).fit_transform(X)  # AT. Never used so double check this

            # Run DBSCAN
            km = DBSCAN(min_samples=1, eps=eps_marker_clustering, metric='precomputed', leaf_size=2).fit(eps)
            labels_ = km.labels_

            # Find the cluster center and generate an array of cluster by cell.
            cluster_centers_ = np.zeros((len(np.unique(labels_)), np.shape(mat_)[0]))
            for i in range(len(np.unique(labels_))):
                cluster_centers_[i, :] = np.mean(mat_[:, labels_ == i], axis=1)

            # Generate marker expression per cluster matrix
            mark_exprr = np.zeros((len(np.unique(labels_)), len(markers)))
            for idx, i in enumerate(labels_):
                mark_exprr[i, idx] += 1

            # Remove known invariants and associated clustered markers
            known_invariants = markers[np.sum(mark_exprr[np.any(mark_exprr[:, np.isin(markers, marker_order)], axis=1), :], axis=0) > 0]

            # Remove invariant markers based on the known invariants
            vt = VarianceThreshold().fit(mat_)
            invariants = markers[vt.variances_ <= np.max(vt.variances_[np.isin(markers, marker_order)])]
            invariants = np.unique(list(invariants) + list(known_invariants))

            # For the variable markers, find closer to the cluster centroids. Slice matrix subsequently
            markers_representative = []
            for idx, i in enumerate(np.sum(mark_exprr, axis=1)):
                m_step = markers[mark_exprr[idx, :] > 0]
                if i == 1:
                    markers_representative += [m_step[0]]
                else:
                    closest, _ = pairwise_distances_argmin_min(cluster_centers_[idx, :][np.newaxis, :],
                                                               X[np.isin(markers, m_step), :])
                    markers_representative += [markers[np.isin(markers, m_step)][closest][0]]

            markers_representative = np.asarray(markers_representative)[np.isin(markers_representative, invariants) == False]
            markers_representative = markers[np.isin(markers, markers_representative)]

            # Generate dictionary for relevant markers
            mdict = dict()
            for i in markers_representative:
                mdict[i] = markers[mark_exprr[labels_[markers == i][0], :] > 0]

            # Store dictionary for every batch
            markers_representative_batches[k] = mdict

        else:

            # Subset cell types
            mat_ = mat_[is_label_b, :]

            # Check bimodality of the markers selected
            center_values = -np.abs(np.mean(mat_, axis=0) - 3)
            max_values = np.sort(center_values)[::-1][max_midpoint_preselection]
            labels_ = center_values > max_values
            markers_representative = markers[np.isin(markers, markers[labels_])]

            # Remove invariant markers based on the known invariants
            vt = VarianceThreshold().fit(mat_)
            invariants = markers[vt.variances_ <= np.max(vt.variances_[np.isin(markers, marker_order)])]
            markers_representative = np.asarray(markers_representative)[np.isin(markers_representative, invariants) == False]
            markers_representative = markers[np.isin(markers, markers_representative)]

            # Generate dictionary for relevant markers
            mdict = dict([(i, i) for i in markers_representative])

            # Store dictionary for every batch
            markers_representative_batches[k] = mdict

    # SELECT RELEVANT MARKERS BASED ON SELECTIONS ACROSS BATCHES AND SLICE DATA

    # Filter markers with appearences in all batches
    m, c = np.unique(list(ite.chain.from_iterable(
        [list(markers_representative_batches[i].keys()) for idx, i in enumerate(markers_representative_batches.keys())])),
        return_counts=True)
    markers_representative = m[c == len(markers_representative_batches.keys())]

    # Merge all batches together and extract main cell type
    mat_ = mat[is_label, :]
    samples_ = samples[is_label]
    batches_ = batches[is_label]

    # Slice matrix and markers using the selected markers through the main gating
    mat_representative = mat_[:, np.isin(markers, markers_representative)]
    markers_representative = markers[np.isin(markers, markers_representative)]

    # STUDY SUBSETS OF MARKERS: Go over every combination of markers and
    # understand the resulting number of cell types and unidentified cells
    x_p = np.linspace(0, 1, 101)
    y_ns = np.arange(101)

    markers_representative_ = check_all_subsets(max_markers=max_markers,
                                                x_p=x_p,
                                                y_ns=y_ns,
                                                mat_=mat_,
                                                markers=markers,
                                                markers_representative=markers_representative,
                                                marker_order=marker_order,
                                                batches=batches_,
                                                samples=samples_,
                                                cell=cell_name,
                                                min_cellxsample=min_cellxsample,
                                                percent_samplesxbatch=percent_samplesxbatch,
                                                min_cells=min_annotations)

    if len(markers_representative_) > 0:
        markers_representative = markers[np.isin(markers, markers_representative_)]
        mat_representative = mat_[:, np.isin(markers, markers_representative_)]

        # Now let's figure out which groups of markers form relevant cell types
        cell_groups, cell_groups_name, clustering_labels, mat_average, markers_average = cell_subdivision(
            mat=mat_,
            mat_representative=mat_representative,
            markers=markers,
            markers_representative=markers_representative,
            batches=batches_,
            samples=samples_,
            marker_order=marker_order,
            three_markers=three_markers,
            percent_samplesxbatch=percent_samplesxbatch,
            min_cellxsample=min_cellxsample,
            cell_name=cell_name)

        # Try to classify undefined cells using a KNN classifier
        if knn_refine:
            clustering_labels = knn_classifier(mat_representative, clustering_labels, knn_min_probability=knn_min_probability)

        return is_label, cell_groups_name, clustering_labels, markers_representative_batches, markers_representative  # AT. Double-check this ', dfdata'
    else:
        return is_label, {-1: cell_name}, np.zeros(np.sum(is_label)) - 1, markers_representative_batches, []
