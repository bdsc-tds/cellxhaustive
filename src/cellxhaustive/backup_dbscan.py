"""
AT. Add general description here.
Goes in identify_phenotypes.py in if test

    bimodality_selection_method: str (default = 'midpoint')
      Two possible methods: 'DBSCAN', which uses the clustering method with the
      same name, and 'midpoint', which uses markers closer to the normalized matrix
      # AT. Remove description if we decide to remove DBSCAN

"""


# Imports utility modules
import numpy as np

# Imports ML modules
from sklearn.cluster import DBSCAN  # AT.Remove if we decide to remove DBSCAN
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min  # AT.Remove if we decide to remove DBSCAN


def identify_phenotypes(mat, markers, batches, samples, is_label,
                        max_markers=15, min_annotations=3,  # bimodality_selection_method='midpoint',  # AT. Remove parameter if we decide to remove DBSCAN
                        min_cellxsample=10, min_samplesxbatch=0.5,
                        knn_refine=True, knn_min_probability=0.5,
                        cell_name=None):

    is_batch = (batches == batch)

    if bimodality_selection_method.upper() == 'DBSCAN':

        # Subset expression matrix to cells of the current batch only
        mat_subset = mat[is_batch, :]

        eps_marker_clustering = np.sqrt((pairwise_distances(mat_subset.transpose(), metric='euclidean') ** 2) / float(np.sum(is_batch)))
        # Notes:
        # - Pairwise distances are computed between rows
        # - Matrix is transposed to have scores per marker and not cell
        eps_marker_clustering = np.quantile(eps_marker_clustering[np.triu_indices(np.shape(eps_marker_clustering)[0], k=1)], q=0.05)
        # Notes:
        # - k=1 offsets the diagonal by 1 on the right, not to take the
        # 0-diagonal into account
        # - np.triu_indices() returns two arrays, one containing
        # x-coordinates, the other containing y-coordinates
        # Overall, this selects the 5th quantile of the upper-right triangle
        # of the transposed expression matrix of the whole batch

        # Create boolean array to select cells matching current label and batch
        is_label_batch = is_label[is_batch]

        # Subset expression matrix to cells of current batch and label only
        mat_subset = mat_subset[is_label_batch, :]

        # AT. Only reviewed until here

        # Transpose matrix to perform marker clustering
        X = mat_subset.transpose()

        # Calculate normalized pairwise distance
        eps = np.sqrt((pairwise_distances(X, metric='euclidean') ** 2) / float(np.shape(X)[1]))

        # Run DBSCAN
        km = DBSCAN(min_samples=1, eps=eps_marker_clustering, metric='precomputed', leaf_size=2).fit(eps)
        labels_ = km.labels_

        # Find the cluster center and generate an array of cluster by cell.
        cluster_centers_ = np.zeros((len(np.unique(labels_)), np.shape(mat_subset)[0]))
        for i in range(len(np.unique(labels_))):
            cluster_centers_[i, :] = np.mean(mat_subset[:, labels_ == i], axis=1)

        # Generate marker expression per cluster matrix
        mark_exprr = np.zeros((len(np.unique(labels_)), len(markers)))
        for idx, i in enumerate(labels_):
            mark_exprr[i, idx] += 1

        # Remove known invariants and associated clustered markers
        known_invariants = markers[np.sum(mark_exprr[np.any(mark_exprr[:, np.isin(markers, marker_order)], axis=1), :], axis=0) > 0]

        # Remove invariant markers based on the known invariants
        vt = VarianceThreshold().fit(mat_subset)
        invariants = markers[vt.variances_ <= np.max(vt.variances_[np.isin(markers, marker_order)])]
        invariants = np.unique(list(invariants) + list(known_invariants))

        # For the variable markers, find closer to the cluster centroids. Slice matrix subsequently
        markers_rep = []
        for idx, i in enumerate(np.sum(mark_exprr, axis=1)):
            m_step = markers[mark_exprr[idx, :] > 0]
            if i == 1:
                markers_rep += [m_step[0]]
            else:
                closest, _ = pairwise_distances_argmin_min(cluster_centers_[idx, :][np.newaxis, :],
                                                           X[np.isin(markers, m_step), :])
                markers_rep += [markers[np.isin(markers, m_step)][closest][0]]

        markers_rep = np.asarray(markers_rep)[np.isin(markers_rep, invariants, invert=True)]
        markers_rep = markers[np.isin(markers, markers_rep)]

        # Generate dictionary for relevant markers
        markers_rep_dict = dict()
        for i in markers_rep:
            markers_rep_dict[i] = markers[mark_exprr[labels_[markers == i][0], :] > 0]
