"""
AT. Add general description here.
"""


# Imports utility modules
import numpy as np

# Imports ML modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# AT. Check presence/absence of all parameters/variable
# Reclassify undefined cells with KNN classifier
def knn_classifier(mat_representative, clustering_labels, knn_min_probability=0.5):
    """
    Reclassify cells based on KNN classifier

    Parameters:
    -----------
    mat_representative: array(float)
      2-D numpy array expression matrix of the representative markers, with
      cells in D0 and markers in D1. In other words, rows contain cells and
      columns contain markers.



    clustering_labels: ndarray
      Clustering indices, where -1 is assigned to undefined nodes/cells.

    knn_min_probability: float (default=0.5)
      Confidence threshold for the KNN classifier to reassign new cell type.

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # It's common practice to standarize features
    undefined = (clustering_labels == -1)

    if np.sum(undefined) > 1 and len(np.unique(clustering_labels)) > 2:
        scaler = StandardScaler()
        mat_representative_scaled = scaler.fit_transform(mat_representative)

        # Start knn classifier
        clf = KNeighborsClassifier(n_neighbors=20)

        # Find cells that were classified and train the classifier
        mat_representative_scaled_train = mat_representative_scaled[undefined == False, :]
        y_train = clustering_labels[undefined == False]
        clf.fit(mat_representative_scaled_train, y_train)

        # Find cells that were not classified and predict label
        y_test = clf.predict_proba(mat_representative_scaled[undefined, :])

        # Figure out which cells should be left undefined and which ones should
        # be reclassified based on the model's ability to classify cells with
        # confidence (p > 0.5)
        newclustering_label = np.argmax(y_test, axis=1)
        newclustering_label[np.sum(y_test > knn_min_probability, axis=1) == 0] = -1

        # Reclassify undefined cells
        clustering_labels[undefined] = newclustering_label

    return clustering_labels
