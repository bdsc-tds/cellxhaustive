"""
Reclassify unidentified cells using a KNN-classifier.
AT. Add general description here.
"""


# Import utility modules
import numpy as np


# Import ML modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# AT. Check presence/absence of all parameters/variable
def knn_classifier(mat_representative,
                   clustering_labels,
                   knn_min_probability=0.5):
    """
    Reclassify unidentified cells using a KNN-classifier.

    Parameters:
    -----------
    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.



    clustering_labels: ndarray
      Clustering indices, where -1 is assigned to undefined nodes/cells.
      # AT. Improve description

# AT. Need to check whether I can use the actual cell_types of if it's more convenient to use numbers

    knn_min_probability: float (default=0.5)
      Confidence threshold for KNN-classifier to reassign a new cell type
      to previously undefined cells.

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # It's common practice to standarize features
    undefined = (clustering_labels == -1)

    if np.sum(undefined) > 1 and len(np.unique(clustering_labels)) > 2:
        scaler = StandardScaler()
        mat_representative_scaled = scaler.fit_transform(mat_representative)

        # Initialise KNN-classifier
        clf = KNeighborsClassifier(n_neighbors=20)

        # Find cells that were classified and train classifier
        mat_representative_scaled_train = mat_representative_scaled[undefined == False, :]
        y_train = clustering_labels[undefined == False]
        clf.fit(mat_representative_scaled_train, y_train)

        # Find cells that were not classified and predict label
        y_test = clf.predict_proba(mat_representative_scaled[undefined, :])

        # Figure out which cells should be left undefined and which ones should
        # be reclassified based on model's ability to classify cells with
        # confidence (p > 0.5)
        newclustering_label = np.argmax(y_test, axis=1)
        newclustering_label[np.sum(y_test > knn_min_probability, axis=1) == 0] = -1

        # Reclassify undefined cells
        clustering_labels[undefined] = newclustering_label

    return clustering_labels
