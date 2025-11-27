"""
Reclassify unidentified cells using a KNN-classifier and return predicted
probability for reclassification.
"""

# Import utility modules
import logging
import numpy as np
from joblib import parallel_config


# Import ML modules
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function used in identify_phenotypes.py
def knn_classifier(
    cell_name,
    best_comb_name,
    mat_representative,
    new_labels,
    is_undef,
    knn_min_probability,
    knn_cpu,
):
    """
    Reclassify unidentified cells using a KNN-classifier and return predicted
    probability for reclassification.

    Parameters:
    -----------
    cell_name: str
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

    best_comb_name: str
      Best combination name.

    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

    new_labels: array(str)
      1-D numpy array with cell types and phenotypes, previously assigned by
      assign_cell_types(), for each cell of 'mat_representative'.

    is_undef: array(bool)
      1-D numpy array with cell type annotation for each cell of
      'mat_representative'. True if cell type is undefined ('Other
      <main type>'), False otherwise.

    knn_min_probability: float (default=0.5)
      Confidence threshold for KNN-classifier to reassign a new cell type to
      previously undefined cells.

    knn_cpu: int (default=1)
      Number of CPUs to use for KNN-classifier processing.

    Returns:
    --------
    reannotated_labels: array(str)
      1-D numpy array with cell types and phenotypes reannotated by
      KNN-classifier for each cell of 'mat_representative'.

    reannotation_proba: array(float, nan)
      1-D numpy array with prediction probability from KNN-classifier for
      reannotated cell types and phenotypes for each cell of
      'mat_representative'.
    """

    # Copy 'new_labels' array. Use dtype="object" to avoid strings getting cut
    reannotated_labels = new_labels.astype(dtype="object")

    # Split data in annotated (train/test) cells and undefined cells (i.e. cells
    # that will be re-annotated by classifier)
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Splitting data in training and test datasets"
    )
    annot_cells_mat = mat_representative[~is_undef]
    annot_phntp = reannotated_labels[~is_undef]
    undef_cells_mat = mat_representative[is_undef]
    undef_phntp = reannotated_labels[is_undef][0]

    # Further split annotated cells in training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        annot_cells_mat,
        annot_phntp,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=annot_phntp,
    )

    # Initialize pipeline with scaler and classifier
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Initializing KNN-classifier and parameters grid"
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("KNN", KNeighborsClassifier(p=2, metric="minkowski", n_jobs=None)),
        ],
        verbose=False,
    )

    # Define parameters grid for hypertuning
    param_grid = {
        "KNN__n_neighbors": [5, 10, 15, 20],
        "KNN__weights": ["uniform", "distance"],
        "KNN__leaf_size": [10, 20, 30],
    }

    # Build parameters grid object
    knn_grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=None,
        refit=True,
        verbose=0,
    )

    # Find best parameters
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Tuning hyperparameters"
    )
    # Use different backend for parallel computing to avoid GridSearchCV hanging
    # and returning joblib loky 'resource_tracker' warnings
    with parallel_config(backend="multiprocessing", n_jobs=knn_cpu):
        best_model = knn_grid.fit(X_train, y_train)

    # Display best parameters
    best_model_str = ", ".join(
        f"{k}: {v}" for k, v in best_model.best_params_.items()
    )
    logging.info(
        f"\t\t\t\t\t\t{cell_name} - ({best_comb_name}): Best parameters found: {best_model_str} with a max accuracy of: {best_model.best_score_:.3f}"
    )

    # Apply classifier to undefined cells
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Applying KNN-classifier to undefined cells"
    )
    undef_cells_pred = best_model.predict_proba(undef_cells_mat)
    # Note: this returns an array of probabilities for a cell to belong to a
    # certain cell type

    # Get max probabilities and indices
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Selecting annotations passing knn_min_probability threshold"
    )
    max_proba = undef_cells_pred.max(axis=1)
    max_idx = undef_cells_pred.argmax(axis=1)

    # Create empty array for reannotation probability and get maximum proba for
    # each row
    reannotation_proba = np.full(len(reannotated_labels), np.nan)
    reannotation_proba[is_undef] = max_proba

    # Check if maximum proba of each row is larger than 'knn_min_probability'
    passes_threshold = max_proba > knn_min_probability

    # Initialise empty array to store updated annotations with undefined labels
    reannotated = np.full(len(undef_cells_mat), undef_phntp, dtype="object")

    # Assign annotations passing threshold
    if np.any(passes_threshold):
        ordered_cell_types = best_model.classes_
        reannotated[passes_threshold] = ordered_cell_types[
            max_idx[passes_threshold]
        ]

    # Assign new annotations to original array
    logging.info(
        f"\t\t\t\t\t{cell_name} - ({best_comb_name}): Assigning new annotations passing threshold"
    )
    reannotated_labels[is_undef] = reannotated
    reannotated_labels = reannotated_labels.astype(dtype="str")

    return reannotated_labels, reannotation_proba
