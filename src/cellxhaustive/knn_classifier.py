"""
Reclassify unidentified cells using a KNN-classifier and return predicted
probability for reclassification.
"""


# Import utility modules
import copy
import logging
import numpy as np
import os
from joblib import parallel_config


# Import ML modules
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function used in identify_phenotypes.py
def knn_classifier(mat_representative, new_labels, is_undef,
                   knn_min_probability, knn_cpu):
    """
    Reclassify unidentified cells using a KNN-classifier and return predicted
    probability for reclassification.

    Parameters:
    -----------
    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers. This
      matrix is a subset of the general expression matrix and contains sliced
      data matching cell label, batch, and representative markers.

    new_labels: array(str)
      1-D numpy array with cell types and phenotypes, previously assigned by
      assign_cell_types(), for each cell of 'mat_representative'.

    is_undef: array(bool)
      1-D numpy array with cell type annotation for each cell of 'mat_representative'.
      True if cell type is undefined ('Other <main type>'), False otherwise.

    knn_min_probability: float (default=0.5)
      Confidence threshold for KNN-classifier to reassign a new cell type to
      previously undefined cells.

    knn_cpu: int (default=1)
      Integer to set up number of CPUs in multiprocessing jobs.

    Returns:
    --------
    reannotated_labels: array(str)
      1-D numpy array with cell types and phenotypes reannotated by KNN-classifier
      for each cell of 'mat_representative'.

    reannotation_proba: array(float, nan)
      1-D numpy array with prediction probability from KNN-classifier for reannotated
      cell types and phenotypes for each cell of 'mat_representative'.
    """

    # Set environment variables to prevent sklearn from using all cores
    tot_cpu = knn_cpu[0] * knn_cpu[1]
    os.environ['OPENBLAS_NUM_THREADS'] = f'{tot_cpu}'
    os.environ['OMP_NUM_THREADS'] = f'{tot_cpu}'
    os.environ['MKL_NUM_THREADS'] = f'{tot_cpu}'

    # Copy 'new_labels' array to avoid changing the original array
    reannotated_labels = copy.deepcopy(new_labels)

    # Split data in annotated (train/test) cells and undefined cells (i.e. cells
    # that will be re-annotated by classifier)
    logging.info('\t\t\t\t\t\tSplitting data in training and test datasets')
    annot_cells_mat = mat_representative[~ is_undef]
    annot_phntp = reannotated_labels[~ is_undef]
    undef_cells_mat = mat_representative[is_undef]
    undef_phntp = reannotated_labels[is_undef][0]

    # Further split annotated cells in training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(annot_cells_mat,
                                                        annot_phntp,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=annot_phntp)

    # Initialise scaler
    logging.info('\t\t\t\t\t\tInitializing KNN-classifier and parameters grid')
    scaler = StandardScaler()

    # Initialise KNN-classifier
    clf = KNeighborsClassifier(p=2, metric='minkowski', n_jobs=knn_cpu[1])
    # Note: default arguments "p=2, metric='minkowski'" are equivalent to
    # calculating Euclidean distances

    # Initialise pipeline with scaler and classifier
    pipeline = Pipeline([('scaler', scaler), ('KNN', clf)], verbose=False)

    # Define parameters grid for hypertuning
    param_grid = {'KNN__n_neighbors': np.arange(5, 21, 5),
                  'KNN__weights': ['uniform', 'distance'],
                  'KNN__leaf_size': np.arange(10, 31, 10)}

    # Build parameters grid object
    knn_grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy',
                            cv=5, n_jobs=knn_cpu[0], refit=True, verbose=0)

    # Find best parameters
    logging.info('\t\t\t\t\t\tTuning hyperparameters')
    # Use different backend for parallel computing to avoid GridSearchCV hanging
    # and returning joblib loky 'resource_tracker' warnings
    with parallel_config(backend='multiprocessing'):
        best_model = knn_grid.fit(X_train, y_train)
    logging.info(f'\t\t\t\t\t\t\tBest parameters found: {best_model.best_params_}')
    logging.info(f'\t\t\t\t\t\t\twith a max accuracy of: {best_model.best_score_:.3f}')

    # Apply classifier to undefined cells
    logging.info(f'\t\t\t\t\t\tApplying KNN-classifier to undefined cells')
    undef_cells_pred = best_model.predict_proba(undef_cells_mat)
    # Note: this returns an array of probabilities for a cell to belong to a
    # certain cell type

    logging.info('\t\t\t\t\t\tSelecting annotations passing knn_min_probability threshold')
    # Initialise empty array to store updated annotations
    reannotated = np.full(undef_cells_mat.shape[0], undef_phntp, dtype='U150')

    # Extract cell types ordered by sklearn
    ordered_cell_types = best_model.classes_

    # Create empty array for reannotation probability
    reannotation_proba = np.full(reannotated_labels.shape[0], np.nan)

    # Get maximum proba for each row
    reannotation_proba[is_undef] = np.max(undef_cells_pred, axis=1)

    # Check if maximum proba of each row is larger than 'knn_min_probability'
    is_max_higher = (reannotation_proba[is_undef] > knn_min_probability)

    # Find index of maximum probability for each row
    max_idx = np.argmax(undef_cells_pred, axis=1)

    # Extract updated annotations passing threshold
    reannotated[is_max_higher] = ordered_cell_types[max_idx][is_max_higher]

    # Assign new annotations to original array
    logging.info('\t\t\t\t\t\tAssigning new annotations passing threshold')
    reannotated_labels[is_undef] = reannotated

    # Reset environment variables
    del os.environ['OPENBLAS_NUM_THREADS']
    del os.environ['OMP_NUM_THREADS']
    del os.environ['MKL_NUM_THREADS']

    return reannotated_labels, reannotation_proba
