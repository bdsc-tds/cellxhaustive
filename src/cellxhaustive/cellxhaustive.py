"""
AT. Improve general description.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Utility functions to run the pipeline for phenotype identification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

--------------------------------------------------------------------
PURPOSE:
These are just the different components of the pipeline:
Main gating, selecting relevant markers, finding new phenotypes...
--------------------------------------------------------------------
NOTES:
Here I am sticking with the clean version of it, but I did try\
several versions of the same pipeline (see github history)
--------------------------------------------------------------------
"""


# Imports utility modules
import numpy as np

# Imports local functions
from identify_phenotypes import identify_phenotypes
# from cellxhaustive.identify_phenotypes import identify_phenotypes  # AT. Double-check path


# AT. Add description
def annotate(mat, markers, batches, samples, cell_labels,
             min_cellxsample=10, percent_samplesxbatch=0.5,
             max_midpoint_preselection=15, max_markers=15,
             min_annotations=3, bimodality_selection_method='midpoint',
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

    markers: array(str)
      1-D numpy array with markers matching each column of mat.

    batches: array(str)
      1-D numpy array with batch names of each cell of mat. Useful for defining
      the thresholds for the new annotations.

    samples: array(str)
      1-D numpy array with sample names of each cell of mat. Useful for defining
      the thresholds for the new annotations.

    cell_labels: array(str)
      1-D numpy array with main cell labels of each cell of mat.
      # AT. Set as an option, as it was used for the main gating
      # AT. Problem with annotations ontology/vocabulary?

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



    max_midpoint_preselection: int or None (default=15)
      # AT. Add parameter description
      # AT. Check if None is valid

    max_markers: int or None (default=15)
      Maximum number of relevant markers selected.

    min_annotations: int (default=3)
      Maximum number of relevant markers selected.
      # AT. Check if None is valid

    bimodality_selection_method: str (default = 'midpoint')
      Two possible methods: 'DBSCAN', which uses the clustering method with the
      same name, and 'midpoint', which uses markers closer to the normalized matrix.



    knn_refine: bool (default=False)
      If True, the clustering done via permutations of relevant markers will be
      refined using a KNN classifier.

    knn_min_probability: float (default=0.5)
      Confidence threshold for the KNN classifier to reassign a new cell type
      to previously undefined cells.

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    annotations = np.asarray(['undefined'] * len(cell_labels)).astype('U100')

    for label in np.unique(cell_labels):
        # Create boolean array to select cells matching current label
        is_label = (cell_labels == label)

        cell_groups, clustering_labels, mdictA, fmarker = identify_phenotypes(  # AT. Rename mdictA and fmarker
            is_label=is_label,
            mat=mat,
            markers=markers,
            batches=batches,
            samples=samples,
            min_cellxsample=min_cellxsample,
            percent_samplesxbatch=percent_samplesxbatch,
            max_midpoint_preselection=max_midpoint_preselection,
            max_markers=max_markers,
            min_annotations=min_annotations,
            bimodality_selection_method=bimodality_selection_method,
            knn_refine=knn_refine,
            cell_name=label)

        cell_dict = dict([tuple([x, cell_groups[x].split(" (")[0]])
                          for x in cell_groups])
        # AT. Because cell numbers is inside the name in (), we have to use .split() here --> Better to change the name in a function before

        annotations[is_label] = np.vectorize(cell_dict.get)(clustering_labels)
