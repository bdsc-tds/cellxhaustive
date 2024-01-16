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
             max_markers=15, min_annotations=3,
             min_cellxsample=10, min_samplesxbatch=0.5,
             knn_refine=True, knn_min_probability=0.5):
    # AT. Missing three_peak_markers to carry over in all functions?
    # AT. Same for peak thresholds
    # AT. ==> Discuss this with Bernat
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

    cell_labels: array(str)
      1-D numpy array with main cell labels of each cell of 'mat'.
      # AT. Set as an option, as it was used for the main gating
      # AT. Problem with annotations ontology/vocabulary?

    max_markers: int (default=15)
      Maximum number of markers to select among the total list of markers from
      the 'markers' array. Must be less than or equal to 'len(markers)'.

    min_annotations: int (default=3)
      Minimum number of phenotypes for a combination of markers to be taken into
      account as a potential cell population. Must be in '[2; len(markers)]',
      but it is advised to choose a value in '[3; len(markers) - 1]'.

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

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    annotations = np.asarray(['undefined'] * len(cell_labels)).astype('U100')

    # AT. Import cell_ontology here? To import it once instead of several times in
    # scripts further down in the workflow?
    # AT. Make this path more flexible, maybe with a default value in argument parsing?
    # Note: this file was created using data from
    # https://github.com/RGLab/rcellontologymapping/blob/main/src/src/ImmportDefinitions.hs
    with open('../data/config/major_cell_types.json') as in_cell_types:
        cell_types_dict = json.load(in_cell_types)

    for label in np.unique(cell_labels):

        # Create boolean array to select cells matching current label
        is_label = (cell_labels == label)

        # AT. Add annotations on what function does
        cell_groups_name, clustering_labels = identify_phenotypes(
        # AT. Some variables are not used anymore, so no point in assigning them: is_label, markers_rep_batches, markers_rep_all
        # is_label, cell_groups_name, clustering_labels, markers_rep_batches, markers_rep_all = identify_phenotypes(
                  mat=mat,
                  markers=markers,
                  batches=batches,
                  samples=samples,
                  is_label=is_label,
                  max_markers=max_markers,
                  min_annotations=min_annotations,
                  min_cellxsample=min_cellxsample,
                  min_samplesxbatch=min_samplesxbatch,
                  cell_types_dict=cell_types_dict,
                  knn_refine=knn_refine,
                  knn_min_probability=knn_min_probability,
                  cell_name=label)

        cell_dict = dict([tuple([x, cell_groups_name[x].split(" (")[0]])
                          for x in cell_groups_name])
        # AT. Because cell numbers is inside the name in (), we have to use .split() here --> Better to change the name in a function before

        annotations[is_label] = np.vectorize(cell_dict.get)(clustering_labels)


# What is the output format? tsv, table, object... ???
