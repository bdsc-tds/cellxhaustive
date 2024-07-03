"""
Function that identifies most probable cell type and phenotype for a group of cells
using expression of its most relevant markers.
"""


# Import utility modules
import logging
import numpy as np
import os
import random
from collections import defaultdict


# Import local functions
from assign_cell_types import assign_cell_types  # AT. Double-check path
from check_all_combinations import check_all_combinations  # AT. Double-check path
from determine_marker_status import determine_marker_status  # AT. Double-check path
from knn_classifier import knn_classifier  # AT. Double-check path
from utils import setup_log  # AT. Double-check path
# from cellxhaustive.assign_cell_types import assign_cell_types
# from cellxhaustive.check_all_combinations import check_all_combinations
# from cellxhaustive.determine_marker_status import determine_marker_status
# from cellxhaustive.knn_classifier import knn_classifier
# from cellxhaustive.utils import setup_log


# Function used in cellxhaustive.py
def identify_phenotypes(is_label, cell_name, mat_representative, batches_label,
                        samples_label, markers_representative, markers_interest,
                        detection_method, cell_types_dict, two_peak_threshold,
                        three_peak_markers, three_peak_low, three_peak_high,
                        max_markers, min_samplesxbatch, min_cellxsample,
                        knn_refine, knn_min_probability, multipop, processpool):
    """
    Function that identifies most probable cell type and phenotype for a group
    of cells using expression of its most relevant markers.

    Parameters:
    -----------
    is_label: array(bool)
      1-D numpy array with booleans to indicate cells matching current cell type.

    cell_name: str or None
      Base name for cell types (e.g. CD4 T-cells for 'CD4T').

    mat_representative: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    batches_label: array(str)
      1-D numpy array with batch names of each cell of 'mat_representative'.

    samples_label: array(str)
      1-D numpy array with sample names of each cell of 'mat_representative'.

    markers_representative: array(str)
      1-D numpy array with markers matching each column of 'mat_representative'.

    markers_interest: array(str)
      1-D numpy array with markers that must appear in optimal marker combinations.

    detection_method: 'auto' or int
      Method used to stop search for optimal marker combinations. If 'auto', use
      default algorithm relying on maximum number of phenotypes. If int, create a
      combination with exactly this number of markers.

    cell_types_dict: {str: list()}
      Dictionary with cell types as keys and list of cell type defining markers
      as values.

    two_peak_threshold: float (default=3)
      Threshold to consider when determining whether a two-peaks marker is
      negative or positive. Expression below this threshold means marker will
      be considered negative. Conversely, expression above this threshold means
      marker will be considered positive.

    three_peak_markers: list(str) (default=[])
      List of markers that have three peaks.

    three_peak_low: float (default=2)
      Threshold to consider when determining whether a three-peaks marker is
      negative or low positive. Expression below this threshold means marker
      will be considered negative. See description of 'three_peak_high' for more
      information on low_positive markers.

    three_peak_high: float (default=4)
      Threshold to consider when determining whether a three-peaks marker is
      low_positive or positive. Expression above this threshold means marker
      will be considered positive. Expression between 'three_peak_low' and
      'three_peak_high' means marker will be considered low_positive.

    max_markers: int (default=15)
      Maximum number of relevant markers to select among total list of markers
      from total markers array. Must be less than or equal to 'len(markers_representative)'.

    min_samplesxbatch: float (default=0.5)
      Minimum proportion of samples within each batch with at least 'min_cellxsample'
      cells for a new annotation to be considered. In other words, by default,
      an annotation needs to be assigned to at least 10 cells/sample (see
      description of next parameter) in at least 50% of samples within a batch
      to be considered.

    min_cellxsample: float (default=10)
      Minimum number of cells within each sample in 'min_samplesxbatch' % of
      samples within each batch for a new annotation to be considered. In other
      words, by default, an annotation needs to be assigned to at least 10
      cells/sample in at least 50% of samples (see description of previous
      parameter) within a batch to be considered.

    knn_refine: bool (default=True)
      If True, clustering done via permutations of relevant markers will be
      refined using a KNN-classifier.

    knn_min_probability: float (default=0.5)
      Confidence threshold for KNN-classifier to reassign a new cell type
      to previously undefined cells.

    multipop: bool
      Boolean indicating whether multiple cell populations are processed.

    processpool: None or pathos.pools.ProcessPool object
      If not None, ProcessPool object to use in downstream nested functions.

    Returns:
    --------
    results_dict: dict {str: list(array(str, float, np.nan))}
      Dictionary with 2 mandatory keys and 2 optional keys:
        - 'new_labels' (mandatory): list of 1-D numpy arrays with cell type for
          each cell of 'mat_representative[is_label]'. 1 array per optimal
          marker combination.
        - 'cell_phntp_comb' (mandatory): list of 1-D numpy arrays with full
          phenotype for each cell of 'mat_representative[is_label]'. 1 array per
          optimal marker combination.
        - 'reannotated_labels' (optional): list of 1-D numpy arrays with cell
          type for each cell of 'mat_representative[is_label]'. 1 array per
          optimal marker combination. Undefined cell types are reannotated by a
          KNN-classifier. Available only if 'knn_refine=True'.
        - 'reannotation_proba' (optional): list of 1-D numpy arrays with
          reannotation prediction probability determined by a KNN-classifier for
          each undefined cell of 'mat_representative[is_label]'. 1 array per
          optimal marker combination. Available only if 'knn_refine=True'.
    """

    # Set-up logging configuration
    setup_log(os.environ['LOG_FILE'], os.environ['LOG_LEVEL'], 'a')

    # Evaluate combinations of markers: go over every combination and find all
    # possible best combinations, phenotypes of all 'mat_subset_rep_markers'
    # cells and among those, phenotypes passing various thresholds (see function
    # definition for more information on those)
    logging.info('\t\tChecking all possible marker combinations')
    nb_solution, best_marker_comb = check_all_combinations(
        mat_representative=mat_representative,
        batches_label=batches_label,
        samples_label=samples_label,
        markers_representative=markers_representative,
        markers_interest=markers_interest,
        detection_method=detection_method,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
        max_markers=max_markers,
        min_samplesxbatch=min_samplesxbatch,
        min_cellxsample=min_cellxsample,
        processpool=processpool,
        cell_name=cell_name)

    # Initialise result dictionary with empty lists
    # Note: even if lists end up with only 1 element, it makes processing results
    # in cellxhaustive.py easier
    results_dict = defaultdict(dict)

    logging.info('\t\tAssigning cell types based on optimal combinations')
    if nb_solution == 0:
        # 'best_marker_comb' is empty, which means that no marker combination
        # was found to properly represent cell type 'label' (from annotate(), so
        # keep original annotation
        logging.warning('\t\t\tNo optimal combination found, reverting to original cell types')
        new_labels = np.full(mat_representative.shape[0], f'Other {cell_name}')
        # No marker combination means no phenotype can be assigned to cells
        cell_phntp_comb = np.full(mat_representative.shape[0], 'No_phenotype')

        # Append results to dictionary
        results_dict[0]['new_labels'] = new_labels
        results_dict[0]['cell_phntp_comb'] = cell_phntp_comb

        if knn_refine:
            # Use default arrays as placeholders for reannotation results
            reannotation_proba = np.full(mat_representative.shape[0], np.nan)
            results_dict[0]['reannotated_labels'] = new_labels
            results_dict[0]['reannotated_phntp'] = cell_phntp_comb
            results_dict[0]['reannotation_proba'] = reannotation_proba

    else:  # At least one solution, but can account for more if needed

        # Initialise counter of undefined cells
        logging.info(f'\t\t\t<{nb_solution}> optimal combinations found, building new cell types on them')
        undef_counter = []

        # Check number of solutions. If too high, randomly pick without repetitions
        if nb_solution > 10:
            logging.warning(f'\t\t\t\tToo many combinations, choosing 10 randomly')
            solutions = random.sample(range(nb_solution), 10)
        else:
            solutions = range(nb_solution)

        logging.info(f'\t\t\tProcessing all {cell_name} combinations')
        for i in solutions:
            logging.info(f'\t\t\t\tProcessing combination {i}: {best_marker_comb[i]}')
            # Slice matrix to keep only expression of best combination
            markers_rep_comb = markers_representative[np.isin(markers_representative, best_marker_comb[i])]
            mat_subset_rep_markers_comb = mat_representative[:, np.isin(markers_representative, best_marker_comb[i])]

            # Determine cell phenotypes using markers from 'markers_rep_comb'
            cell_phntp_comb = determine_marker_status(
                mat_comb=mat_subset_rep_markers_comb,
                markers_comb=markers_rep_comb,
                two_peak_threshold=two_peak_threshold,
                three_peak_markers=three_peak_markers,
                three_peak_low=three_peak_low,
                three_peak_high=three_peak_high)

            # Initialise empty string array to record 'significant' phenotypes,
            # i.e. passing toth min_cellxsample and min_samplesxbatch thresholds
            best_phntp_comb = np.empty(0, dtype=cell_phntp_comb.dtype)
            # Note: get dtype from previous array to avoid using 'object'

            # Determine 'significant' phenotypes
            for phenotype in np.unique(cell_phntp_comb):

                # Initialise boolean to keep 'phenotype'
                keep_phenotype = True

                # Process batches separately
                for batch in np.unique(batches_label):
                    # Split cell type data according to batch
                    phenotypes_batch = cell_phntp_comb[batches_label == batch]

                    # Split sample data, first according to batch and then cell type
                    phenotype_samples = samples_label[batches_label == batch][phenotypes_batch == phenotype]

                    # If there are no cells of type 'phenotype' in 'batch', that means
                    # 'phenotype' cannot be present in all batches, so stop now
                    if phenotype_samples.size == 0:
                        keep_phenotype = False
                        break

                    # Calculate number of unique samples in current batch and cell type
                    samples_nb = float(len(np.unique(phenotype_samples)))

                    # Count number of cells per phenotype in each sample
                    cell_count_sample = np.asarray([np.sum(phenotype_samples == smpl)
                                                    for smpl in np.unique(phenotype_samples)])

                    # Check whether previous counts satisfy cell/sample threshold
                    keep_phenotype_batch = (cell_count_sample >= min_cellxsample)

                    # Calculate proportion of samples in current batch satisfying
                    # cell/sample threshold
                    keep_phenotype_batch = (np.sum(keep_phenotype_batch, axis=0) / samples_nb)
                    # Notes:
                    # - 'keep_phenotype_batch' is a boolean array, so it can be summed

                    # Check whether previous proportion satisfies sample/batch threshold
                    keep_phenotype_batch = (keep_phenotype_batch >= min_samplesxbatch)

                    # Intersect batch results with general results
                    keep_phenotype = (keep_phenotype and keep_phenotype_batch)
                    # Note: for consistency, phenotypes have to be present in all batches,
                    # hence usage of 'and'

                # If 'phenotype' is 'significant', keep it
                if keep_phenotype:
                    best_phntp_comb = np.append(best_phntp_comb, phenotype)

            # Assign cell type using only markers from 'best_marker_comb[i]'
            logging.info('\t\t\t\t\tAssigning cell types to each cell')
            new_labels, names_conv = assign_cell_types(
                mat_representative=mat_subset_rep_markers_comb,
                batches_label=batches_label,
                samples_label=samples_label,
                markers_representative=markers_rep_comb,
                cell_types_dict=cell_types_dict,
                cell_name=cell_name,
                cell_phntp=cell_phntp_comb,
                best_phntp=best_phntp_comb)

            # Update non-significant phenotypes to make final results easier to
            # understand and process
            cell_phntp_comb = cell_phntp_comb.astype(dtype=object)  # To avoid strings getting cut
            cell_phntp_comb[np.isin(cell_phntp_comb,
                                    best_phntp_comb,
                                    invert=True)] = f'Other {cell_name} phenotype'
            cell_phntp_comb = cell_phntp_comb.astype(dtype=str)

            # Append results to dictionary
            results_dict[i]['new_labels'] = new_labels
            results_dict[i]['cell_phntp_comb'] = cell_phntp_comb

            if knn_refine:
                # Reannotate only if conditions to run KNN-classifier are met
                is_undef = (new_labels == f'Unannotated {cell_name}')  # Get number of undefined cells

                # At least 2 undefined cells and 2 cell types different from 'Unannotated'
                if ((np.sum(is_undef) > 1) and (len(np.unique(new_labels)) > 2)):
                    # Adapt CPUs used by KNN-classifier to running conditions
                    logging.info('\t\t\t\t\tSetting CPUs for KNN-classifier')
                    if not processpool:
                        knn_cpu = 1
                    else:
                        if multipop:  # If multiple population, use ~half of CPUs
                            knn_cpu = processpool.ncpus // 2
                        else:  # If only population, use all CPUs
                            knn_cpu = processpool.ncpus
                    logging.info(f"\t\t\t\t\t\tUsing {knn_cpu}{'s' if knn_cpu > 1 else ''} CPUs")
                    # Reannotate cells
                    logging.info('\t\t\t\t\tRefining annotations with KNN-classifier')
                    if not processpool:  # Cannot use ProcessPool to parallelise
                        reannotated_labels, reannotation_proba = knn_classifier(
                            mat_representative=mat_subset_rep_markers_comb,
                            new_labels=new_labels,
                            is_undef=is_undef,
                            knn_min_probability=knn_min_probability,
                            knn_cpu=knn_cpu)
                    else:  # Use 'pipe' to submit task to ProcessPool and parallelise
                        reannotated_labels, reannotation_proba = processpool.pipe(
                            knn_classifier,
                            mat_representative=mat_subset_rep_markers_comb,
                            new_labels=new_labels,
                            is_undef=is_undef,
                            knn_min_probability=knn_min_probability,
                            knn_cpu=knn_cpu)

                    # Reverse dictionary to convert cell types into phenotypes
                    rev_names_conv = {val: key for key, val in names_conv.items()}

                    # Update phenotypes of reannotated cells
                    reannotated_phntp = np.vectorize(rev_names_conv.get, otypes=[object])(reannotated_labels,
                                                                                          'tmp')
                    # Note: 'tmp' is only used to easily identify unidentified cells
                    still_undef = (reannotated_phntp == 'tmp')
                    reannotated_phntp[still_undef] = cell_phntp_comb[still_undef]

                    # Append results to dictionary
                    results_dict[i]['reannotated_labels'] = reannotated_labels
                    results_dict[i]['reannotated_phntp'] = reannotated_phntp
                    results_dict[i]['reannotation_proba'] = reannotation_proba

                    # Record number of undefined cells after reannotation
                    nb_undef = np.sum(reannotated_labels == f'Unannotated {cell_name}')
                    undef_counter.append(nb_undef)

                else:  # If conditions are not met, no reannotation
                    logging.warning('\t\t\t\t\tNot enough cell types or undefined cells to refine annotations with KNN-classifier')
                    logging.warning(f'\t\t\t\t\t\tUndefined cells: <{np.sum(is_undef)}>')
                    logging.warning(f'\t\t\t\t\t\tAnnotations: <{len(np.unique(new_labels))}>')
                    # Use default arrays as placeholders for reannotation results
                    reannotation_proba = np.full(new_labels.shape[0], np.nan)
                    results_dict[i]['reannotated_labels'] = new_labels
                    results_dict[i]['reannotated_phntp'] = cell_phntp_comb
                    results_dict[i]['reannotation_proba'] = reannotation_proba

                    # Record number of undefined cells after reannotation
                    nb_undef = np.sum(is_undef)
                    undef_counter.append(nb_undef)

        # If there are several solutions and KNN-classifier was run, check
        # number of remaining undefined cells to keep only solutions minimising
        # this number
        if knn_refine and nb_solution > 1:
            logging.info('\t\t\tRedefining optimal combinations after KNN-classification')
            # Get index of undefined cells minimum
            min_undef_idx = [i for i, x in enumerate(undef_counter)
                             if x == min(undef_counter)]

            # Filter results using previous indices
            for key in list(results_dict.keys()):
                if key not in min_undef_idx:
                    del results_dict[key]  # Keep only indices with minimum undefined cells
            logging.info(f'\t\t\t\tFound {len(results_dict)} optimal combinations')

    return results_dict
