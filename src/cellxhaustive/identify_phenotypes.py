"""
Function that identifies most probable cell type and phenotype for a group of cells
using expression of its most relevant markers.
"""


# Import utility modules
import logging
import numpy as np
import random
from collections import defaultdict


# Import local functions
from assign_cell_types import assign_cell_types  # AT. Double-check path
from check_all_combinations import check_all_combinations  # AT. Double-check path
from knn_classifier import knn_classifier  # AT. Double-check path
# from cellxhaustive.assign_cell_types import assign_cell_types
# from cellxhaustive.check_all_combinations import check_all_combinations
# from cellxhaustive.knn_classifier import knn_classifier


# Function used in cellxhaustive.py
def identify_phenotypes(is_label, cell_name, mat_representative, batches_label,
                        samples_label, markers_representative, cell_types_dict,
                        two_peak_threshold, three_peak_markers,
                        three_peak_low, three_peak_high,
                        max_markers, min_annotations, max_solutions,
                        min_samplesxbatch, min_cellxsample,
                        knn_refine, knn_min_probability, cpu_eval_keep):
    """
    Function that identifies most probable cell type and phenotype for a group
    of cells using expression of its most relevant markers.

    Parameters
    ----------
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

    min_annotations: int (default=3)
      Minimum number of phenotypes for a combination of markers to be taken into
      account as a potential cell population. Must be in '[2; len(markers_representative)]',
      but it is advised to choose a value in '[3; len(markers_representative) - 1]'.

    max_solutions: int (default=10)
      Maximum number of optimal solutions to keep. If script finds more than
      'max_solutions' optimal marker combinations, 'max_solutions' combinations
      will be randomly chosen to be further processed and appear in final
      output. This parameter aims to limit computational burden.

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

    cpu_eval_keep: tuple(int) (default=(1, 1))
      Tuple of integers to set up CPU numbers in downstream nested functions.

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

    # Evaluate combinations of markers: go over every combination and find all
    # possible best combinations, phenotypes of all 'mat_subset_rep_markers'
    # cells and among those, phenotypes passing various thresholds (see function
    # definition for more information on those)
    logging.info('\t\tChecking all possible marker combinations')
    nb_solution, best_marker_comb, cell_phntp_comb, best_phntp_comb = check_all_combinations(
        mat_representative=mat_representative,
        batches_label=batches_label,
        samples_label=samples_label,
        markers_representative=markers_representative,
        two_peak_threshold=two_peak_threshold,
        three_peak_markers=three_peak_markers,
        three_peak_low=three_peak_low,
        three_peak_high=three_peak_high,
        max_markers=max_markers,
        min_annotations=min_annotations,
        min_samplesxbatch=min_samplesxbatch,
        min_cellxsample=min_cellxsample,
        cpu_eval_keep=cpu_eval_keep)

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
        new_labels = np.full(mat_subset_rep_markers.shape[0], cell_name)
        # No marker combination means no phenotype can be assigned to cells
        cell_phntp_comb = np.full(mat_subset_rep_markers.shape[0], 'No_phenotype')

        # Append results to dictionary
        results_dict[0]['new_labels'] = new_labels
        results_dict[0]['cell_phntp_comb'] = cell_phntp_comb

        if knn_refine:
            # Use default arrays as placeholders for reannotation results
            reannotation_proba = np.full(mat_subset_rep_markers.shape[0], np.nan)
            results_dict[0]['reannotated_labels'] = new_labels
            results_dict[0]['reannotated_phntp'] = cell_phntp_comb
            results_dict[0]['reannotation_proba'] = reannotation_proba

    elif nb_solution == 1:
        logging.info('\t\t\t<1> optimal combination found, building new cell types on it')
        logging.info(f'\t\t\t\tBest combination is {best_marker_comb}')
        # Slice matrix to keep only expression of best combination
        markers_rep_comb = markers_representative[np.isin(markers_representative, best_marker_comb)]
        mat_subset_rep_markers_comb = mat_representative[:, np.isin(markers_representative, best_marker_comb)]

        # Assign cell type using only markers from 'best_marker_comb'
        logging.info('\t\t\tAssigning cell types to each cell')
        new_labels, names_conv = assign_cell_types(
            mat_representative=mat_subset_rep_markers_comb,
            batches_label=batches_label,
            samples_label=samples_label,
            markers_representative=markers_rep_comb,
            cell_types_dict=cell_types_dict,
            cell_name=cell_name,
            cell_phntp=cell_phntp_comb,
            best_phntp=best_phntp_comb)

        # Append results to dictionary
        results_dict[0]['new_labels'] = new_labels
        results_dict[0]['cell_phntp_comb'] = cell_phntp_comb

        if knn_refine:
            # Reannotate only if conditions to run KNN-classifier are met
            is_undef = (new_labels == f'Other {cell_name}')  # Get number of undefined cells

            # At least 2 undefined cells and 2 cell types different from 'Other'
            if ((np.sum(is_undef) > 1) and (len(np.unique(new_labels)) > 2)):
                # Reannotate cells
                logging.info('\t\t\t\tRefining annotations with KNN-classifier')
                reannotated_labels, reannotation_proba = knn_classifier(
                    mat_representative=mat_subset_rep_markers_comb,
                    new_labels=new_labels,
                    is_undef=is_undef,
                    knn_min_probability=knn_min_probability,
                    knn_cpu=cpu_eval_keep)

                # Reverse dictionary to convert cell types back into phenotypes
                rev_names_conv = {val: key for key, val in names_conv.items()}

                # Update phenotypes of reannotated cells
                reannotated_phntp = np.vectorize(rev_names_conv.get)(reannotated_labels,
                                                                     'tmp')
                # Note: 'tmp' is only used to easily identify unidentified cells
                still_undef = (reannotated_phntp == 'tmp')
                reannotated_phntp[still_undef] = cell_phntp_comb[still_undef]

                # Append results to dictionary
                results_dict[0]['reannotated_labels'] = reannotated_labels
                results_dict[0]['reannotated_phntp'] = reannotated_phntp
                results_dict[0]['reannotation_proba'] = reannotation_proba

            else:  # If conditions are not met, no reannotation
                logging.warning('\t\t\t\tNot enough cell types or undefined cells to refine annotations with KNN-classifier')
                logging.warning(f'\t\t\t\t\tUndefined cells: <{np.sum(is_undef)}>')
                logging.warning(f'\t\t\t\t\tAnnotations: <{len(np.unique(new_labels))}>')
                # Use default arrays as placeholders for reannotation results
                reannotation_proba = np.full(new_labels.shape[0], np.nan)
                results_dict[0]['reannotated_labels'] = new_labels
                results_dict[0]['reannotated_phntp'] = cell_phntp_comb
                results_dict[0]['reannotation_proba'] = reannotation_proba

    else:  # Several solutions

        # Initialise counter of undefined cells
        logging.info(f'\t\t\t<{nb_solution}> optimal combinations found, building new cell types on them')
        undef_counter = []

        # Check number of solutions. If too high, randomly pick without repetitions
        if nb_solution > max_solutions:
            logging.warning(f'\t\t\t\tToo many combinations, choosing {max_solutions} randomly')
            logging.warning("\t\t\t\tIncrease '-s' parameter to process all of them")
            solutions = random.sample(range(nb_solution), max_solutions)
        else:
            solutions = range(nb_solution)

        logging.info('\t\t\tProcessing the different combinations')
        for i in solutions:
            logging.info(f'\t\t\t\tProcessing combination {i}: {best_marker_comb[i]}')
            # Slice matrix to keep only expression of best combination
            markers_rep_comb = markers_representative[np.isin(markers_representative, best_marker_comb[i])]
            mat_subset_rep_markers_comb = mat_representative[:, np.isin(markers_representative, best_marker_comb[i])]

            # Assign cell type using only markers from 'best_marker_comb[i]'
            logging.info('\t\t\t\t\tAssigning cell types to each cell')
            new_labels, names_conv = assign_cell_types(
                mat_representative=mat_subset_rep_markers_comb,
                batches_label=batches_label,
                samples_label=samples_label,
                markers_representative=markers_rep_comb,
                cell_types_dict=cell_types_dict,
                cell_name=cell_name,
                cell_phntp=cell_phntp_comb[i],
                best_phntp=best_phntp_comb[i])

            # Append results to dictionary
            results_dict[i]['new_labels'] = new_labels
            results_dict[i]['cell_phntp_comb'] = cell_phntp_comb[i]

            if knn_refine:
                # Reannotate only if conditions to run KNN-classifier are met
                is_undef = (new_labels == f'Other {cell_name}')  # Get number of undefined cells

                # At least 2 undefined cells and 2 cell types different from 'Other'
                if ((np.sum(is_undef) > 1) and (len(np.unique(new_labels)) > 2)):
                    # Reannotate cells
                    logging.info('\t\t\t\t\tRefining annotations with KNN-classifier')
                    reannotated_labels, reannotation_proba = knn_classifier(
                        mat_representative=mat_subset_rep_markers_comb,
                        new_labels=new_labels,
                        is_undef=is_undef,
                        knn_min_probability=knn_min_probability,
                        knn_cpu=cpu_eval_keep)

                    # Reverse dictionary to convert cell types into phenotypes
                    rev_names_conv = {val: key for key, val in names_conv.items()}

                    # Update phenotypes of reannotated cells
                    reannotated_phntp = np.vectorize(rev_names_conv.get)(reannotated_labels,
                                                                         'tmp')
                    # Note: 'tmp' is only used to easily identify unidentified cells
                    still_undef = (reannotated_phntp == 'tmp')
                    reannotated_phntp[still_undef] = cell_phntp_comb[i][still_undef]

                    # Append results to dictionary
                    results_dict[i]['reannotated_labels'] = reannotated_labels
                    results_dict[i]['reannotated_phntp'] = reannotated_phntp
                    results_dict[i]['reannotation_proba'] = reannotation_proba

                    # Record number of undefined cells after reannotation
                    nb_undef = np.sum(reannotated_labels == f'Other {cell_name}')
                    undef_counter.append(nb_undef)

                else:  # If conditions are not met, no reannotation
                    logging.warning('\t\t\t\t\tNot enough cell types or undefined cells to refine annotations with KNN-classifier')
                    logging.warning(f'\t\t\t\t\t\tUndefined cells: <{np.sum(is_undef)}>')
                    logging.warning(f'\t\t\t\t\t\tAnnotations: <{len(np.unique(new_labels))}>')
                    # Use default arrays as placeholders for reannotation results
                    reannotation_proba = np.full(new_labels.shape[0], np.nan)
                    results_dict[i]['reannotated_labels'] = new_labels
                    results_dict[i]['reannotated_phntp'] = cell_phntp_comb[i]
                    results_dict[i]['reannotation_proba'] = reannotation_proba

                    # Record number of undefined cells after reannotation
                    nb_undef = np.sum(is_undef)
                    undef_counter.append(nb_undef)

        if knn_refine:  # If KNN-classifier was run, count undefined cells again
            logging.info('\t\t\tRedefining optimal combinations after KNN-classification')
            # Get index of undefined cells minimum
            min_undef_idx = [i for i, x in enumerate(undef_counter) if x == min(undef_counter)]

            # Filter results using previous indices
            for key in list(results_dict.keys()):
                if key not in min_undef_idx:
                    del results_dict[key]  # Keep only indices with minimum undefined cells
            logging.info(f'\t\t\t\tFound {len(results_dict)} optimal combinations')

    return results_dict
