#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Utility functions to run the pipeline for phenotype identification
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#----------------------------------------------------------------------
# PURPOSE:
# These are just the different components of the pipeline:
# Main gating, selecting relevant markers, finding new phenotypes,...
#-----------------------------------------------------------------------
# NOTES:
# Here I am sticking with the clean version of it, but I did try\
# several versions of the same pipeline (see github history)
#-----------------------------------------------------------------------

# Utility libraries
import numpy as np
import pandas as pd
import copy
import itertools as ite
import re
import sys
import pickle

# ML libraries
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Local imports
from cellxhaustive.cell_ontology_names import *
from cellxhaustive.main_gating import gaussian_gating
from cellxhaustive.utility import find_set_differences
from cellxhaustive.select import select_cells

# Brute force way to select cells given normalized data
def select_cells(mat, markers, positive=[], negative=[], lowpositive=[], three_marker=[], two_peak_threshold=3, three_peak_lower_threshold = 2, three_peak_upper_threshold = 4):
    """Main gating strategy

       Parameters
       ----------
       mat : ndarray
         A 2-D array normalized expression matrix

       markers : array
         A 1-D array with the markers in `mat` corresponding to each column

       positive : list(str)
         List of markers that are positively expressed.
         
       negative : list(str)
         List of markers that are negatively expressed.
         
       lowpositive : list(str)
         List of markers that are express a middle peak.

       three_marker : list(str)
         List of markers that have three peaks.
    """
    
    # Start a vector that will define the filtering
    pos = True
    
    # Find markers with two peaks and markers with three
    three_marker = np.array(three_marker)
    two_marker = markers[np.isin(markers, three_marker)==False]
    
    # Start by selecting on markers with three peaks
    if len(three_marker)>0:
        # If positive, find values above the corresponding threshold
        for i in three_marker[np.isin(three_marker, positive)]:
            pos = np.logical_and(pos, mat[:, np.where(markers==i)[0][0]]>=three_peak_upper_threshold)
        # If middle peak, find values above and below the corresponding threshold
        
        for i in three_marker[np.isin(three_marker, lowpositive)]:
            pos = np.logical_and(pos,
            np.logical_and(
                        mat[:, np.where(markers==i)[0][0]]<three_peak_upper_threshold,
                        mat[:, np.where(markers==i)[0][0]]>=three_peak_lower_threshold)
                        )
                        
        # If negative, find values below the corresponding threshold
        for i in three_marker[np.isin(three_marker, negative)]:
            pos = np.logical_and(pos, mat[:, np.where(markers==i)[0][0]]<three_peak_lower_threshold)
            
    if len(two_marker)>0:
        # If positive, find values above the corresponding threshold
        for i in two_marker[np.isin(two_marker, positive)]:
            pos = np.logical_and(pos, mat[:, np.where(markers==i)[0][0]]>=two_peak_threshold)
            
        # If negative, find values below the corresponding threshold
        for i in two_marker[np.isin(two_marker, negative)]:
            pos = np.logical_and(pos, mat[:, np.where(markers==i)[0][0]]<two_peak_threshold)
    
    return pos

# Permute across positive and negative expression of the relevant markers, and identify new cell types
def cell_subdivision_counts(mat, mat_representative, markers, markers_representative, batches, samples, marker_order, three_marker=[], p_min=np.array([0.1,0.2]), s_min=np.array([5,10])):
    """Cell line subdivision

       Parameters
       ----------
       mat : ndarray
         A 2-D array expression matrix
         
       mat_representative : ndarray
         A 2-D array expression matrix of the representative markers

       markers : array
         A 1-D array with the markers in `mat` corresponding to each column

       markers_representative : array
         A 1-D array with the relevant markers in `mat_representative` corresponding to each column

       marker_order : list(str)
         List of markers used in the gating strategy ordered accordingly.

       three_marker : list(str)
         List of markers with potentially three peaks.
         
       p_min : int
         minimum proportion of samples for annotation to be considered
         
       random_state : int or None, (default=None)
          random seed.
    """

    undefined = np.zeros((len(p_min), len(s_min)))
    results = np.zeros((len(p_min), len(s_min)))

    # Find different groups of positive cells
    three_marker_ = np.asarray(three_marker)[np.isin(three_marker, markers_representative)]
    three_marker_low = [x + "_low" for x in three_marker_]
    three_markers = list(three_marker_)
    markers_representative_ = list(markers_representative) + list(three_marker_low)
    groups = [tuple(np.unique(i)) for i in ite.combinations_with_replacement(
        markers_representative_,len(markers_representative_))]
    groups = [tuple([])] + groups
    
    # Loop over unique groups
    for i in set(groups):
        iteration = np.unique(i)
        if np.any(np.isin(three_marker_[np.isin(three_marker_low, iteration)], iteration)):
            continue
        else:
            # The groups are defined by positive markers
            positives = np.asarray(markers_representative)[np.isin(markers_representative, iteration)]
            
            # The negatives should be defined easily
            negatives = np.asarray(markers_representative)[np.isin(markers_representative, iteration)==False]
            
            # The low positives are those markers that appear in three_marker_low
            lowpositives = np.asarray(three_marker_)[np.isin(three_marker_low, iteration)]
        
        # Figure out which cells fulfill these rules
        cells = select_cells(mat_representative,
                                 markers_representative,
                                 positive=positives,
                                 negative=negatives,
                                 lowpositive = lowpositives,
                                 three_marker=three_markers)

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = np.zeros(np.shape(results))==0
        for b in np.unique(batches):
            cells_ = cells[batches == b]
            samples_ = samples[batches == b]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_]==x) for x in np.unique(samples_)])
            keep_cell_type_ = keep_cell_type_[:,np.newaxis]>s_min
            keep_cell_type_ = (np.sum(keep_cell_type_, axis=0)/float(len(np.unique(samples_))))[np.newaxis,:]
            keep_cell_type_ = keep_cell_type_>p_min[:,np.newaxis]
            keep_cell_type = np.logical_and(keep_cell_type, keep_cell_type_)

        results += keep_cell_type * 1
        undefined += (keep_cell_type == False) * np.sum(cells)

    return results, undefined
    
def check_all_subsets(max_markers, x_p, y_ns, mat_, mat_representative, markers, markers_representative, marker_order, batches, samples, cell, p, ns, min_cells):

    # Find which should be the maximum number of markers
    max_markers = np.min([max_markers, len(markers_representative)])
    
    # Start indices. Notice that here we assume that the minimum number of relevant markers is 2 (i.e. one marker alone can not define phenotypes)
    counter = 2
    index = 0
    mset = dict()
    
    # Resmax variable is a way to avoid going through all permutations, as if the grouping is not good, we shouldn't explore further permutations
    resmax = 0
    resmax_ = 0
    
    # Empty vectors to store results
    best_indices = []
    best_undefined = []
    best_cells = []
    best_matx = []
    best_maty = []
    maty_ = np.asarray([list(y_ns) for ljh in range(len(x_p))]).astype(float)
    matx_ = np.transpose(np.asarray([list(x_p) for ljh in range(len(y_ns))])).astype(float)
    
    # Slice the grid variables
    matx_ = matx_[:, y_ns>=ns]
    matx_ = matx_[x_p>=p, :]
    maty_ = maty_[:, y_ns>=ns]
    maty_ = maty_[x_p>=p, :]

    # Basically, we will keep permuting until no better solution can be found
    while True:
    
        # Cut the while loop if we reached the maximum number of clusters or if the possible solution using more markers will necessarily be worse than the current best
        if counter > max_markers or resmax > resmax_:
            break
        else:
            resmax = resmax_

        # For a given number of markers "counter", look at all possible marker combinations
        for g in ite.combinations(markers_representative,counter):
        
            # Slice data based on the current marker combination g
            markers_subset = np.asarray(list(g))
            mat_subset = mat_[:, np.isin(markers, markers_subset)]
            markers_subset = markers[np.isin(markers, markers_subset)]

            # This variable is probably unnecessary but it's not a big deal to keep it
            mset[index] = g

            # Find number of cells and undefined cells for a given marker combination across the p and ns grid
            results, undefined = cell_subdivision_counts(
                mat = mat_,
                mat_representative = mat_subset,
                markers = markers,
                markers_representative = markers_subset,
                batches = batches,
                samples = samples,
                marker_order = list(marker_order),
                three_marker = ["CD4"],
                p_min = x_p,
                s_min = y_ns)

            # Deep copy of the grid variables (potential for efficientcy)
            matx = copy.deepcopy(matx_)
            maty = copy.deepcopy(maty_)
            
            # Slice N phenotypes matrix given the conditions p and ns and calculate the maximum number of phenotypes for the marker combination g
            results = results[:, y_ns>=ns]
            results = results[x_p>=p, :]
            resmax_ = np.max(results)
            
            # Slice the undefined matrix as above
            undefined = undefined[:, y_ns>=ns]
            undefined = undefined[x_p>=p, :]
            
            # Further constrain matrix given the minimum number of phenotypes condition.
            condi = results<min_cells
            results[condi] = np.nan
            undefined[condi] = np.nan
            matx[condi] = np.nan
            maty[condi] = np.nan

            # If there are possible good solutions, store the results
            if np.any(np.isnan(results)==False):
            
                # The best solution will have the maximum number of new phenotypes...
                best_indices += [index]
                best_cells += [np.max(results[np.isnan(results)==False])]
                
                # ...and the minimum number of undefined cells...
                undefined[results != np.max(results[np.isnan(results)==False])] = np.nan
                best_undefined += [np.min(undefined[np.isnan(undefined)==False])]
                
                # and the maximum percentage within batch
                matx[undefined != np.min(undefined[np.isnan(undefined)==False])] = np.nan
                best_matx += [np.max(matx[np.isnan(matx)==False])]
                
                # and the maximum cells per sample
                maty[matx != np.max(matx[np.isnan(matx)==False])] = np.nan
                best_maty += [np.max(maty[np.isnan(maty)==False])]

            index += 1
        
        counter += 1
    
    best_cells=np.asarray(best_cells)
    best_undefined=np.asarray(best_undefined)
    best_indices=np.asarray(best_indices)
    best_matx=np.asarray(best_matx)
    best_maty=np.asarray(best_maty)
    
    if len(best_cells)>0:
        # Filter based on total number of new phenotypes
        best_indices = best_indices[best_cells ==  np.max(best_cells)]
        best_matx = best_matx[best_cells ==  np.max(best_cells)]
        best_maty = best_maty[best_cells ==  np.max(best_cells)]
        best_undefined = best_undefined[best_cells ==  np.max(best_cells)]
        best_cells = best_cells[best_cells ==  np.max(best_cells)]

    if len(best_indices)>1:
        i = np.where(best_undefined ==  np.min(best_undefined))[0]
                
        if len(i)>1:
            best_indices = best_indices[i]
            best_matx = best_matx[i]
            j = np.where(best_matx ==  np.max(best_matx))[0]
            return list(mset[best_indices[j[0]]])
        else:
            return list(mset[best_indices[i[0]]])
            
    elif len(best_indices)==0:
        return []
        
    else:
        return list(mset[best_indices[0]])
    

# Permute across positive and negative expression of the relevant markers, and identify new cell types
def cell_subdivision(mat, mat_representative, markers, markers_representative, batches, samples, marker_order,
                     three_marker=[], cell_name = "None", p_min=0.4,s_min=10):
    """Cell line subdivision

       Parameters
       ----------
       mat : ndarray
         A 2-D array expression matrix
         
       mat_representative : ndarray
         A 2-D array expression matrix of the representative markers

       markers : array
         A 1-D array with the markers in `mat` corresponding to each column

       markers_representative : array
         A 1-D array with the relevant markers in `mat_representative` corresponding to each column

       marker_order : list(str)
         List of markers used in the gating strategy ordered accordingly.

       three_marker : list(str)
         List of markers with potentially three peaks.
         
       p_min : int
         minimum proportion of samples for annotation to be considered
         
       random_state : int or None, (default=None)
          random seed.
    """
    
    # Let's trim down the cell classification to remove any protein that is not present (major cell types to renaming)
    major_cell_types_ = dict()
    for i in major_cell_types:
        major_cell_types_[i] = dict()
        for j in major_cell_types[i]:
            major_cell_types_[i][j] = list(np.asarray(major_cell_types[i][j])[np.isin(major_cell_types[i][j], markers)])
    
    # Now clean up redundancies in major_cell_types_
    number_of_proteins = np.array([len(v["positive"])+len(v["negative"]) for v in major_cell_types.values()])
    major_cell_types_clean = dict()
    for k,v in major_cell_types_.items():
        comparison = np.array([[(j==v)*1, i] for i, j in major_cell_types_.items()])
        condi = comparison[:,0].astype(bool)
        if np.sum(condi)==1:
            major_cell_types_clean[k] = v
        elif comparison[condi,1][np.argmin(number_of_proteins[condi])] == k:
            major_cell_types_clean[k] = v
        else:
            continue
            
    number_of_proteins = {k:(len(v["positive"])+len(v["negative"])) for k, v in major_cell_types.items()}
    
    # Find different groups of positive cells
    three_marker_ = np.asarray(three_marker)[np.isin(three_marker, markers_representative)]
    three_marker_low = [x + "_low" for x in three_marker_]
    three_markers = list(three_marker_)
    markers_representative_ = list(markers_representative) + list(three_marker_low)
    groups = [tuple(np.unique(i)) for i in ite.combinations_with_replacement(
        markers_representative_,len(markers_representative_))]
    groups = [tuple([])] + groups

    # Start dictionaries to store results
    cell_groups_renaming = {}
    cell_groups = {}
    cell_groups_ = {}
    repeated_names = {}
    
    # An array to define the clustering groups
    clustering_labels = np.zeros(np.shape(mat_representative)[0])-1
    
    # Initial values
    gdx = 0
    cell_groups[-1] = "other " + cell_name
    cell_groups_[-1] =  "other " + cell_name
    unidentified = 0
    cell_groups_renaming[-1] = set([])
    
    # Loop over unique groups
    for i in set(groups):
        iteration = np.unique(i)
        if np.any(np.isin(three_marker_[np.isin(three_marker_low, iteration)], iteration)):
            continue
        else:
            # The groups are defined by positive markers
            positives = np.asarray(markers_representative)[np.isin(markers_representative, iteration)]
            
            # The negatives should be defined easily
            negatives = np.asarray(markers_representative)[np.isin(markers_representative, iteration)==False]
            
            # The low positives are those markers that appear in three_marker_low
            lowpositives = np.asarray(three_marker_)[np.isin(three_marker_low, iteration)]
        
        # Figure out which cells fulfill these rules
        cells = select_cells(mat_representative,
                             markers_representative,
                             positive = positives,
                             negative = negatives,
                             lowpositive = lowpositives,
                             three_marker = three_markers)

        # If there are enough cells to consider them a cell type, go ahead and store it
        keep_cell_type = True
        for b in np.unique(batches):
            cells_ = cells[batches == b]
            samples_ = samples[batches == b]
            keep_cell_type_ = np.asarray([np.sum(samples_[cells_]==x)>s_min for x in np.unique(samples_)])
            keep_cell_type_ = np.sum(keep_cell_type_, axis=0)/float(len(np.unique(samples_)))
            keep_cell_type = keep_cell_type and keep_cell_type_>p_min

        if keep_cell_type:
            # To store it, let's find a name for it
            mat_avg = np.mean(mat[cells,:], axis=0)
            positives_ = markers[mat_avg>=3]
            negatives_ = markers[mat_avg<3]

            condi = [np.all(np.isin(x["negative"], list(negatives_))) and
                 np.all(np.isin(x["positive"], list(positives_)))
                 for x in major_cell_types_clean.values()]
            potential_name = np.asarray(list(major_cell_types_clean.keys()))[condi]

            if len(potential_name)==0:
                potential_name = cell_name
            elif len(potential_name) == 1:
                potential_name = potential_name[0]
            else:
                potential_name = potential_name[np.argmax(np.array([number_of_proteins[x] for x in potential_name]))]

            try:
                x = repeated_names[potential_name]
                repeated_names[potential_name] += 1
            except KeyError:
                repeated_names[potential_name] = 0

            x = repeated_names[potential_name]
            clustering_labels[cells] = int(gdx)
            if len(three_marker_) == 0:
                cell_groups_renaming[gdx] = set(list(map(lambda ls: ls+"+", positives)) + list(map(lambda ls: ls+"-", negatives)))
            else:
                cell_groups_renaming[gdx] = set(list(map(lambda ls: ls+"++", positives)) + list(map(lambda ls: ls+"-", negatives)) + list(map(lambda ls: ls+"+", lowpositives)))

            cell_groups_[gdx] = potential_name
            cell_groups[gdx] = potential_name + " " + str(int(x))
            cell_groups[gdx] = cell_groups[gdx] + " ({} cells)".format(np.sum(cells))
            gdx += 1
        else:
            unidentified += np.sum(cells)
    
    # It is useful to generate a dictionary with the clustering index and the different positive/negative sequence of markers
    cell_groups[-1] = cell_groups[-1] + " ({} cells)".format(unidentified)
    mat_average = mat[:, np.isin(markers, marker_order + list(markers_representative))]
    markers_average = markers[np.isin(markers, marker_order + list(markers_representative))]
        
    for repeated_x in repeated_names:
        x = find_set_differences({k: cell_groups_renaming[k] for k, v in cell_groups_.items() if v == repeated_x}, baseline_name="")
        for k in x.keys():
            cell_groups_[k] = (x[k] + " " + cell_groups_[k]).strip()

    cell_groups_renaming = find_set_differences(cell_groups_renaming)
    
    for x in cell_groups_renaming.keys():
        cell_groups_renaming[x] += " (" + cell_groups[x].split(" (")[1]
        cell_groups_[x] += " (" + cell_groups[x].split(" (")[1]
    
    if all(clustering_labels==-1):
        cell_groups_[-1] = cell_name

    return cell_groups_renaming, cell_groups_, clustering_labels, mat_average, markers_average

# Reclassify undefined cells with kNN classifier
def knn_classifier(mat_representative, clustering_labels, min_probability = 0.5):
    """Reclassify cells based on knn classifier

       Parameters
       ----------
       mat_representative : ndarray
         A 2-D array expression matrix of the representative markers

       clustering_labels : array
         clustering indices, where -1 is assigned to undefined nodes/cells

    """
    
    # It's common practice to standarize features
    undefined = clustering_labels == - 1

    if np.sum(undefined)>1 and len(np.unique(clustering_labels))>2:
        scaler = StandardScaler()
        mat_representative_scaled = scaler.fit_transform(mat_representative)

        # Start knn classifier
        clf = KNeighborsClassifier(n_neighbors=20)

        # Find cells that were classified and train the classifier
        mat_representative_scaled_train = mat_representative_scaled[undefined == False,:]
        y_train = clustering_labels[undefined == False]
        clf.fit(mat_representative_scaled_train, y_train)

        # Find cells that were not classified and predict label
        y_test = clf.predict_proba(mat_representative_scaled[undefined,:])

        # Figure out which cells should be left undefined and which ones should be reclassified
        # based on the model's ability to classify things with confidence (p>0.5)
        newclustering_label = np.argmax(y_test, axis=1)
        newclustering_label[np.sum(y_test > 0.5, axis=1)==0] = -1

        # Reclassify undefined cells
        clustering_labels[undefined] = newclustering_label
    
    return clustering_labels

# Full pipeline
def cell_identification(mat,
                        markers,
                        batches,
                        samples,
                        marker_order,
                        positive,
                        three_marker=[],
                        s_min=10, p_min=0.5,
                        max_midpoint_preselection = 15,
                        max_markers = 15,
                        min_annotations = 3, bimodality_selection_method = "midpoint",
                        mat_raw = None, cell_name = "None", knn_refine = True,
                        knn_min_probability = 0.5,
                        random_state = None):

    """Pipeline for automated gating, feature selection, and clustering to generate new annotations.

       Parameters
       ----------
       mat : ndarray
         A 2-D array expression matrix
         
       markers : array
         A 1-D array with the markers in `mat` corresponding to each column

       batches : array(str)
         A list of batch names per cell, useful for defining the thresholds for the new annotations.

       samples : array(str)
         A list of sample names per cell, useful for defining the thresholds for the new annotations.

       marker_order : list(str)
         List of markers used in the gating strategy, ordered accordingly.
         
       positive : list(bool)
         List indicating whether the markers in `marker_order` are positively (True) or negatively (False) expressed.

       three_marker : list(str)
         List of markers with potentially three peaks.

       s_min : float, (default=10)
         minimum number of cells within sample in 'p_min' % of samples within each batch for a new annotation to be considered.

       p_min : float, (default=0.5)
         minimum proportion of samples within batch with 's_min' cells for a new annotation to be considered.

       max_markers : int or None, (default=15)
          maximum number of relevant markers selected.
          
       bimodality_selection_method : str, (default = "midpoint")
         Two methods possible: 'DBSCAN', which uses the clustering method with the same name; and, 'midpoint', which basically uses those markers closer to the normalized matrix
        
       mat_raw : ndarray or None, (default=None)
         A 2-D array raw expression matrix

       cell_name : str, (default = "None")
         Base name for cell types (e.g. CD4 T-cells for 'CD4T')

       knn_refine : bool, (default=False)
         If True, the clustering done via permutations of relevant markers will be refined using a knn classifier

       knn_min_probability : float, (default=0.5)
         confidence threshold for the knn classifier to reassign new cell type
         
       random_state : int or None, (default=None)
         random seed.
    """
    
    ### MAIN GATING: we perform the gating for every batch independently. But that might be a bit of an overkill...
    
    markers_representative_batches = dict()
    truefalse = np.zeros(np.shape(mat)[0])!=0
    
    for kdx, k in enumerate(np.unique(batches)):
        batch = batches==k
        
        # if a raw matrix is provided, we use this for the main gating
        if mat_raw is None:
            truefalse_b = gaussian_gating(
                                 mat[batch,:],
                                 markers,
                                 marker_order = marker_order,
                                 positive = positive,
                                 random_state = random_state
                                 )
        else:
            truefalse_b = gaussian_gating(
                                 mat_raw[batch,:],
                                 markers,
                                 marker_order = marker_order,
                                 positive = positive,
                                 random_state = random_state
                                 )

        # Subset batch
        truefalse[batch] = truefalse_b
        mat_ = mat[batch,:]

        if bimodality_selection_method == "DBSCAN":

            eps_marker_clustering = np.sqrt((pairwise_distances(mat_.transpose(), metric="euclidean")**2)/float(np.sum(batch)))
            eps_marker_clustering = np.quantile(eps_marker_clustering[np.triu_indices(np.shape(eps_marker_clustering)[0], k=1)], q=0.05)

            # Subset cell types
            mat_ = mat_[truefalse_b,:]

            # Transpose matrix to perform marker clustering
            X = mat_.transpose()
            
            # Calculate normalized pairwise distance
            eps = np.sqrt((pairwise_distances(X, metric="euclidean")**2)/float(np.shape(X)[1]))

            # Calculate PCA space
            Xt = PCA(n_components=2).fit_transform(X)

            # Run DBSCAN
            km = DBSCAN(min_samples=1, eps=eps_marker_clustering, metric='precomputed', leaf_size=2).fit(eps)
            labels_ = km.labels_

            # Find the cluster center and generate an array of cluster by cell.
            cluster_centers_ = np.zeros((len(np.unique(labels_)), np.shape(mat_)[0]))
            for i in range(len(np.unique(labels_))):
                cluster_centers_[i, :] = np.mean(mat_[:,labels_ ==i], axis=1)
        
            # Generate marker expression per cluster matrix
            mark_exprr = np.zeros((len(np.unique(labels_)), len(markers)))
            for idx, i in enumerate(labels_):
                mark_exprr[i, idx] += 1

            # Remove known invariants and associated clustered markers
            known_invariants = markers[np.sum(mark_exprr[np.any(mark_exprr[:,np.isin(markers, marker_order)], axis=1),:], axis=0)>0]

            # Remove invariant markers based on the known invariants
            vt = VarianceThreshold().fit(mat_)
            invariants = markers[vt.variances_<=np.max(vt.variances_[np.isin(markers,marker_order)])]
            invariants = np.unique(list(invariants)+list(known_invariants))

            # For the variable markers, find closer to the cluster centroids. Slice matrix subsequently
            markers_representative = []
            for idx, i in enumerate(np.sum(mark_exprr, axis=1)):
                m_step = markers[mark_exprr[idx, :]>0]
                if i==1:
                    markers_representative += [m_step[0]]
                else:
                    closest, _ = pairwise_distances_argmin_min(cluster_centers_[idx, :][np.newaxis, :],
                                                               X[np.isin(markers, m_step),:])
                    markers_representative += [markers[np.isin(markers, m_step)][closest][0]]
                    
            markers_representative = np.asarray(markers_representative)[np.isin(markers_representative,invariants)==False]
            markers_representative = markers[np.isin(markers, markers_representative)]
            
            # Generate dictionary for relevant markers
            mdict = dict()
            for i in markers_representative:
                mdict[i] = markers[mark_exprr[labels_[markers==i][0], :] > 0]

            # Store dictionary for every batch
            markers_representative_batches[k] = mdict
            
        else:
        
            # Subset cell types
            mat_ = mat_[truefalse_b,:]

            # Check bimodality of the markers selected
            center_values = -np.abs(np.mean(mat_, axis=0)-3)
            max_values = np.sort(center_values)[::-1][max_midpoint_preselection]
            labels_ = center_values>max_values
            markers_representative = markers[np.isin(markers, markers[labels_])]

            # Remove invariant markers based on the known invariants
            vt = VarianceThreshold().fit(mat_)
            invariants = markers[vt.variances_<=np.max(vt.variances_[np.isin(markers,marker_order)])]
            markers_representative = np.asarray(markers_representative)[np.isin(markers_representative,invariants)==False]
            markers_representative = markers[np.isin(markers, markers_representative)]
            
            # Generate dictionary for relevant markers
            mdict = dict([(i,i) for i in markers_representative])

            # Store dictionary for every batch
            markers_representative_batches[k] = mdict

    ### SELECT RELEVANT MARKERS BASED ON SELECTIONS ACROSS BATCHES AND SLICE DATA
    
    # Filter markers with appearences in all batches
    m, c = np.unique(list(ite.chain.from_iterable(
        [list(markers_representative_batches[i].keys()) for idx, i in enumerate(markers_representative_batches.keys())])),
        return_counts = True)
    markers_representative = m[c == len(markers_representative_batches.keys())]

    # Merge all batches together and extract main cell type
    mat_ = mat[truefalse, :]
    samples_ = samples[truefalse]
    batches_ = batches[truefalse]
    
    # Slice matrix and markers using the selected markers through the main gating
    mat_representative = mat_[:, np.isin(markers, markers_representative)]
    markers_representative = markers[np.isin(markers, markers_representative)]

    ### STUDY SUBSETS OF MARKERS: Go over every combination of markers and understand the resulting number of cell types and unidentified cells
    x_p = np.linspace(0,1,101)
    y_ns = np.arange(101)
    
    markers_representative_ = check_all_subsets(max_markers = max_markers,
                                                x_p = x_p,
                                                y_ns = y_ns,
                                                mat_ = mat_,
                                                mat_representative = mat_representative,
                                                markers = markers,
                                                markers_representative = markers_representative,
                                                marker_order = marker_order,
                                                batches = batches_,
                                                samples = samples_,
                                                cell = cell_name,
                                                ns = s_min, p = p_min,
                                                min_cells = min_annotations)

    if len(markers_representative_)>0:
        markers_representative = markers[np.isin(markers, markers_representative_)]
        mat_representative = mat_[:, np.isin(markers, markers_representative_)]
        
        # Now let's figure out which groups of markers form relevant cell types
        cell_groups, cell_groups_name, clustering_labels, mat_average, markers_average = cell_subdivision(
                        mat = mat_,
                        mat_representative = mat_representative,
                        markers = markers,
                        markers_representative = markers_representative,
                        batches = batches_,
                        samples = samples_,
                        marker_order = marker_order,
                        three_marker = three_marker,
                        p_min = p_min,
                        s_min = s_min,
                        cell_name = cell_name)

        # Try to classify undefined cells using a knn classifier
        if knn_refine:
            clustering_labels = knn_classifier(mat_representative, clustering_labels, min_probability = knn_min_probability)

        return truefalse, cell_groups_name, clustering_labels, markers_representative_batches, markers_representative #, dfdata
    else:
        return truefalse, {-1 : cell_name}, np.zeros(np.sum(truefalse))-1, markers_representative_batches, []
        
# Full pipeline
def identify_phenotypes(truefalse,
                        mat,
                        markers,
                        batches,
                        samples,
                        s_min=10, p_min=0.5,
                        max_midpoint_preselection = 15,
                        max_markers = 15,
                        min_annotations = 3, bimodality_selection_method = "midpoint",
                        cell_name = "None", knn_refine = True,
                        knn_min_probability = 0.5,
                        random_state = None):
    """Pipeline for automated gating, feature selection, and clustering to generate new annotations.

       Parameters
       ----------
       mat : ndarray
         A 2-D array expression matrix
         
       markers : array
         A 1-D array with the markers in `mat` corresponding to each column

       batches : array(str)
         A list of batch names per cell, useful for defining the thresholds for the new annotations.

       samples : array(str)
         A list of sample names per cell, useful for defining the thresholds for the new annotations.

       marker_order : list(str)
         List of markers used in the gating strategy, ordered accordingly.
         
       positive : list(bool)
         List indicating whether the markers in `marker_order` are positively (True) or negatively (False) expressed.

       three_marker : list(str)
         List of markers with potentially three peaks.

       s_min : float, (default=10)
         minimum number of cells within sample in 'p_min' % of samples within each batch for a new annotation to be considered.

       p_min : float, (default=0.5)
         minimum proportion of samples within batch with 's_min' cells for a new annotation to be considered.

       max_markers : int or None, (default=15)
          maximum number of relevant markers selected.
          
       bimodality_selection_method : str, (default = "midpoint")
         Two methods possible: 'DBSCAN', which uses the clustering method with the same name; and, 'midpoint', which basically uses those markers closer to the normalized matrix
        
       mat_raw : ndarray or None, (default=None)
         A 2-D array raw expression matrix

       cell_name : str, (default = "None")
         Base name for cell types (e.g. CD4 T-cells for 'CD4T')

       knn_refine : bool, (default=False)
         If True, the clustering done via permutations of relevant markers will be refined using a knn classifier

       knn_min_probability : float, (default=0.5)
         confidence threshold for the knn classifier to reassign new cell type
         
       random_state : int or None, (default=None)
         random seed.
    """
    
    ### MAIN GATING: we perform the gating for every batch independently. But that might be a bit of an overkill...
    
    
    for kdx, k in enumerate(np.unique(batches)):
        batch = batches==k

        # Subset batch
        truefalse_b = truefalse[batch]
        mat_ = mat[batch,:]

        if bimodality_selection_method == "DBSCAN":

            eps_marker_clustering = np.sqrt((pairwise_distances(mat_.transpose(), metric="euclidean")**2)/float(np.sum(batch)))
            eps_marker_clustering = np.quantile(eps_marker_clustering[np.triu_indices(np.shape(eps_marker_clustering)[0], k=1)], q=0.05)

            # Subset cell types
            mat_ = mat_[truefalse_b,:]

            # Transpose matrix to perform marker clustering
            X = mat_.transpose()
            
            # Calculate normalized pairwise distance
            eps = np.sqrt((pairwise_distances(X, metric="euclidean")**2)/float(np.shape(X)[1]))

            # Calculate PCA space
            Xt = PCA(n_components=2).fit_transform(X)

            # Run DBSCAN
            km = DBSCAN(min_samples=1, eps=eps_marker_clustering, metric='precomputed', leaf_size=2).fit(eps)
            labels_ = km.labels_

            # Find the cluster center and generate an array of cluster by cell.
            cluster_centers_ = np.zeros((len(np.unique(labels_)), np.shape(mat_)[0]))
            for i in range(len(np.unique(labels_))):
                cluster_centers_[i, :] = np.mean(mat_[:,labels_ ==i], axis=1)
        
            # Generate marker expression per cluster matrix
            mark_exprr = np.zeros((len(np.unique(labels_)), len(markers)))
            for idx, i in enumerate(labels_):
                mark_exprr[i, idx] += 1

            # Remove known invariants and associated clustered markers
            known_invariants = markers[np.sum(mark_exprr[np.any(mark_exprr[:,np.isin(markers, marker_order)], axis=1),:], axis=0)>0]

            # Remove invariant markers based on the known invariants
            vt = VarianceThreshold().fit(mat_)
            invariants = markers[vt.variances_<=np.max(vt.variances_[np.isin(markers,marker_order)])]
            invariants = np.unique(list(invariants)+list(known_invariants))

            # For the variable markers, find closer to the cluster centroids. Slice matrix subsequently
            markers_representative = []
            for idx, i in enumerate(np.sum(mark_exprr, axis=1)):
                m_step = markers[mark_exprr[idx, :]>0]
                if i==1:
                    markers_representative += [m_step[0]]
                else:
                    closest, _ = pairwise_distances_argmin_min(cluster_centers_[idx, :][np.newaxis, :],
                                                               X[np.isin(markers, m_step),:])
                    markers_representative += [markers[np.isin(markers, m_step)][closest][0]]
                    
            markers_representative = np.asarray(markers_representative)[np.isin(markers_representative,invariants)==False]
            markers_representative = markers[np.isin(markers, markers_representative)]
            
            # Generate dictionary for relevant markers
            mdict = dict()
            for i in markers_representative:
                mdict[i] = markers[mark_exprr[labels_[markers==i][0], :] > 0]

            # Store dictionary for every batch
            markers_representative_batches[k] = mdict
            
        else:
        
            # Subset cell types
            mat_ = mat_[truefalse_b,:]

            # Check bimodality of the markers selected
            center_values = -np.abs(np.mean(mat_, axis=0)-3)
            max_values = np.sort(center_values)[::-1][max_midpoint_preselection]
            labels_ = center_values>max_values
            markers_representative = markers[np.isin(markers, markers[labels_])]

            # Remove invariant markers based on the known invariants
            vt = VarianceThreshold().fit(mat_)
            invariants = markers[vt.variances_<=np.max(vt.variances_[np.isin(markers,marker_order)])]
            markers_representative = np.asarray(markers_representative)[np.isin(markers_representative,invariants)==False]
            markers_representative = markers[np.isin(markers, markers_representative)]
            
            # Generate dictionary for relevant markers
            mdict = dict([(i,i) for i in markers_representative])

            # Store dictionary for every batch
            markers_representative_batches[k] = mdict

    ### SELECT RELEVANT MARKERS BASED ON SELECTIONS ACROSS BATCHES AND SLICE DATA
    
    # Filter markers with appearences in all batches
    m, c = np.unique(list(ite.chain.from_iterable(
        [list(markers_representative_batches[i].keys()) for idx, i in enumerate(markers_representative_batches.keys())])),
        return_counts = True)
    markers_representative = m[c == len(markers_representative_batches.keys())]

    # Merge all batches together and extract main cell type
    mat_ = mat[truefalse, :]
    samples_ = samples[truefalse]
    batches_ = batches[truefalse]
    
    # Slice matrix and markers using the selected markers through the main gating
    mat_representative = mat_[:, np.isin(markers, markers_representative)]
    markers_representative = markers[np.isin(markers, markers_representative)]

    ### STUDY SUBSETS OF MARKERS: Go over every combination of markers and understand the resulting number of cell types and unidentified cells
    x_p = np.linspace(0,1,101)
    y_ns = np.arange(101)
    
    markers_representative_ = check_all_subsets(max_markers = max_markers,
                                                x_p = x_p,
                                                y_ns = y_ns,
                                                mat_ = mat_,
                                                mat_representative = mat_representative,
                                                markers = markers,
                                                markers_representative = markers_representative,
                                                marker_order = marker_order,
                                                batches = batches_,
                                                samples = samples_,
                                                cell = cell_name,
                                                ns = s_min, p = p_min,
                                                min_cells = min_annotations)

    if len(markers_representative_)>0:
        markers_representative = markers[np.isin(markers, markers_representative_)]
        mat_representative = mat_[:, np.isin(markers, markers_representative_)]
        
        # Now let's figure out which groups of markers form relevant cell types
        cell_groups, cell_groups_name, clustering_labels, mat_average, markers_average = cell_subdivision(
                        mat = mat_,
                        mat_representative = mat_representative,
                        markers = markers,
                        markers_representative = markers_representative,
                        batches = batches_,
                        samples = samples_,
                        marker_order = marker_order,
                        three_marker = three_marker,
                        p_min = p_min,
                        s_min = s_min,
                        cell_name = cell_name)

        # Try to classify undefined cells using a knn classifier
        if knn_refine:
            clustering_labels = knn_classifier(mat_representative, clustering_labels, min_probability = knn_min_probability)

        return truefalse, cell_groups_name, clustering_labels, markers_representative_batches, markers_representative #, dfdata
    else:
        return truefalse, {-1 : cell_name}, np.zeros(np.sum(truefalse))-1, markers_representative_batches, []

def annotate(mat,
            markers,
            batches,
            samples,
            labels,
            min_cellxsample = 10,
            percent_samplesxbatch = 0.5,
            max_midpoint_preselection = 15,
            max_markers = 15,
            min_annotations = 3,
            bimodality_selection_method = "midpoint",
            knn_refine = True,
            knn_min_probability = 0.5,
            min_annotations = 3
            random_state = None):
            
    """Pipeline for automated gating, feature selection, and clustering to generate new annotations.

       Parameters
       ----------
       mat : ndarray
         A 2-D numpy array expression matrix
         
       markers : array
         A 1-D numpy array with the markers in `mat` corresponding to each column

       labels : array(str)
         A 1-D numpy array with the main cell labels.

       batches : array(str)
         A 1-D numpy array with batch names per cell.

       samples : array(str)
         A 1-D numpy array with sample names per cell.
    
       min_cellxsample : float, (default=10)
         minimum number of cells within sample in 'p_min' % of samples within each batch for a new annotation to be considered.

       percent_samplesxbatch : float, (default=0.5)
         minimum proportion of samples within batch with 's_min' cells for a new annotation to be considered.

       max_markers : int or None, (default=15)
         maximum number of relevant markers selected.
          
       bimodality_selection_method : str, (default = "midpoint")
         Two methods possible: 'DBSCAN', which uses the clustering method with the same name; and, 'midpoint', which basically uses those markers closer to the normalized matrix

       knn_refine : bool, (default=False)
         If True, the clustering done via permutations of relevant markers will be refined using a knn classifier

       knn_min_probability : float, (default=0.5)
         confidence threshold for the knn classifier to reassign new cell type
         
       random_state : int or None, (default=None)
         random seed.
    """

    annotations = np.asarray(["undefined"]*len(labels)).astype('U100')

    for idx, i in enumerate(np.unique(labels)):
        truefalse = labels == i
        cell_groups, clustering_labels, mdictA, fmarker = identify_phenotypes(
                            truefalse = truefalse,
                            mat = mat,
                            markers = markers,
                            batches = batches,
                            samples = samples,
                            p_min = percent_samplesxbatch,
                            s_min = min_cellxsample,
                            max_midpoint_preselection = max_midpoint_preselection,
                            max_markers = max_markers,
                            min_annotations = min_annotations,
                            bimodality_selection_method = bimodality_selection_method,
                            random_state = random_state,
                            knn_refine = knn_refine,
                            cell_name = i)

        cell_dict = dict([tuple([x,cell_groups[x].split(" (")[0]]) for x in cell_groups])
        annotations[truefalse] = np.vectorize(cell_dict.get)(clustering_labels)
