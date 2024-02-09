"""
Script to generate test datasets for cellxhaustive package
"""

# Import utility modules
import argparse
import copy as cp
import json
import numpy as np
import pandas as pd
import string
from scipy.stats import truncnorm


# Parse arguments
parser = argparse.ArgumentParser(description='Script to generate test datasets for cellxhaustive package')
parser.add_argument('-n', '--nb-cell', dest='nb_cell', type=int,
                    help='Number of cells to generate [500]',
                    required=False, default=500)
parser.add_argument('-m', '--nb-marker', dest='nb_marker', type=int,
                    help='Number of markers to generate [5]',
                    required=False, default=5)
parser.add_argument('-b', '--batch', dest='nb_batch', type=int,
                    help='Number of batches to define [1]',
                    required=False, default=1)
parser.add_argument('-s', '--sample', dest='nb_sample', type=int,
                    help='Number of samples to define [1]',
                    required=False, default=1)
parser.add_argument('-c', '--cell-type', dest='cell_type', type=str,
                    help='Path to file with cell types ontology [cell_types.json]',
                    required=False, default='cell_types.json')
parser.add_argument('-pmin', '--pos-min', dest='pos_min', type=float,
                    help='Minimum expression value in positive distribution [3]',
                    required=False, default=3)
parser.add_argument('-pmax', '--pos-max', dest='pos_max', type=float,
                    help='Maximum expression value in positive distribution [6]',
                    required=False, default=6)
parser.add_argument('-pmean', '--pos-mean', dest='pos_mean', type=float,
                    help='Mean expression in positive distribution [(pos_max+pos_min)/2]',
                    required=False)
parser.add_argument('-pstd', '--pos-std', dest='pos_std', type=float,
                    help='Expression standard deviation in positive distribution [1]',
                    required=False, default=1)
parser.add_argument('-nmin', '--neg-min', dest='neg_min', type=float,
                    help='Minimum expression value in negative distribution [0]',
                    required=False, default=0)
parser.add_argument('-nmax', '--neg-max', dest='neg_max', type=float,
                    help='Maximum expression value in negative distribution [3]',
                    required=False, default=3)
parser.add_argument('-nmean', '--neg-mean', dest='neg_mean', type=float,
                    help='Mean expression in negative distribution [(neg_max+neg_min)/2]',
                    required=False)
parser.add_argument('-nstd', '--neg-std', dest='neg_std', type=float,
                    help='Expression standard deviation in negative distribution [1]',
                    required=False, default=1)
args = parser.parse_args()


# Function to add missing markers in full phenotype
def get_full_phenotype(rng, tot_markers, cell_dict, idx):
    pheno = cp.deepcopy(cell_dict[idx])
    for marker in tot_markers:
        if marker not in ''.join(pheno):
            status = rng.choice(['+', '-'], p=[0.5, 0.5])
            marker_state = f'{marker}{status}'
            pheno.append(marker_state)
    pheno.sort()  # Sort all markers in alphabetical order
    return pheno


# Function to get expression value from a distribution depending on marker sign
def get_expression(rng, pos_dis, neg_dis, sign):
    if sign == '+':
        expression = pos_dis.rvs(size=1, random_state=rng)
    else:
        expression = neg_dis.rvs(size=1, random_state=rng)
    return expression


# Vectorised version of get_expression()
vec_get_expression = np.vectorize(get_expression, excluded=['rng', 'pos_dis', 'neg_dis'])


# Main script execution
if __name__ == '__main__':

    # Get parameter values from argument parsing
    nb_cell = args.nb_cell  # Total number of cells
    nb_marker = args.nb_marker
    tot_markers = list(string.ascii_lowercase)[:nb_marker]  # Use lowercase alphabet as markers

    # Create numpy random generator with seed for reproducibility
    rng = np.random.default_rng(42)

    # Import cell types ontology
    with open(args.cell_type) as in_cell_types:
        cell_types_dict = json.load(in_cell_types)

    # Convert dictionary keys to int
    cell_types_dict = {int(k): v for k, v in cell_types_dict.items()}

    # Create dictionary with strings instead of lists
    cell_types_str = {k: '/'.join(v) for k, v in cell_types_dict.items()}

    # Randomly pick cell types with an equal proportion
    nb_cell_types = len(cell_types_dict)
    proba = np.full(nb_cell_types, (1 / nb_cell_types))
    cell_type = rng.choice(nb_cell_types, size=nb_cell, p=proba)

    # Get cell ontology phenotypes
    cell_phntp_onto = np.vectorize(cell_types_str.get)(cell_type)

    # Get full cell phenotypes: start with cell ontology and add the missing markers
    # with a random state
    cell_phntp_full = []
    cell_phntp_full_str = []
    for idx in cell_type:
        mker = get_full_phenotype(rng, tot_markers, cell_types_dict, idx)
        cell_phntp_full.append(mker)
        cell_phntp_full_str.append('/'.join(mker))

    # Get batch data
    nb_batch = args.nb_batch
    prob_batch = np.full(nb_batch, (1 / nb_batch))
    batch_name = [f'batch{i}' for i in range(nb_batch)]
    batch = rng.choice(batch_name, size=nb_cell, p=prob_batch)

    # Get sample data
    nb_sample = args.nb_sample
    prob_sample = np.full(nb_sample, (1 / nb_sample))
    sample_name = [f'sample{i}' for i in range(nb_sample)]
    sample = rng.choice(sample_name, size=nb_cell, p=prob_sample)

    # Initialise final dictionary
    tot_dict = {'cell_type': cell_type,
                'cell_phntp_onto': cell_phntp_onto,
                'cell_phntp_full': cell_phntp_full_str,
                'batch': batch,
                'sample': sample}

    # Create 2 distributions for positive marker expression and negative marker expression
    # Distribution will be truncated normal distributions to fit ADT normalised values
    # Positive distribution
    pos_max = args.pos_max
    pos_min = args.pos_min
    if not args.pos_mean:
        pos_mean = (pos_max + pos_min) / 2
    else:
        pos_mean = args.pos_mean
    pos_std = args.pos_std
    pos_min_dis = (pos_min - pos_mean) / pos_std
    pos_max_dis = (pos_max - pos_mean) / pos_std
    pos_dis = truncnorm(pos_min_dis, pos_max_dis, loc=pos_mean, scale=pos_std)

    # Negative distribution
    neg_max = args.neg_max
    neg_min = args.neg_min
    if not args.neg_mean:
        neg_mean = (neg_max + neg_min) / 2
    else:
        neg_mean = args.neg_mean
    neg_std = args.neg_std
    neg_min_dis = (neg_min - neg_mean) / neg_std
    neg_max_dis = (neg_max - neg_mean) / neg_std
    neg_dis = truncnorm(neg_min_dis, neg_max_dis, loc=neg_mean, scale=neg_std)

    # Generate expression values for all markers
    exp_dict = {}
    for mker_idx in range(len(tot_markers)):
        mker_lst = [phntp[mker_idx] for phntp in cell_phntp_full]  # Extract marker for all cells
        mker_lst = [mk.replace(tot_markers[mker_idx], '') for mk in mker_lst]  # Remove letter
        mker_exp = list(vec_get_expression(rng, pos_dis, neg_dis, mker_lst))  # Get expression
        exp_dict[tot_markers[mker_idx]] = mker_exp

    # Update dictionary with expression dictionary
    tot_dict |= exp_dict

    # Convert dictionary to dataframe
    final_table = pd.DataFrame.from_dict(tot_dict)
    final_table.index.names = ['cell_nb']  # Rename index

    # Build output name
    output = f'cell_expression_{nb_cell}cells_{nb_marker}mkers_{nb_sample}samples_{nb_batch}batches'
    output += f'_pmin{pos_min}_pmax{pos_max}_pmean{pos_mean}_pstd{pos_std}'
    output += f'_nmin{neg_min}_nmax{neg_max}_nmean{neg_mean}_nstd{neg_std}.tsv'

    # Save table to file
    final_table.to_csv(output, sep='\t', header=True, index=True)
    print(f'Generated dataset is {output}')


# Notes: potential conflict and assignation problem between cell types 0, 1 and 2?


# # To plot distrib
# import matplotlib.pyplot as plt
# pos_dis = truncnorm(-1.5, 1.5, loc=4.5, scale=2)
# neg_dis = truncnorm(-1.5, 1.5, loc=1.5, scale=2)
# x_range = np.linspace(-0, 10, 1000)
# # plt.plot(x_range, truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std))
# plt.plot(x_range, pos_dis.pdf(x_range))
# plt.show()
# plt.plot(x_range, neg_dis.pdf(x_range))
# plt.show()


# Check whether theoretical distribution (truncated normal distribution) fit
# actual marker expression distribution


# Datasets to check:
# 500 cells, 5 markers, 1 batch, 1 sample, equal cell type distribution
# 500 cells, 5 markers, several batches, 1 sample, equal cell type distribution
# 500 cells, 5 markers, 1 batch, several samples, equal cell type distribution
# 500 cells, 5 markers, several batches, several sample, equal cell type distribution
# 10 markers?
# Non-equal cell type distribution?
# Check impact of limiting the number of markers. For example, if mam-markers = 3, do we see cell types with more than 3 markers?


# Test presence/absence of batch, sample, cell_type
