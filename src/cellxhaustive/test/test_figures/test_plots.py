"""
Script to generate plots for cellxhaustive test datasets analysis
"""

# Import utility modules
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score


# Function to add jittering on scatterplots
def jitter(val, j):
    return val + np.random.normal(j, 0.03, len(val))


# Create numpy random generator with seed for reproducibility
np.random.seed(42)

# Plot AMI across std
# Initialise objects
markers = ['a', 'b', 'c', 'd', 'e']  # Markers list
std_dir = 'default_std_test/'  # std folder
std_files = [f for f in os.listdir(std_dir) if f.endswith('.tsv')]  # std tsv files
std_rows = []  # Data list
std_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in std_files:
    # Get std value
    std = float(file.replace('cell_expression_5000cells_std', '').replace('_default_annotated.tsv', ''))
    # Build path
    std_file = os.path.join(std_dir, file)
    # Load file
    std_res = pd.read_csv(std_file, sep='\t', index_col=0)
    # Get true cell types
    std_labels_true = std_res['cell_type']
    # Get expression data
    std_exp = std_res.loc[:, markers]
    std_exp['std'] = std
    # Add it to expression table
    std_exp_df = pd.concat([std_exp_df, std_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    std_labels_col = [col for col in std_res.columns if 'KNN_annot' in col]
    for annot in std_labels_col:
        std_labels_pred = std_res[annot]
        score = adjusted_mutual_info_score(std_labels_true, std_labels_pred)
        std_rows.append({'std': std, 'score': score})

# Build final AMI dataframe, sort by std and reset indices
df_std = pd.DataFrame(std_rows).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_std = df_std.groupby('std', as_index=False)['score'].mean()

# Melt expression table
std_exp_df_melted = pd.melt(std_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=df_std['std'], y=df_std['score'], color='blue', ax=ax[0])
g = sns.scatterplot(x=jitter(avg_std['std'], 0.05), y=avg_std['score'], color='red', ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI', ylim=[0, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI'])
ax[0].title.set_text('Adjusted Mutual Information (AMI) score')

# Expression density plot
h = sns.kdeplot(data=std_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of standard deviation on AMI')
fig.tight_layout()
fig.figure.savefig(os.path.join(std_dir, 'std_AMI_expression.jpg'))







# Datasets to check:
# Default = 5000 cells, 5 markers, 1 batch, 1 sample, min = 0, max = 6, nmean = 1.5, pmean = 4.5, std = 0.75, all non-defining markers negative
    # cell_expression_5000cells_5mkers_1samples_1batches_min0_max6_nmean1.5_pmean4.5_std0.75_default
    # Varying std

# 1. Default but non-defining markers both positive and negative
    # cell_expression_5000cells_5mkers_1samples_1batches_min0_max6_nmean1.5_pmean4.5_std0.75_mixed
# 2.
# 3.
# 4.
# 5.
# 6.
# 7.
# 8.

# Check whether theoretical distribution (truncated normal distribution) fit
# actual marker expression distribution

# Check potential conflict and assignation problem between cell types 0 vs 1 and 0 vs 2? (0 is included both in 1 and 2)



# Same but with non-defining markers positive or negative across all cells


# 1000 cells, 5 markers, several batches, 1 sample, equal cell type distribution
# 1000 cells, 5 markers, 1 batch, several samples, equal cell type distribution

# 1000 cells, 5 markers, several batches, several sample, equal cell type distribution

# mutual information criteria, Jaccard index? for cell type identification
# Average marker phenotype overlap (overlap between annotated phenotype and actual phenotype) across all cells?

# Plot over std (x), MIC (y)
# Plot over batches (x), MIC (y) boxscore
# Use xmin param

# 4 batches, 10 samples

# Average score when there are several combinations


# 10 markers?
# Check impact of limiting the number of markers. For example, if max-markers = 3, do we see cell types with more than 3 markers?


# Test presence/absence of batch, sample, cell_type

