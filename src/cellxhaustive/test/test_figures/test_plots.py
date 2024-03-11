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
from sklearn.metrics import jaccard_score


# Function to add jittering on scatterplots
def jitter(val, loc, scale):
    return val + np.random.normal(loc, scale, len(val))


# Initialise common objects
np.random.seed(42)  # Numpy random generator with seed for reproducibility
markers = ['a', 'b', 'c', 'd', 'e']  # Markers list

# Plot AMI/Jaccard similarity across std - only negative markers
# Initialise objects
std_dir = '../test_results/default_fake_std_test/'  # Folder
std_files = [f for f in os.listdir(std_dir) if f.endswith('.tsv')]  # tsv files
std_rows_ami = []  # AMI data list
std_rows_jaccard = []  # Jaccard data list
std_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in std_files:
    # Get std value
    std = float(file.replace('default_cell_expression_fake_std', '').replace('_annotated.tsv', ''))
    # Build path
    std_file = os.path.join(std_dir, file)
    # Load file
    std_res = pd.read_csv(std_file, sep='\t', index_col=0)
    # Get true cell types
    std_labels_true = std_res['cell_phntp_full']
    # Get expression data
    std_exp = std_res.loc[:, markers]
    std_exp['std'] = std
    # Add it to expression table
    std_exp_df = pd.concat([std_exp_df, std_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    std_labels_col = [col for col in std_res.columns if 'Phenotypes_' in col]
    for annot in std_labels_col:
        std_labels_pred = std_res[annot]
        std_score_ami = adjusted_mutual_info_score(std_labels_true, std_labels_pred)
        std_rows_ami.append({'std': std, 'std_score_ami': std_score_ami})
        std_score_jaccard = jaccard_score(std_labels_true, std_labels_pred, average='weighted')
        std_rows_jaccard.append({'std': std, 'std_score_jaccard': std_score_jaccard})

# Build final AMI dataframe, sort by std and reset indices
df_std_ami = pd.DataFrame(std_rows_ami).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_std_ami = df_std_ami.groupby('std', as_index=False)['std_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_std_jaccard = pd.DataFrame(std_rows_jaccard).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_std_jaccard = df_std_jaccard.groupby('std', as_index=False)['std_score_jaccard'].mean()

# Melt expression table
std_exp_df_melted = pd.melt(std_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_std_ami['std'], 0.05, 0.03), y=df_std_ami['std_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_std_ami['std'], 0.05, 0.03), y=avg_std_ami['std_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_std_jaccard['std'], 0.05, 0.03), y=df_std_jaccard['std_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_std_jaccard['std'], 0.05, 0.03), y=avg_std_jaccard['std_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=std_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_fake_std.jpg', dpi=600)
plt.close()
print('Saved default_fake_std.jpg')


# Plot AMI/Jaccard similarity across std with 4 batches - only negative markers
# Initialise objects
batch_std_dir = '../test_results/default_batch_std_test/'  # Folder
batch_std_files = [f for f in os.listdir(batch_std_dir) if f.endswith('.tsv')]  # tsv files
batch_std_rows_ami = []  # AMI data list
batch_std_rows_jaccard = []  # Jaccard data list
batch_std_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_std_files:
    # Get std value
    std = float(file.replace('default_cell_expression_batch_std', '').replace('_annotated.tsv', ''))
    # Build path
    batch_std_file = os.path.join(batch_std_dir, file)
    # Load file
    batch_std_res = pd.read_csv(batch_std_file, sep='\t', index_col=0)
    # Get true cell types
    batch_std_labels_true = batch_std_res['cell_phntp_full']
    # Get expression data
    batch_std_exp = batch_std_res.loc[:, markers]
    batch_std_exp['std'] = std
    # Add it to expression table
    batch_std_exp_df = pd.concat([batch_std_exp_df, batch_std_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_std_labels_col = [col for col in batch_std_res.columns if 'Phenotypes_' in col]
    for annot in batch_std_labels_col:
        batch_std_labels_pred = batch_std_res[annot]
        batch_std_score_ami = adjusted_mutual_info_score(batch_std_labels_true, batch_std_labels_pred)
        batch_std_rows_ami.append({'std': std, 'batch_std_score_ami': batch_std_score_ami})
        batch_std_score_jaccard = jaccard_score(batch_std_labels_true, batch_std_labels_pred, average='weighted')
        batch_std_rows_jaccard.append({'std': std, 'batch_std_score_jaccard': batch_std_score_jaccard})

# Build final AMI dataframe, sort by std and reset indices
df_batch_std_ami = pd.DataFrame(batch_std_rows_ami).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_std_ami = df_batch_std_ami.groupby('std', as_index=False)['batch_std_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_batch_std_jaccard = pd.DataFrame(batch_std_rows_jaccard).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_std_jaccard = df_batch_std_jaccard.groupby('std', as_index=False)['batch_std_score_jaccard'].mean()

# Melt expression table
batch_std_exp_df_melted = pd.melt(batch_std_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_std_ami['std'], 0.05, 0.03), y=df_batch_std_ami['batch_std_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_std_ami['std'], 0.05, 0.03), y=avg_batch_std_ami['batch_std_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_std_jaccard['std'], 0.05, 0.03), y=df_batch_std_jaccard['batch_std_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_std_jaccard['std'], 0.05, 0.03), y=avg_batch_std_jaccard['batch_std_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_std_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation and increased batch number on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_batch_std.jpg', dpi=600)
plt.close()
print('Saved default_batch_std.jpg')


# Plot AMI/Jaccard similarity across std with 4 samples - only negative markers
# Initialise objects
sample_dir = '../test_results/default_sample_std_test/'  # Folder
sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.tsv')]  # tsv files
sample_rows_ami = []  # AMI data list
sample_rows_jaccard = []  # Jaccard data list
sample_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in sample_files:
    # Get std value
    std = float(file.replace('default_cell_expression_sample_std', '').replace('_annotated.tsv', ''))
    # Build path
    sample_file = os.path.join(sample_dir, file)
    # Load file
    sample_res = pd.read_csv(sample_file, sep='\t', index_col=0)
    # Get true cell types
    sample_labels_true = sample_res['cell_phntp_full']
    # Get expression data
    sample_exp = sample_res.loc[:, markers]
    sample_exp['std'] = std
    # Add it to expression table
    sample_exp_df = pd.concat([sample_exp_df, sample_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    sample_labels_col = [col for col in sample_res.columns if 'Phenotypes_' in col]
    for annot in sample_labels_col:
        sample_labels_pred = sample_res[annot]
        sample_score_ami = adjusted_mutual_info_score(sample_labels_true, sample_labels_pred)
        sample_rows_ami.append({'std': std, 'sample_score_ami': sample_score_ami})
        sample_score_jaccard = jaccard_score(sample_labels_true, sample_labels_pred, average='weighted')
        sample_rows_jaccard.append({'std': std, 'sample_score_jaccard': sample_score_jaccard})

# Build final AMI dataframe, sort by std and reset indices
df_sample_ami = pd.DataFrame(sample_rows_ami).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_ami = df_sample_ami.groupby('std', as_index=False)['sample_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_sample_jaccard = pd.DataFrame(sample_rows_jaccard).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_jaccard = df_sample_jaccard.groupby('std', as_index=False)['sample_score_jaccard'].mean()

# Melt expression table
sample_exp_df_melted = pd.melt(sample_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_sample_ami['std'], 0.05, 0.03), y=df_sample_ami['sample_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_ami['std'], 0.05, 0.03), y=avg_sample_ami['sample_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_sample_jaccard['std'], 0.05, 0.03), y=df_sample_jaccard['sample_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_jaccard['std'], 0.05, 0.03), y=avg_sample_jaccard['sample_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=sample_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation and increased sample number on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_sample_std.jpg', dpi=600)
plt.close()
print('Saved default_sample_std.jpg')


# Plot AMI/Jaccard similarity across std with batches and samples - only negative markers
# Initialise objects
batch_sample_dir = '../test_results/default_batch_sample_std_test/'  # Folder
batch_sample_files = [f for f in os.listdir(batch_sample_dir) if f.endswith('.tsv')]  # tsv files
batch_sample_rows_ami = []  # AMI data list
batch_sample_rows_jaccard = []  # Jaccard data list
batch_sample_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_sample_files:
    # Get std value
    std = float(file.replace('default_cell_expression_batch_sample_std', '').replace('_annotated.tsv', ''))
    # Build path
    batch_sample_file = os.path.join(batch_sample_dir, file)
    # Load file
    batch_sample_res = pd.read_csv(batch_sample_file, sep='\t', index_col=0)
    # Get true cell types
    batch_sample_labels_true = batch_sample_res['cell_phntp_full']
    # Get expression data
    batch_sample_exp = batch_sample_res.loc[:, markers]
    batch_sample_exp['std'] = std
    # Add it to expression table
    batch_sample_exp_df = pd.concat([batch_sample_exp_df, batch_sample_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_sample_labels_col = [col for col in batch_sample_res.columns if 'Phenotypes_' in col]
    for annot in batch_sample_labels_col:
        batch_sample_labels_pred = batch_sample_res[annot]
        batch_sample_score_ami = adjusted_mutual_info_score(batch_sample_labels_true, batch_sample_labels_pred)
        batch_sample_rows_ami.append({'std': std, 'batch_sample_score_ami': batch_sample_score_ami})
        batch_sample_score_jaccard = jaccard_score(batch_sample_labels_true, batch_sample_labels_pred, average='weighted')
        batch_sample_rows_jaccard.append({'std': std, 'batch_sample_score_jaccard': batch_sample_score_jaccard})

# Build final AMI dataframe, sort by std and reset indices
df_batch_sample_ami = pd.DataFrame(batch_sample_rows_ami).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_ami = df_batch_sample_ami.groupby('std', as_index=False)['batch_sample_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_batch_sample_jaccard = pd.DataFrame(batch_sample_rows_jaccard).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_jaccard = df_batch_sample_jaccard.groupby('std', as_index=False)['batch_sample_score_jaccard'].mean()

# Melt expression table
batch_sample_exp_df_melted = pd.melt(batch_sample_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_sample_ami['std'], 0.05, 0.03), y=df_batch_sample_ami['batch_sample_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_ami['std'], 0.05, 0.03), y=avg_batch_sample_ami['batch_sample_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_sample_jaccard['std'], 0.05, 0.03), y=df_batch_sample_jaccard['batch_sample_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_jaccard['std'], 0.05, 0.03), y=avg_batch_sample_jaccard['batch_sample_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_sample_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation and increased sample/batch number on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_batch_sample_std.jpg', dpi=600)
plt.close()
print('Saved default_batch_sample_std.jpg')


# Plot AMI/Jaccard similarity across batches - only negative markers
# Initialise objects
batch_dir = '../test_results/default_batch_test/'  # Folder
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.tsv')]  # tsv files
batch_rows_ami = []  # AMI data list
batch_rows_jaccard = []  # Jaccard data list
batch_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_files:
    # Get std value
    nb_batch = float(file[:-29].replace('default_cell_expression_', ''))
    # Build path
    batch_file = os.path.join(batch_dir, file)
    # Load file
    batch_res = pd.read_csv(batch_file, sep='\t', index_col=0)
    # Get true cell types
    batch_labels_true = batch_res['cell_phntp_full']
    # Get expression data
    batch_exp = batch_res.loc[:, markers]
    batch_exp['nb_batch'] = nb_batch
    # Add it to expression table
    batch_exp_df = pd.concat([batch_exp_df, batch_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_labels_col = [col for col in batch_res.columns if 'Phenotypes_' in col]
    for annot in batch_labels_col:
        batch_labels_pred = batch_res[annot]
        batch_score_ami = adjusted_mutual_info_score(batch_labels_true, batch_labels_pred)
        batch_rows_ami.append({'nb_batch': nb_batch, 'batch_score_ami': batch_score_ami})
        batch_score_jaccard = jaccard_score(batch_labels_true, batch_labels_pred, average='weighted')
        batch_rows_jaccard.append({'nb_batch': nb_batch, 'batch_score_jaccard': batch_score_jaccard})

# Build final AMI dataframe, sort by nb_batch and reset indices
df_batch_ami = pd.DataFrame(batch_rows_ami).sort_values(by='nb_batch', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_ami = df_batch_ami.groupby('nb_batch', as_index=False)['batch_score_ami'].mean()

# Build final ARI dataframe, sort by nb_batch and reset indices
df_batch_jaccard = pd.DataFrame(batch_rows_jaccard).sort_values(by='nb_batch', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_jaccard = df_batch_jaccard.groupby('nb_batch', as_index=False)['batch_score_jaccard'].mean()

# Melt expression table
batch_exp_df_melted = pd.melt(batch_exp_df, id_vars='nb_batch')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_ami['nb_batch'], 0, 0.1), y=df_batch_ami['batch_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_ami['nb_batch'], 0, 0.1), y=avg_batch_ami['batch_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_jaccard['nb_batch'], 0, 0.1), y=df_batch_jaccard['batch_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_jaccard['nb_batch'], 0, 0.1), y=avg_batch_jaccard['batch_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Number of batches', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_exp_df_melted, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of batch number on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_batch.jpg', dpi=600)
plt.close()
print('Saved default_batch.jpg')


# Plot AMI/Jaccard similarity across samples - only negative markers
# Initialise objects
sample_dir = '../test_results/default_sample_test/'  # Folder
sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.tsv')]  # tsv files
sample_rows_ami = []  # AMI data list
sample_rows_jaccard = []  # Jaccard data list
sample_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in sample_files:
    # Get std value
    nb_sample = float(file[:-29].replace('default_cell_expression_', ''))
    # Build path
    sample_file = os.path.join(sample_dir, file)
    # Load file
    sample_res = pd.read_csv(sample_file, sep='\t', index_col=0)
    # Get true cell types
    sample_labels_true = sample_res['cell_phntp_full']
    # Get expression data
    sample_exp = sample_res.loc[:, markers]
    sample_exp['nb_sample'] = nb_sample
    # Add it to expression table
    sample_exp_df = pd.concat([sample_exp_df, sample_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    sample_labels_col = [col for col in sample_res.columns if 'Phenotypes_' in col]
    for annot in sample_labels_col:
        sample_labels_pred = sample_res[annot]
        sample_score_ami = adjusted_mutual_info_score(sample_labels_true, sample_labels_pred)
        sample_rows_ami.append({'nb_sample': nb_sample, 'sample_score_ami': sample_score_ami})
        sample_score_jaccard = jaccard_score(sample_labels_true, sample_labels_pred, average='weighted')
        sample_rows_jaccard.append({'nb_sample': nb_sample, 'sample_score_jaccard': sample_score_jaccard})

# Build final AMI dataframe, sort by nb_sample and reset indices
df_sample_ami = pd.DataFrame(sample_rows_ami).sort_values(by='nb_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_ami = df_sample_ami.groupby('nb_sample', as_index=False)['sample_score_ami'].mean()

# Build final ARI dataframe, sort by nb_sample and reset indices
df_sample_jaccard = pd.DataFrame(sample_rows_jaccard).sort_values(by='nb_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_jaccard = df_sample_jaccard.groupby('nb_sample', as_index=False)['sample_score_jaccard'].mean()

# Melt expression table
sample_exp_df_melted = pd.melt(sample_exp_df, id_vars='nb_sample')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_sample_ami['nb_sample'], 0, 0.1), y=df_sample_ami['sample_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_ami['nb_sample'], 0, 0.1), y=avg_sample_ami['sample_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_sample_jaccard['nb_sample'], 0, 0.1), y=df_sample_jaccard['sample_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_jaccard['nb_sample'], 0, 0.1), y=avg_sample_jaccard['sample_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Number of samples', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=sample_exp_df_melted, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of sample number on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_sample.jpg', dpi=600)
plt.close()
print('Saved default_sample.jpg')


# Plot AMI/Jaccard similarity across batches and samples - only negative markers
# Initialise objects
batch_sample_dir = '../test_results/default_batch_sample_test/'  # Folder
batch_sample_files = [f for f in os.listdir(batch_sample_dir) if f.endswith('.tsv')]  # tsv files
batch_sample_rows_ami = []  # AMI data list
batch_sample_rows_jaccard = []  # Jaccard data list
batch_sample_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_sample_files:
    # Get std value
    nb_batch_sample = float(file[:-38].replace('default_cell_expression_', ''))
    # Build path
    batch_sample_file = os.path.join(batch_sample_dir, file)
    # Load file
    batch_sample_res = pd.read_csv(batch_sample_file, sep='\t', index_col=0)
    # Get true cell types
    batch_sample_labels_true = batch_sample_res['cell_phntp_full']
    # Get expression data
    batch_sample_exp = batch_sample_res.loc[:, markers]
    batch_sample_exp['nb_batch_sample'] = nb_batch_sample
    # Add it to expression table
    batch_sample_exp_df = pd.concat([batch_sample_exp_df, batch_sample_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_sample_labels_col = [col for col in batch_sample_res.columns if 'Phenotypes_' in col]
    for annot in batch_sample_labels_col:
        batch_sample_labels_pred = batch_sample_res[annot]
        batch_sample_score_ami = adjusted_mutual_info_score(batch_sample_labels_true, batch_sample_labels_pred)
        batch_sample_rows_ami.append({'nb_batch_sample': nb_batch_sample, 'batch_sample_score_ami': batch_sample_score_ami})
        batch_sample_score_jaccard = jaccard_score(batch_sample_labels_true, batch_sample_labels_pred, average='weighted')
        batch_sample_rows_jaccard.append({'nb_batch_sample': nb_batch_sample, 'batch_sample_score_jaccard': batch_sample_score_jaccard})

# Build final AMI dataframe, sort by nb_batch_sample and reset indices
df_batch_sample_ami = pd.DataFrame(batch_sample_rows_ami).sort_values(by='nb_batch_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_ami = df_batch_sample_ami.groupby('nb_batch_sample', as_index=False)['batch_sample_score_ami'].mean()

# Build final ARI dataframe, sort by nb_batch_sample and reset indices
df_batch_sample_jaccard = pd.DataFrame(batch_sample_rows_jaccard).sort_values(by='nb_batch_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_jaccard = df_batch_sample_jaccard.groupby('nb_batch_sample', as_index=False)['batch_sample_score_jaccard'].mean()

# Melt expression table
batch_sample_exp_df_melted = pd.melt(batch_sample_exp_df, id_vars='nb_batch_sample')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_sample_ami['nb_batch_sample'], 0, 0.1), y=df_batch_sample_ami['batch_sample_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_ami['nb_batch_sample'], 0, 0.1), y=avg_batch_sample_ami['batch_sample_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_sample_jaccard['nb_batch_sample'], 0, 0.1), y=df_batch_sample_jaccard['batch_sample_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_jaccard['nb_batch_sample'], 0, 0.1), y=avg_batch_sample_jaccard['batch_sample_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Number of batches/samples', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_sample_exp_df_melted, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of batch and sample number on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_batch_sample.jpg', dpi=600)
plt.close()
print('Saved default_batch_sample.jpg')


# Plot AMI/Jaccard similarity across xmin - only negative markers
# Initialise objects
xmin_dir = '../test_results/default_batch_sample_std_xmin_test/'  # Folder
xmin_files = [f for f in os.listdir(xmin_dir) if f.endswith('.tsv')]  # tsv files
xmin_rows_ami = []  # AMI data list
xmin_rows_jaccard = []  # Jaccard data list
xmin_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in xmin_files:
    # Get std value
    nb_xmin = float(file.replace('default_cell_expression_4batches_4samples_std0.75_xmin', '').replace('_annotated.tsv', ''))
    # Build path
    xmin_file = os.path.join(xmin_dir, file)
    # Load file
    xmin_res = pd.read_csv(xmin_file, sep='\t', index_col=0)
    # Get true cell types
    xmin_labels_true = xmin_res['cell_phntp_full']
    # Get expression data
    xmin_exp = xmin_res.loc[:, markers]
    xmin_exp['nb_xmin'] = nb_xmin
    # Add it to expression table
    xmin_exp_df = pd.concat([xmin_exp_df, xmin_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    xmin_labels_col = [col for col in xmin_res.columns if 'Phenotypes_' in col]
    for annot in xmin_labels_col:
        xmin_labels_pred = xmin_res[annot]
        xmin_score_ami = adjusted_mutual_info_score(xmin_labels_true, xmin_labels_pred)
        xmin_rows_ami.append({'nb_xmin': nb_xmin, 'xmin_score_ami': xmin_score_ami})
        xmin_score_jaccard = jaccard_score(xmin_labels_true, xmin_labels_pred, average='weighted')
        xmin_rows_jaccard.append({'nb_xmin': nb_xmin, 'xmin_score_jaccard': xmin_score_jaccard})

# Build final AMI dataframe, sort by nb_xmin and reset indices
df_xmin_ami = pd.DataFrame(xmin_rows_ami).sort_values(by='nb_xmin', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_xmin_ami = df_xmin_ami.groupby('nb_xmin', as_index=False)['xmin_score_ami'].mean()

# Build final ARI dataframe, sort by nb_xmin and reset indices
df_xmin_jaccard = pd.DataFrame(xmin_rows_jaccard).sort_values(by='nb_xmin', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_xmin_jaccard = df_xmin_jaccard.groupby('nb_xmin', as_index=False)['xmin_score_jaccard'].mean()

# Melt expression table
xmin_exp_df_melted = pd.melt(xmin_exp_df, id_vars='nb_xmin')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_xmin_ami['nb_xmin'], 0, 1.5), y=df_xmin_ami['xmin_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_xmin_ami['nb_xmin'], 0, 1.5), y=avg_xmin_ami['xmin_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_xmin_jaccard['nb_xmin'], 0, 1.5), y=df_xmin_jaccard['xmin_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_xmin_jaccard['nb_xmin'], 0, 1.5), y=avg_xmin_jaccard['xmin_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Minimum cell per phenotype per sample', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=xmin_exp_df_melted, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of minimum number of cells per sample per phenotype on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_xmin.jpg', dpi=600)
plt.close()
print('Saved default_xmin.jpg')


# Plot AMI/Jaccard similarity across maxmarkers - only negative markers
# Initialise objects
maxmarkers_dir = '../test_results/default_batch_sample_std_maxmarkers_test/'  # Folder
maxmarkers_files = [f for f in os.listdir(maxmarkers_dir) if f.endswith('.tsv')]  # tsv files
maxmarkers_rows_ami = []  # AMI data list
maxmarkers_rows_jaccard = []  # Jaccard data list
maxmarkers_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in maxmarkers_files:
    # Get std value
    nb_maxmarkers = float(file.replace('default_cell_expression_4batches_4samples_std0.75_', '').replace('markers_annotated.tsv', ''))
    # Build path
    maxmarkers_file = os.path.join(maxmarkers_dir, file)
    # Load file
    maxmarkers_res = pd.read_csv(maxmarkers_file, sep='\t', index_col=0)
    # Get true cell types
    maxmarkers_labels_true = maxmarkers_res['cell_phntp_full']
    # Get expression data
    maxmarkers_exp = maxmarkers_res.loc[:, markers]
    maxmarkers_exp['nb_maxmarkers'] = nb_maxmarkers
    # Add it to expression table
    maxmarkers_exp_df = pd.concat([maxmarkers_exp_df, maxmarkers_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    maxmarkers_labels_col = [col for col in maxmarkers_res.columns if 'Phenotypes_' in col]
    for annot in maxmarkers_labels_col:
        maxmarkers_labels_pred = maxmarkers_res[annot]
        maxmarkers_score_ami = adjusted_mutual_info_score(maxmarkers_labels_true, maxmarkers_labels_pred)
        maxmarkers_rows_ami.append({'nb_maxmarkers': nb_maxmarkers, 'maxmarkers_score_ami': maxmarkers_score_ami})
        maxmarkers_score_jaccard = jaccard_score(maxmarkers_labels_true, maxmarkers_labels_pred, average='weighted')
        maxmarkers_rows_jaccard.append({'nb_maxmarkers': nb_maxmarkers, 'maxmarkers_score_jaccard': maxmarkers_score_jaccard})

# Build final AMI dataframe, sort by nb_maxmarkers and reset indices
df_maxmarkers_ami = pd.DataFrame(maxmarkers_rows_ami).sort_values(by='nb_maxmarkers', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_maxmarkers_ami = df_maxmarkers_ami.groupby('nb_maxmarkers', as_index=False)['maxmarkers_score_ami'].mean()

# Build final ARI dataframe, sort by nb_maxmarkers and reset indices
df_maxmarkers_jaccard = pd.DataFrame(maxmarkers_rows_jaccard).sort_values(by='nb_maxmarkers', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_maxmarkers_jaccard = df_maxmarkers_jaccard.groupby('nb_maxmarkers', as_index=False)['maxmarkers_score_jaccard'].mean()

# Melt expression table
maxmarkers_exp_df_melted = pd.melt(maxmarkers_exp_df, id_vars='nb_maxmarkers')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_maxmarkers_ami['nb_maxmarkers'], 0, 0.05), y=df_maxmarkers_ami['maxmarkers_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_maxmarkers_ami['nb_maxmarkers'], 0, 0.05), y=avg_maxmarkers_ami['maxmarkers_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_maxmarkers_jaccard['nb_maxmarkers'], 0, 0.05), y=df_maxmarkers_jaccard['maxmarkers_score_jaccard'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_maxmarkers_jaccard['nb_maxmarkers'], 0, 0.05), y=avg_maxmarkers_jaccard['maxmarkers_score_jaccard'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Maximum number of markers per phenotype', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=maxmarkers_exp_df_melted, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of maximum markers used in phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_maxmarkers.jpg', dpi=600)
plt.close()
print('Saved default_maxmarkers.jpg')


# Plot AMI with and without KNN - only negative markers
# Initialise objects
knn_dir = '../test_results/default_fake_std_test/'  # Folder
knn_files = [f for f in os.listdir(knn_dir) if f.endswith('.tsv')]  # tsv files
knn_rows_ami = []  # AMI data list
knn_rows_ami2 = []  # AMI KNN data list
knn_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in knn_files:
    # Get std value
    std = float(file.replace('default_cell_expression_fake_std', '').replace('_annotated.tsv', ''))
    # Build path
    knn_file = os.path.join(knn_dir, file)
    # Load file
    knn_res = pd.read_csv(knn_file, sep='\t', index_col=0)
    # Get true cell types
    knn_labels_true = knn_res['cell_phntp_full']
    # Get expression data
    knn_exp = knn_res.loc[:, markers]
    knn_exp['std'] = std
    # Add it to expression table
    knn_exp_df = pd.concat([knn_exp_df, knn_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    knn_labels_col = [col for col in knn_res.columns if 'Phenotypes_' in col]
    knn_labels_col2 = [col for col in knn_res.columns if 'KNN_phenotype_' in col]
    for annot, annot2 in zip(knn_labels_col, knn_labels_col2):
        knn_labels_pred = knn_res[annot]
        knn_score_ami = adjusted_mutual_info_score(knn_labels_true, knn_labels_pred)
        knn_rows_ami.append({'std': std, 'knn_score_ami': knn_score_ami})
        knn_labels_pred2 = knn_res[annot2]
        knn_score_ami2 = adjusted_mutual_info_score(knn_labels_true, knn_labels_pred2)
        knn_rows_ami2.append({'std': std, 'knn_score_ami': knn_score_ami2})

# Build final AMI dataframe, sort by std and reset indices
df_knn_ami = pd.DataFrame(knn_rows_ami).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_knn_ami = df_knn_ami.groupby('std', as_index=False)['knn_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_knn_ami2 = pd.DataFrame(knn_rows_ami2).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_knn_ami2 = df_knn_ami2.groupby('std', as_index=False)['knn_score_ami'].mean()

# Melt expression table
knn_exp_df_melted = pd.melt(knn_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_knn_ami['std'], 0.05, 0.03), y=df_knn_ami['knn_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_knn_ami['std'], 0.05, 0.03), y=avg_knn_ami['knn_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_knn_ami2['std'], 0.05, 0.03), y=df_knn_ami2['knn_score_ami'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_knn_ami2['std'], 0.05, 0.03), y=avg_knn_ami2['knn_score_ami'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'AMI post-KNN', 'Mean AMI post-KNN'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=knn_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of KNN-reclassification on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_fake_knn.jpg', dpi=600)
plt.close()
print('Saved default_fake_knn.jpg')


# Plot AMI with and without unidentified cells - only negative markers
# Initialise objects
no_ident_dir = '../test_results/default_batch_sample_std_test/'  # Folder
no_ident_files = [f for f in os.listdir(no_ident_dir) if f.endswith('.tsv')]  # tsv files
no_ident_rows_ami = []  # AMI data list
no_ident_rows_ami2 = []  # AMI no unidentified data list
no_ident_exp_df = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in no_ident_files:
    # Get std value
    std = float(file.replace('default_cell_expression_batch_sample_std', '').replace('_annotated.tsv', ''))
    # Build path
    no_ident_file = os.path.join(no_ident_dir, file)
    # Load file
    no_ident_res = pd.read_csv(no_ident_file, sep='\t', index_col=0)
    # Get true cell types
    no_ident_labels_true = no_ident_res['cell_phntp_full']
    # Get expression data
    no_ident_exp = no_ident_res.loc[:, markers]
    no_ident_exp['std'] = std
    # Add it to expression table
    no_ident_exp_df = pd.concat([no_ident_exp_df, no_ident_exp])
    # Get columns containing annotations (several optimal combinations --> several columns)
    no_ident_labels_col = [col for col in no_ident_res.columns if 'Phenotypes_' in col]
    for annot in no_ident_labels_col:
        no_ident_labels_pred = no_ident_res[annot]
        no_ident_score_ami = adjusted_mutual_info_score(no_ident_labels_true, no_ident_labels_pred)
        no_ident_rows_ami.append({'std': std, 'no_ident_score_ami': no_ident_score_ami})
    # Get table without unidentified cells
    no_ident_res2 = no_ident_res[no_ident_res['KNN_annotations_1'] != 'Other 42'].copy()
    # Get true cell types
    no_ident_labels_true2 = no_ident_res2['cell_phntp_full']
    # Get columns containing annotations (several optimal combinations --> several columns)
    no_ident_labels_col2 = [col for col in no_ident_res2.columns if 'Phenotypes_' in col]
    for annot2 in no_ident_labels_col2:
        no_ident_labels_pred2 = no_ident_res2[annot2]
        no_ident_score_ami2 = adjusted_mutual_info_score(no_ident_labels_true2, no_ident_labels_pred2)
        no_ident_rows_ami2.append({'std': std, 'no_ident_score_ami2': no_ident_score_ami2})

# Build final AMI dataframe, sort by std and reset indices
df_no_ident_ami = pd.DataFrame(no_ident_rows_ami).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_no_ident_ami = df_no_ident_ami.groupby('std', as_index=False)['no_ident_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_no_ident_ami2 = pd.DataFrame(no_ident_rows_ami2).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_no_ident_ami2 = df_no_ident_ami2.groupby('std', as_index=False)['no_ident_score_ami2'].mean()

# Melt expression table
no_ident_exp_df_melted = pd.melt(no_ident_exp_df, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_no_ident_ami['std'], 0.05, 0.03), y=df_no_ident_ami['no_ident_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_no_ident_ami['std'], 0.05, 0.03), y=avg_no_ident_ami['no_ident_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_no_ident_ami2['std'], 0.05, 0.03), y=df_no_ident_ami2['no_ident_score_ami2'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_no_ident_ami2['std'], 0.05, 0.03), y=avg_no_ident_ami2['no_ident_score_ami2'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'AMI w.o unIDed', 'Mean AMI w.o unIDed'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=no_ident_exp_df_melted, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of unidentified cells on phenotype identification\n(Negative markers)')
fig.tight_layout()
fig.figure.savefig('default_fake_unidentified.jpg', dpi=600)
plt.close()
print('Saved default_fake_unidentified.jpg')










# Plot AMI/Jaccard similarity across std - negative and positive markers
# Initialise objects
std_dir_mixed = '../test_results/mixed_fake_std_test/'  # Folder
std_files_mixed = [f for f in os.listdir(std_dir_mixed) if f.endswith('.tsv')]  # tsv files
std_rows_ami_mixed = []  # AMI data list
std_rows_jaccard_mixed = []  # Jaccard data list
std_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in std_files_mixed:
    # Get std value
    std = float(file.replace('mixed_cell_expression_fake_std', '').replace('_annotated.tsv', ''))
    # Build path
    std_file_mixed = os.path.join(std_dir_mixed, file)
    # Load file
    std_res_mixed = pd.read_csv(std_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    std_labels_true_mixed = std_res_mixed['cell_phntp_full']
    # Get expression data
    std_exp_mixed = std_res_mixed.loc[:, markers]
    std_exp_mixed['std'] = std
    # Add it to expression table
    std_exp_df_mixed = pd.concat([std_exp_df_mixed, std_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    std_labels_col_mixed = [col for col in std_res_mixed.columns if 'Phenotypes_' in col]
    for annot in std_labels_col_mixed:
        std_labels_pred_mixed = std_res_mixed[annot]
        std_score_ami_mixed = adjusted_mutual_info_score(std_labels_true_mixed, std_labels_pred_mixed)
        std_rows_ami_mixed.append({'std': std, 'std_score_ami_mixed': std_score_ami_mixed})
        std_score_jaccard_mixed = jaccard_score(std_labels_true_mixed, std_labels_pred_mixed, average='weighted')
        std_rows_jaccard_mixed.append({'std': std, 'std_score_jaccard_mixed': std_score_jaccard_mixed})

# Build final AMI dataframe, sort by std and reset indices
df_std_ami_mixed = pd.DataFrame(std_rows_ami_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_std_ami_mixed = df_std_ami_mixed.groupby('std', as_index=False)['std_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_std_jaccard_mixed = pd.DataFrame(std_rows_jaccard_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_std_jaccard_mixed = df_std_jaccard_mixed.groupby('std', as_index=False)['std_score_jaccard_mixed'].mean()

# Melt expression table
std_exp_df_melted_mixed = pd.melt(std_exp_df_mixed, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_std_ami_mixed['std'], 0.05, 0.03), y=df_std_ami_mixed['std_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_std_ami_mixed['std'], 0.05, 0.03), y=avg_std_ami_mixed['std_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_std_jaccard_mixed['std'], 0.05, 0.03), y=df_std_jaccard_mixed['std_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_std_jaccard_mixed['std'], 0.05, 0.03), y=avg_std_jaccard_mixed['std_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=std_exp_df_melted_mixed, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_fake_std.jpg', dpi=600)
plt.close()
print('Saved mixed_fake_std.jpg')


# Plot AMI/Jaccard similarity across std with 4 batches - negative and positive markers
# Initialise objects
batch_std_dir_mixed = '../test_results/mixed_batch_std_test/'  # Folder
batch_std_files_mixed = [f for f in os.listdir(batch_std_dir_mixed) if f.endswith('.tsv')]  # tsv files
batch_std_rows_ami_mixed = []  # AMI data list
batch_std_rows_jaccard_mixed = []  # Jaccard data list
batch_std_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_std_files_mixed:
    # Get std value
    std = float(file.replace('mixed_cell_expression_batch_std', '').replace('_annotated.tsv', ''))
    # Build path
    batch_std_file_mixed = os.path.join(batch_std_dir_mixed, file)
    # Load file
    batch_std_res_mixed = pd.read_csv(batch_std_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    batch_std_labels_true_mixed = batch_std_res_mixed['cell_phntp_full']
    # Get expression data
    batch_std_exp_mixed = batch_std_res_mixed.loc[:, markers]
    batch_std_exp_mixed['std'] = std
    # Add it to expression table
    batch_std_exp_df_mixed = pd.concat([batch_std_exp_df_mixed, batch_std_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_std_labels_col_mixed = [col for col in batch_std_res_mixed.columns if 'Phenotypes_' in col]
    for annot in batch_std_labels_col_mixed:
        batch_std_labels_pred_mixed = batch_std_res_mixed[annot]
        batch_std_score_ami_mixed = adjusted_mutual_info_score(batch_std_labels_true_mixed, batch_std_labels_pred_mixed)
        batch_std_rows_ami_mixed.append({'std': std, 'batch_std_score_ami_mixed': batch_std_score_ami_mixed})
        batch_std_score_jaccard_mixed = jaccard_score(batch_std_labels_true_mixed, batch_std_labels_pred_mixed, average='weighted')
        batch_std_rows_jaccard_mixed.append({'std': std, 'batch_std_score_jaccard_mixed': batch_std_score_jaccard_mixed})

# Build final AMI dataframe, sort by std and reset indices
df_batch_std_ami_mixed = pd.DataFrame(batch_std_rows_ami_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_std_ami_mixed = df_batch_std_ami_mixed.groupby('std', as_index=False)['batch_std_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_batch_std_jaccard_mixed = pd.DataFrame(batch_std_rows_jaccard_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_std_jaccard_mixed = df_batch_std_jaccard_mixed.groupby('std', as_index=False)['batch_std_score_jaccard_mixed'].mean()

# Melt expression table
batch_std_exp_df_melted_mixed = pd.melt(batch_std_exp_df_mixed, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_std_ami_mixed['std'], 0.05, 0.03), y=df_batch_std_ami_mixed['batch_std_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_std_ami_mixed['std'], 0.05, 0.03), y=avg_batch_std_ami_mixed['batch_std_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_std_jaccard_mixed['std'], 0.05, 0.03), y=df_batch_std_jaccard_mixed['batch_std_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_std_jaccard_mixed['std'], 0.05, 0.03), y=avg_batch_std_jaccard_mixed['batch_std_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_std_exp_df_melted_mixed, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation and increased batch number on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_batch_std.jpg', dpi=600)
plt.close()
print('Saved mixed_batch_std.jpg')


# Plot AMI/Jaccard similarity across std with 4 samples - negative and positive markers
# Initialise objects
sample_dir_mixed = '../test_results/mixed_sample_std_test/'  # Folder
sample_files_mixed = [f for f in os.listdir(sample_dir_mixed) if f.endswith('.tsv')]  # tsv files
sample_rows_ami_mixed = []  # AMI data list
sample_rows_jaccard_mixed = []  # Jaccard data list
sample_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in sample_files_mixed:
    # Get std value
    std = float(file.replace('mixed_cell_expression_sample_std', '').replace('_annotated.tsv', ''))
    # Build path
    sample_file_mixed = os.path.join(sample_dir_mixed, file)
    # Load file
    sample_res_mixed = pd.read_csv(sample_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    sample_labels_true_mixed = sample_res_mixed['cell_phntp_full']
    # Get expression data
    sample_exp_mixed = sample_res_mixed.loc[:, markers]
    sample_exp_mixed['std'] = std
    # Add it to expression table
    sample_exp_df_mixed = pd.concat([sample_exp_df_mixed, sample_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    sample_labels_col_mixed = [col for col in sample_res_mixed.columns if 'Phenotypes_' in col]
    for annot in sample_labels_col_mixed:
        sample_labels_pred_mixed = sample_res_mixed[annot]
        sample_score_ami_mixed = adjusted_mutual_info_score(sample_labels_true_mixed, sample_labels_pred_mixed)
        sample_rows_ami_mixed.append({'std': std, 'sample_score_ami_mixed': sample_score_ami_mixed})
        sample_score_jaccard_mixed = jaccard_score(sample_labels_true_mixed, sample_labels_pred_mixed, average='weighted')
        sample_rows_jaccard_mixed.append({'std': std, 'sample_score_jaccard_mixed': sample_score_jaccard_mixed})

# Build final AMI dataframe, sort by std and reset indices
df_sample_ami_mixed = pd.DataFrame(sample_rows_ami_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_ami_mixed = df_sample_ami_mixed.groupby('std', as_index=False)['sample_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_sample_jaccard_mixed = pd.DataFrame(sample_rows_jaccard_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_jaccard_mixed = df_sample_jaccard_mixed.groupby('std', as_index=False)['sample_score_jaccard_mixed'].mean()

# Melt expression table
sample_exp_df_melted_mixed = pd.melt(sample_exp_df_mixed, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_sample_ami_mixed['std'], 0.05, 0.03), y=df_sample_ami_mixed['sample_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_ami_mixed['std'], 0.05, 0.03), y=avg_sample_ami_mixed['sample_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_sample_jaccard_mixed['std'], 0.05, 0.03), y=df_sample_jaccard_mixed['sample_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_jaccard_mixed['std'], 0.05, 0.03), y=avg_sample_jaccard_mixed['sample_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=sample_exp_df_melted_mixed, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation and increased sample number on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_sample_std.jpg', dpi=600)
plt.close()
print('Saved mixed_sample_std.jpg')


# Plot AMI/Jaccard similarity across std with batches and samples - negative and positive markers
# Initialise objects
batch_sample_dir_mixed = '../test_results/mixed_batch_sample_std_test/'  # Folder
batch_sample_files_mixed = [f for f in os.listdir(batch_sample_dir_mixed) if f.endswith('.tsv')]  # tsv files
batch_sample_rows_ami_mixed = []  # AMI data list
batch_sample_rows_jaccard_mixed = []  # Jaccard data list
batch_sample_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_sample_files_mixed:
    # Get std value
    std = float(file.replace('mixed_cell_expression_batch_sample_std', '').replace('_annotated.tsv', ''))
    # Build path
    batch_sample_file_mixed = os.path.join(batch_sample_dir_mixed, file)
    # Load file
    batch_sample_res_mixed = pd.read_csv(batch_sample_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    batch_sample_labels_true_mixed = batch_sample_res_mixed['cell_phntp_full']
    # Get expression data
    batch_sample_exp_mixed = batch_sample_res_mixed.loc[:, markers]
    batch_sample_exp_mixed['std'] = std
    # Add it to expression table
    batch_sample_exp_df_mixed = pd.concat([batch_sample_exp_df_mixed, batch_sample_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_sample_labels_col_mixed = [col for col in batch_sample_res_mixed.columns if 'Phenotypes_' in col]
    for annot in batch_sample_labels_col_mixed:
        batch_sample_labels_pred_mixed = batch_sample_res_mixed[annot]
        batch_sample_score_ami_mixed = adjusted_mutual_info_score(batch_sample_labels_true_mixed, batch_sample_labels_pred_mixed)
        batch_sample_rows_ami_mixed.append({'std': std, 'batch_sample_score_ami_mixed': batch_sample_score_ami_mixed})
        batch_sample_score_jaccard_mixed = jaccard_score(batch_sample_labels_true_mixed, batch_sample_labels_pred_mixed, average='weighted')
        batch_sample_rows_jaccard_mixed.append({'std': std, 'batch_sample_score_jaccard_mixed': batch_sample_score_jaccard_mixed})

# Build final AMI dataframe, sort by std and reset indices
df_batch_sample_ami_mixed = pd.DataFrame(batch_sample_rows_ami_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_ami_mixed = df_batch_sample_ami_mixed.groupby('std', as_index=False)['batch_sample_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_batch_sample_jaccard_mixed = pd.DataFrame(batch_sample_rows_jaccard_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_jaccard_mixed = df_batch_sample_jaccard_mixed.groupby('std', as_index=False)['batch_sample_score_jaccard_mixed'].mean()

# Melt expression table
batch_sample_exp_df_melted_mixed = pd.melt(batch_sample_exp_df_mixed, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_sample_ami_mixed['std'], 0.05, 0.03), y=df_batch_sample_ami_mixed['batch_sample_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_ami_mixed['std'], 0.05, 0.03), y=avg_batch_sample_ami_mixed['batch_sample_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_sample_jaccard_mixed['std'], 0.05, 0.03), y=df_batch_sample_jaccard_mixed['batch_sample_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_jaccard_mixed['std'], 0.05, 0.03), y=avg_batch_sample_jaccard_mixed['batch_sample_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_sample_exp_df_melted_mixed, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of expression standard deviation and increased sample/batch number on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_batch_sample_std.jpg', dpi=600)
plt.close()
print('Saved mixed_batch_sample_std.jpg')


# Plot AMI/Jaccard similarity across batches - negative and positive markers
# Initialise objects
batch_dir_mixed = '../test_results/mixed_batch_test/'  # Folder
batch_files_mixed = [f for f in os.listdir(batch_dir_mixed) if f.endswith('.tsv')]  # tsv files
batch_rows_ami_mixed = []  # AMI data list
batch_rows_jaccard_mixed = []  # Jaccard data list
batch_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_files_mixed:
    # Get std value
    nb_batch = float(file[:-29].replace('mixed_cell_expression_', ''))
    # Build path
    batch_file_mixed = os.path.join(batch_dir_mixed, file)
    # Load file
    batch_res_mixed = pd.read_csv(batch_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    batch_labels_true_mixed = batch_res_mixed['cell_phntp_full']
    # Get expression data
    batch_exp_mixed = batch_res_mixed.loc[:, markers]
    batch_exp_mixed['nb_batch'] = nb_batch
    # Add it to expression table
    batch_exp_df_mixed = pd.concat([batch_exp_df_mixed, batch_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_labels_col_mixed = [col for col in batch_res_mixed.columns if 'Phenotypes_' in col]
    for annot in batch_labels_col_mixed:
        batch_labels_pred_mixed = batch_res_mixed[annot]
        batch_score_ami_mixed = adjusted_mutual_info_score(batch_labels_true_mixed, batch_labels_pred_mixed)
        batch_rows_ami_mixed.append({'nb_batch': nb_batch, 'batch_score_ami_mixed': batch_score_ami_mixed})
        batch_score_jaccard_mixed = jaccard_score(batch_labels_true_mixed, batch_labels_pred_mixed, average='weighted')
        batch_rows_jaccard_mixed.append({'nb_batch': nb_batch, 'batch_score_jaccard_mixed': batch_score_jaccard_mixed})

# Build final AMI dataframe, sort by nb_batch and reset indices
df_batch_ami_mixed = pd.DataFrame(batch_rows_ami_mixed).sort_values(by='nb_batch', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_ami_mixed = df_batch_ami_mixed.groupby('nb_batch', as_index=False)['batch_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by nb_batch and reset indices
df_batch_jaccard_mixed = pd.DataFrame(batch_rows_jaccard_mixed).sort_values(by='nb_batch', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_jaccard_mixed = df_batch_jaccard_mixed.groupby('nb_batch', as_index=False)['batch_score_jaccard_mixed'].mean()

# Melt expression table
batch_exp_df_melted_mixed = pd.melt(batch_exp_df_mixed, id_vars='nb_batch')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_ami_mixed['nb_batch'], 0, 0.1), y=df_batch_ami_mixed['batch_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_ami_mixed['nb_batch'], 0, 0.1), y=avg_batch_ami_mixed['batch_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_jaccard_mixed['nb_batch'], 0, 0.1), y=df_batch_jaccard_mixed['batch_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_jaccard_mixed['nb_batch'], 0, 0.1), y=avg_batch_jaccard_mixed['batch_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Number of batches', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_exp_df_melted_mixed, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of batch number on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_batch.jpg', dpi=600)
plt.close()
print('Saved mixed_batch.jpg')


# Plot AMI/Jaccard similarity across samples - negative and positive markers
# Initialise objects
sample_dir_mixed = '../test_results/mixed_sample_test/'  # Folder
sample_files_mixed = [f for f in os.listdir(sample_dir_mixed) if f.endswith('.tsv')]  # tsv files
sample_rows_ami_mixed = []  # AMI data list
sample_rows_jaccard_mixed = []  # Jaccard data list
sample_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in sample_files_mixed:
    # Get std value
    nb_sample = float(file[:-29].replace('mixed_cell_expression_', ''))
    # Build path
    sample_file_mixed = os.path.join(sample_dir_mixed, file)
    # Load file
    sample_res_mixed = pd.read_csv(sample_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    sample_labels_true_mixed = sample_res_mixed['cell_phntp_full']
    # Get expression data
    sample_exp_mixed = sample_res_mixed.loc[:, markers]
    sample_exp_mixed['nb_sample'] = nb_sample
    # Add it to expression table
    sample_exp_df_mixed = pd.concat([sample_exp_df_mixed, sample_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    sample_labels_col_mixed = [col for col in sample_res_mixed.columns if 'Phenotypes_' in col]
    for annot in sample_labels_col_mixed:
        sample_labels_pred_mixed = sample_res_mixed[annot]
        sample_score_ami_mixed = adjusted_mutual_info_score(sample_labels_true_mixed, sample_labels_pred_mixed)
        sample_rows_ami_mixed.append({'nb_sample': nb_sample, 'sample_score_ami_mixed': sample_score_ami_mixed})
        sample_score_jaccard_mixed = jaccard_score(sample_labels_true_mixed, sample_labels_pred_mixed, average='weighted')
        sample_rows_jaccard_mixed.append({'nb_sample': nb_sample, 'sample_score_jaccard_mixed': sample_score_jaccard_mixed})

# Build final AMI dataframe, sort by nb_sample and reset indices
df_sample_ami_mixed = pd.DataFrame(sample_rows_ami_mixed).sort_values(by='nb_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_ami_mixed = df_sample_ami_mixed.groupby('nb_sample', as_index=False)['sample_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by nb_sample and reset indices
df_sample_jaccard_mixed = pd.DataFrame(sample_rows_jaccard_mixed).sort_values(by='nb_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_sample_jaccard_mixed = df_sample_jaccard_mixed.groupby('nb_sample', as_index=False)['sample_score_jaccard_mixed'].mean()

# Melt expression table
sample_exp_df_melted_mixed = pd.melt(sample_exp_df_mixed, id_vars='nb_sample')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_sample_ami_mixed['nb_sample'], 0, 0.1), y=df_sample_ami_mixed['sample_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_ami_mixed['nb_sample'], 0, 0.1), y=avg_sample_ami_mixed['sample_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_sample_jaccard_mixed['nb_sample'], 0, 0.1), y=df_sample_jaccard_mixed['sample_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_sample_jaccard_mixed['nb_sample'], 0, 0.1), y=avg_sample_jaccard_mixed['sample_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Number of samples', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=sample_exp_df_melted_mixed, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of sample number on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_sample.jpg', dpi=600)
plt.close()
print('Saved mixed_sample.jpg')


# Plot AMI/Jaccard similarity across batches and samples - negative and positive markers
# Initialise objects
batch_sample_dir_mixed = '../test_results/mixed_batch_sample_test/'  # Folder
batch_sample_files_mixed = [f for f in os.listdir(batch_sample_dir_mixed) if f.endswith('.tsv')]  # tsv files
batch_sample_rows_ami_mixed = []  # AMI data list
batch_sample_rows_jaccard_mixed = []  # Jaccard data list
batch_sample_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in batch_sample_files_mixed:
    # Get std value
    nb_batch_sample = float(file[:-38].replace('mixed_cell_expression_', ''))
    # Build path
    batch_sample_file_mixed = os.path.join(batch_sample_dir_mixed, file)
    # Load file
    batch_sample_res_mixed = pd.read_csv(batch_sample_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    batch_sample_labels_true_mixed = batch_sample_res_mixed['cell_phntp_full']
    # Get expression data
    batch_sample_exp_mixed = batch_sample_res_mixed.loc[:, markers]
    batch_sample_exp_mixed['nb_batch_sample'] = nb_batch_sample
    # Add it to expression table
    batch_sample_exp_df_mixed = pd.concat([batch_sample_exp_df_mixed, batch_sample_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    batch_sample_labels_col_mixed = [col for col in batch_sample_res_mixed.columns if 'Phenotypes_' in col]
    for annot in batch_sample_labels_col_mixed:
        batch_sample_labels_pred_mixed = batch_sample_res_mixed[annot]
        batch_sample_score_ami_mixed = adjusted_mutual_info_score(batch_sample_labels_true_mixed, batch_sample_labels_pred_mixed)
        batch_sample_rows_ami_mixed.append({'nb_batch_sample': nb_batch_sample, 'batch_sample_score_ami_mixed': batch_sample_score_ami_mixed})
        batch_sample_score_jaccard_mixed = jaccard_score(batch_sample_labels_true_mixed, batch_sample_labels_pred_mixed, average='weighted')
        batch_sample_rows_jaccard_mixed.append({'nb_batch_sample': nb_batch_sample, 'batch_sample_score_jaccard_mixed': batch_sample_score_jaccard_mixed})

# Build final AMI dataframe, sort by nb_batch_sample and reset indices
df_batch_sample_ami_mixed = pd.DataFrame(batch_sample_rows_ami_mixed).sort_values(by='nb_batch_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_ami_mixed = df_batch_sample_ami_mixed.groupby('nb_batch_sample', as_index=False)['batch_sample_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by nb_batch_sample and reset indices
df_batch_sample_jaccard_mixed = pd.DataFrame(batch_sample_rows_jaccard_mixed).sort_values(by='nb_batch_sample', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_batch_sample_jaccard_mixed = df_batch_sample_jaccard_mixed.groupby('nb_batch_sample', as_index=False)['batch_sample_score_jaccard_mixed'].mean()

# Melt expression table
batch_sample_exp_df_melted_mixed = pd.melt(batch_sample_exp_df_mixed, id_vars='nb_batch_sample')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_batch_sample_ami_mixed['nb_batch_sample'], 0, 0.1), y=df_batch_sample_ami_mixed['batch_sample_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_ami_mixed['nb_batch_sample'], 0, 0.1), y=avg_batch_sample_ami_mixed['batch_sample_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_batch_sample_jaccard_mixed['nb_batch_sample'], 0, 0.1), y=df_batch_sample_jaccard_mixed['batch_sample_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_batch_sample_jaccard_mixed['nb_batch_sample'], 0, 0.1), y=avg_batch_sample_jaccard_mixed['batch_sample_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Number of batches/samples', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=batch_sample_exp_df_melted_mixed, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of batch and sample number on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_batch_sample.jpg', dpi=600)
plt.close()
print('Saved mixed_batch_sample.jpg')


# Plot AMI/Jaccard similarity across xmin - negative and positive markers
# Initialise objects
xmin_dir_mixed = '../test_results/mixed_batch_sample_std_xmin_test/'  # Folder
xmin_files_mixed = [f for f in os.listdir(xmin_dir_mixed) if f.endswith('.tsv')]  # tsv files
xmin_rows_ami_mixed = []  # AMI data list
xmin_rows_jaccard_mixed = []  # Jaccard data list
xmin_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in xmin_files_mixed:
    # Get std value
    nb_xmin = float(file.replace('mixed_cell_expression_4batches_4samples_std0.75_xmin', '').replace('_annotated.tsv', ''))
    # Build path
    xmin_file_mixed = os.path.join(xmin_dir_mixed, file)
    # Load file
    xmin_res_mixed = pd.read_csv(xmin_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    xmin_labels_true_mixed = xmin_res_mixed['cell_phntp_full']
    # Get expression data
    xmin_exp_mixed = xmin_res_mixed.loc[:, markers]
    xmin_exp_mixed['nb_xmin'] = nb_xmin
    # Add it to expression table
    xmin_exp_df_mixed = pd.concat([xmin_exp_df_mixed, xmin_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    xmin_labels_col_mixed = [col for col in xmin_res_mixed.columns if 'Phenotypes_' in col]
    for annot in xmin_labels_col_mixed:
        xmin_labels_pred_mixed = xmin_res_mixed[annot]
        xmin_score_ami_mixed = adjusted_mutual_info_score(xmin_labels_true_mixed, xmin_labels_pred_mixed)
        xmin_rows_ami_mixed.append({'nb_xmin': nb_xmin, 'xmin_score_ami_mixed': xmin_score_ami_mixed})
        xmin_score_jaccard_mixed = jaccard_score(xmin_labels_true_mixed, xmin_labels_pred_mixed, average='weighted')
        xmin_rows_jaccard_mixed.append({'nb_xmin': nb_xmin, 'xmin_score_jaccard_mixed': xmin_score_jaccard_mixed})

# Build final AMI dataframe, sort by nb_xmin and reset indices
df_xmin_ami_mixed = pd.DataFrame(xmin_rows_ami_mixed).sort_values(by='nb_xmin', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_xmin_ami_mixed = df_xmin_ami_mixed.groupby('nb_xmin', as_index=False)['xmin_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by nb_xmin and reset indices
df_xmin_jaccard_mixed = pd.DataFrame(xmin_rows_jaccard_mixed).sort_values(by='nb_xmin', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_xmin_jaccard_mixed = df_xmin_jaccard_mixed.groupby('nb_xmin', as_index=False)['xmin_score_jaccard_mixed'].mean()

# Melt expression table
xmin_exp_df_melted_mixed = pd.melt(xmin_exp_df_mixed, id_vars='nb_xmin')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_xmin_ami_mixed['nb_xmin'], 0, 1.5), y=df_xmin_ami_mixed['xmin_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_xmin_ami_mixed['nb_xmin'], 0, 1.5), y=avg_xmin_ami_mixed['xmin_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_xmin_jaccard_mixed['nb_xmin'], 0, 1.5), y=df_xmin_jaccard_mixed['xmin_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_xmin_jaccard_mixed['nb_xmin'], 0, 1.5), y=avg_xmin_jaccard_mixed['xmin_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Minimum cell per phenotype per sample', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=xmin_exp_df_melted_mixed, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of minimum number of cells per sample per phenotype on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_xmin.jpg', dpi=600)
plt.close()
print('Saved mixed_xmin.jpg')


# Plot AMI/Jaccard similarity across maxmarkers - negative and positive markers
# Initialise objects
maxmarkers_dir_mixed = '../test_results/mixed_batch_sample_std_maxmarkers_test/'  # Folder
maxmarkers_files_mixed = [f for f in os.listdir(maxmarkers_dir_mixed) if f.endswith('.tsv')]  # tsv files
maxmarkers_rows_ami_mixed = []  # AMI data list
maxmarkers_rows_jaccard_mixed = []  # Jaccard data list
maxmarkers_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in maxmarkers_files_mixed:
    # Get std value
    nb_maxmarkers = float(file.replace('mixed_cell_expression_4batches_4samples_std0.75_', '').replace('markers_annotated.tsv', ''))
    # Build path
    maxmarkers_file_mixed = os.path.join(maxmarkers_dir_mixed, file)
    # Load file
    maxmarkers_res_mixed = pd.read_csv(maxmarkers_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    maxmarkers_labels_true_mixed = maxmarkers_res_mixed['cell_phntp_full']
    # Get expression data
    maxmarkers_exp_mixed = maxmarkers_res_mixed.loc[:, markers]
    maxmarkers_exp_mixed['nb_maxmarkers'] = nb_maxmarkers
    # Add it to expression table
    maxmarkers_exp_df_mixed = pd.concat([maxmarkers_exp_df_mixed, maxmarkers_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    maxmarkers_labels_col_mixed = [col for col in maxmarkers_res_mixed.columns if 'Phenotypes_' in col]
    for annot in maxmarkers_labels_col_mixed:
        maxmarkers_labels_pred_mixed = maxmarkers_res_mixed[annot]
        maxmarkers_score_ami_mixed = adjusted_mutual_info_score(maxmarkers_labels_true_mixed, maxmarkers_labels_pred_mixed)
        maxmarkers_rows_ami_mixed.append({'nb_maxmarkers': nb_maxmarkers, 'maxmarkers_score_ami_mixed': maxmarkers_score_ami_mixed})
        maxmarkers_score_jaccard_mixed = jaccard_score(maxmarkers_labels_true_mixed, maxmarkers_labels_pred_mixed, average='weighted')
        maxmarkers_rows_jaccard_mixed.append({'nb_maxmarkers': nb_maxmarkers, 'maxmarkers_score_jaccard_mixed': maxmarkers_score_jaccard_mixed})

# Build final AMI dataframe, sort by nb_maxmarkers and reset indices
df_maxmarkers_ami_mixed = pd.DataFrame(maxmarkers_rows_ami_mixed).sort_values(by='nb_maxmarkers', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_maxmarkers_ami_mixed = df_maxmarkers_ami_mixed.groupby('nb_maxmarkers', as_index=False)['maxmarkers_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by nb_maxmarkers and reset indices
df_maxmarkers_jaccard_mixed = pd.DataFrame(maxmarkers_rows_jaccard_mixed).sort_values(by='nb_maxmarkers', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_maxmarkers_jaccard_mixed = df_maxmarkers_jaccard_mixed.groupby('nb_maxmarkers', as_index=False)['maxmarkers_score_jaccard_mixed'].mean()

# Melt expression table
maxmarkers_exp_df_melted_mixed = pd.melt(maxmarkers_exp_df_mixed, id_vars='nb_maxmarkers')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_maxmarkers_ami_mixed['nb_maxmarkers'], 0, 0.05), y=df_maxmarkers_ami_mixed['maxmarkers_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_maxmarkers_ami_mixed['nb_maxmarkers'], 0, 0.05), y=avg_maxmarkers_ami_mixed['maxmarkers_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_maxmarkers_jaccard_mixed['nb_maxmarkers'], 0, 0.05), y=df_maxmarkers_jaccard_mixed['maxmarkers_score_jaccard_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_maxmarkers_jaccard_mixed['nb_maxmarkers'], 0, 0.05), y=avg_maxmarkers_jaccard_mixed['maxmarkers_score_jaccard_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Maximum number of markers per phenotype', ylabel='AMI / Jaccard similarity', ylim=[-0.05, 1.1])
ax[0].legend(loc='center right', labels=['AMI', 'Mean AMI', 'Jaccard similarity', 'Mean Jaccard similarity'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=maxmarkers_exp_df_melted_mixed, x='value', hue='variable', fill=True,
                common_norm=False, alpha=0.1, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Marker')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of maximum markers used in phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_maxmarkers.jpg', dpi=600)
plt.close()
print('Saved mixed_maxmarkers.jpg')


# Plot AMI with and without KNN - only negative markers
# Initialise objects
knn_dir_mixed = '../test_results/mixed_fake_std_test/'  # Folder
knn_files_mixed = [f for f in os.listdir(knn_dir_mixed) if f.endswith('.tsv')]  # tsv files
knn_rows_ami_mixed = []  # AMI data list
knn_rows_ami2_mixed = []  # AMI KNN data list
knn_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in knn_files_mixed:
    # Get std value
    std = float(file.replace('mixed_cell_expression_fake_std', '').replace('_annotated.tsv', ''))
    # Build path
    knn_file_mixed = os.path.join(knn_dir_mixed, file)
    # Load file
    knn_res_mixed = pd.read_csv(knn_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    knn_labels_true = knn_res_mixed['cell_phntp_full']
    # Get expression data
    knn_exp_mixed = knn_res_mixed.loc[:, markers]
    knn_exp_mixed['std'] = std
    # Add it to expression table
    knn_exp_df_mixed = pd.concat([knn_exp_df_mixed, knn_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    knn_labels_col_mixed = [col for col in knn_res_mixed.columns if 'Phenotypes_' in col]
    knn_labels_col2_mixed = [col for col in knn_res_mixed.columns if 'KNN_phenotype_' in col]
    for annot, annot2 in zip(knn_labels_col_mixed, knn_labels_col2_mixed):
        knn_labels_pred_mixed = knn_res_mixed[annot]
        knn_score_ami_mixed = adjusted_mutual_info_score(knn_labels_true, knn_labels_pred_mixed)
        knn_rows_ami_mixed.append({'std': std, 'knn_score_ami': knn_score_ami_mixed})
        knn_labels_pred2_mixed = knn_res_mixed[annot2]
        knn_score_ami2_mixed = adjusted_mutual_info_score(knn_labels_true, knn_labels_pred2_mixed)
        knn_rows_ami2_mixed.append({'std': std, 'knn_score_ami': knn_score_ami2_mixed})

# Build final AMI dataframe, sort by std and reset indices
df_knn_ami_mixed = pd.DataFrame(knn_rows_ami_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_knn_ami_mixed = df_knn_ami_mixed.groupby('std', as_index=False)['knn_score_ami'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_knn_ami2_mixed = pd.DataFrame(knn_rows_ami2_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_knn_ami2_mixed = df_knn_ami2_mixed.groupby('std', as_index=False)['knn_score_ami'].mean()

# Melt expression table
knn_exp_df_melted_mixed = pd.melt(knn_exp_df_mixed, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_knn_ami_mixed['std'], 0.05, 0.03), y=df_knn_ami_mixed['knn_score_ami'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_knn_ami_mixed['std'], 0.05, 0.03), y=avg_knn_ami_mixed['knn_score_ami'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_knn_ami2_mixed['std'], 0.05, 0.03), y=df_knn_ami2_mixed['knn_score_ami'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_knn_ami2_mixed['std'], 0.05, 0.03), y=avg_knn_ami2_mixed['knn_score_ami'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'AMI post-KNN', 'Mean AMI post-KNN'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=knn_exp_df_melted_mixed, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of KNN-reclassification on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_fake_knn.jpg', dpi=600)
plt.close()
print('Saved mixed_fake_knn.jpg')


# Plot AMI with and without unidentified cells - only negative markers
# Initialise objects
no_ident_dir_mixed = '../test_results/mixed_batch_sample_std_test/'  # Folder
no_ident_files_mixed = [f for f in os.listdir(no_ident_dir_mixed) if f.endswith('.tsv')]  # tsv files
no_ident_rows_ami_mixed = []  # AMI data list
no_ident_rows_ami2_mixed = []  # AMI no unidentified data list
no_ident_exp_df_mixed = pd.DataFrame()  # Expression data frame

# Fill dataframe
for file in no_ident_files_mixed:
    # Get std value
    std = float(file.replace('mixed_cell_expression_batch_sample_std', '').replace('_annotated.tsv', ''))
    # Build path
    no_ident_file_mixed = os.path.join(no_ident_dir_mixed, file)
    # Load file
    no_ident_res_mixed = pd.read_csv(no_ident_file_mixed, sep='\t', index_col=0)
    # Get true cell types
    no_ident_labels_true_mixed = no_ident_res_mixed['cell_phntp_full']
    # Get expression data
    no_ident_exp_mixed = no_ident_res_mixed.loc[:, markers]
    no_ident_exp_mixed['std'] = std
    # Add it to expression table
    no_ident_exp_df_mixed = pd.concat([no_ident_exp_df_mixed, no_ident_exp_mixed])
    # Get columns containing annotations (several optimal combinations --> several columns)
    no_ident_labels_col_mixed = [col for col in no_ident_res_mixed.columns if 'Phenotypes_' in col]
    for annot in no_ident_labels_col_mixed:
        no_ident_labels_pred_mixed = no_ident_res_mixed[annot]
        no_ident_score_ami_mixed = adjusted_mutual_info_score(no_ident_labels_true_mixed, no_ident_labels_pred_mixed)
        no_ident_rows_ami_mixed.append({'std': std, 'no_ident_score_ami_mixed': no_ident_score_ami_mixed})
    # Get table without unidentified cells
    no_ident_res2_mixed = no_ident_res_mixed[no_ident_res_mixed['KNN_annotations_1'] != 'Other 42'].copy()
    # Get true cell types
    no_ident_labels_true2_mixed = no_ident_res2_mixed['cell_phntp_full']
    # Get columns containing annotations (several optimal combinations --> several columns)
    no_ident_labels_col2_mixed = [col for col in no_ident_res2_mixed.columns if 'Phenotypes_' in col]
    for annot2 in no_ident_labels_col2_mixed:
        no_ident_labels_pred2_mixed = no_ident_res2_mixed[annot2]
        no_ident_score_ami2_mixed = adjusted_mutual_info_score(no_ident_labels_true2_mixed, no_ident_labels_pred2_mixed)
        no_ident_rows_ami2_mixed.append({'std': std, 'no_ident_score_ami2_mixed': no_ident_score_ami2_mixed})

# Build final AMI dataframe, sort by std and reset indices
df_no_ident_ami_mixed = pd.DataFrame(no_ident_rows_ami_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_no_ident_ami_mixed = df_no_ident_ami_mixed.groupby('std', as_index=False)['no_ident_score_ami_mixed'].mean()

# Build final ARI dataframe, sort by std and reset indices
df_no_ident_ami2_mixed = pd.DataFrame(no_ident_rows_ami2_mixed).sort_values(by='std', ignore_index=True)

# Calculate average AMI in case there are several combinations
avg_no_ident_ami2_mixed = df_no_ident_ami2_mixed.groupby('std', as_index=False)['no_ident_score_ami2_mixed'].mean()

# Melt expression table
no_ident_exp_df_melted_mixed = pd.melt(no_ident_exp_df_mixed, id_vars='std')

# Plot figures
plt.clf()  # Make sure there are no underlying figure
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [.4, .6]})  # To create figures side by side

# AMI Scatterplot
g = sns.scatterplot(x=jitter(df_no_ident_ami_mixed['std'], 0.05, 0.03), y=df_no_ident_ami_mixed['no_ident_score_ami_mixed'], color='lightblue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_no_ident_ami_mixed['std'], 0.05, 0.03), y=avg_no_ident_ami_mixed['no_ident_score_ami_mixed'], color='blue', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(df_no_ident_ami2_mixed['std'], 0.05, 0.03), y=df_no_ident_ami2_mixed['no_ident_score_ami2_mixed'], color='orange', linewidth=0, ax=ax[0])
g = sns.scatterplot(x=jitter(avg_no_ident_ami2_mixed['std'], 0.05, 0.03), y=avg_no_ident_ami2_mixed['no_ident_score_ami2_mixed'], color='red', linewidth=0, ax=ax[0])
g.set(xlabel='Marker distributions std', ylabel='AMI', ylim=[-0.05, 1.1])
ax[0].legend(loc='upper right', labels=['AMI', 'Mean AMI', 'AMI w.o unIDed', 'Mean AMI w.o unIDed'])
ax[0].title.set_text('Similarity scores')

# Expression density plot
h = sns.kdeplot(data=no_ident_exp_df_melted_mixed, x='value', hue='std', fill=True,
                common_norm=False, alpha=0.4, palette='crest', ax=ax[1])
h.set(xlabel='ADT expression')
ax[1].legend_.set_title('Standard deviation')
ax[1].title.set_text('Distribution of markers expression')
fig.figure.suptitle('Impact of unidentified cells on phenotype identification\n(Positive and negative markers)')
fig.tight_layout()
fig.figure.savefig('mixed_fake_unidentified.jpg', dpi=600)
plt.close()
print('Saved mixed_fake_unidentified.jpg')
