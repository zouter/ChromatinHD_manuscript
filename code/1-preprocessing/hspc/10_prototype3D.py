#%%
import os
import torch
import imageio
import numpy as np
import pandas as pd
import chromatinhd as chd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name
promoter_name, window = "10k10k", np.array([-10000, 10000])

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)

# get latent time
latent_time = pd.read_csv(folder_data_preproc / 'MV2_latent_time_myeloid.csv')
# create ranking column
latent_time['rank_raw'] = latent_time['latent_time'].rank()
latent_time['rank'] = latent_time['rank_raw'] / latent_time.shape[0]

# get gene info
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
s_genes = info_genes_cells['s_genes'].dropna().tolist()
g2m_genes = info_genes_cells['g2m_genes'].dropna().tolist()
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()
# %%
# fragments.mapping object specifies from cell and gene for each fragment
mapping = fragments.mapping

# fragments.coordinates object specifies cutsites for each fragment
coordinates = fragments.coordinates

#%%
# normalize coordinates between 0 and 1
coordinates = coordinates + 10000
coordinates = coordinates / 20000

cutsite_gene = torch.bincount(mapping[:, 1]) * 2

#%%
# calculate the range that contains 90% of the data
sorted_tensor, _ = torch.sort(cutsite_gene)
ten_percent = cutsite_gene.numel() // 10
min_val, max_val = sorted_tensor[ten_percent], sorted_tensor[-ten_percent]

# set the style to match the standards of top-tier publications like Nature
sns.set_style("white")
sns.set_context("paper", font_scale=1.4)

# create a histogram using Matplotlib and Seaborn
fig, ax = plt.subplots()
sns.histplot(cutsite_gene.numpy(), bins=50, kde=False, stat="density", color="black", ax=ax)
ax.set_title("Number of cut sites per gene")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.axvline(min_val, color='r', linestyle='--')
ax.axvline(max_val, color='r', linestyle='--')

# set the y-axis limits to be between 0 and 1
ax.set_ylim([0, 0.0001])

# remove the top and right spines of the plot
sns.despine()

# someting is wrong with the plot yaxis, the values are too small
fig.savefig(folder_data_preproc / f'plots/n_cutsites.png')


#%%
for x in range(promoters.shape[0]):
    # pick a gene
    gene = promoters.index[x]

    # find out the gene_ix for chosen gene
    gene_ix = fragments.var.loc[gene]['ix']

    # use gene_ix to filter mapping and coordinates
    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]
    n_cutsites = coordinates_sub.shape[0]

    # create df
    tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
    df = pd.DataFrame(tens.numpy())
    df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
    df['height'] = 1

    # join latent time
    df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)

    # check latent time differences
    # df_lt = df.drop_duplicates(subset=['latent_time'])
    # df_lt.sort_values(by='latent_time', ascending=True, inplace=True)
    # df_lt['diff'] = df_lt['latent_time'] - df_lt['latent_time'].shift(1)
    # df_lt['diff'].hist(bins=200)

    # reshape
    df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'rank', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')
    df = df_long.rename(columns={'position': 'x', 'rank': 'y'})

    # Set figure size
    fig, ax = plt.subplots(figsize=(15, 15))

    # Create scatter plot with rectangular markers
    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')

    # Set plot title and axis labels
    ax.set_title(f"{gene} (cut sites = {2 * n_cutsites})", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Latent Time', fontsize=12)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Set plot background color to white
    ax.set_facecolor('white')

    plt.savefig(folder_data_preproc / f'plots/cutsites/{gene}.png')

print(f"Done! Plots saved to {folder_data_preproc / 'plots/cutsites'}")

#%%
for x in range(promoters.shape[0]):
    # pick a gene
    gene = promoters.index[x]

    # find out the gene_ix for chosen gene
    gene_ix = fragments.var.loc[gene]['ix']

    # Use gene_ix to filter mapping and coordinates
    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]
    n_cutsites = coordinates_sub.shape[0]

    # Create df
    tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
    df = pd.DataFrame(tens.numpy())
    df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
    df['height'] = 1

    # Join latent time
    df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)

    # Reshape
    df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'rank', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')
    df_long = df_long.rename(columns={'position': 'x', 'rank': 'y'})

    # Set figure size and create subplot grid
    fig, axs = plt.subplots(figsize=(15, 10), ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    # Create histogram subplot
    ax_hist = axs[0]
    ax_hist.hist(df['rank'], bins=100, orientation='horizontal')
    ax_hist.set_xlabel('n cells')
    ax_hist.set_ylabel('Rank')
    ax_hist.set_ylim([0, 1])
    ax_hist.invert_xaxis()

    # Create scatter plot subplot
    ax_scatter = axs[1]
    ax_scatter.scatter(df_long['x'], df_long['y'], s=1, marker='s', color='black')
    ax_scatter.set_xlabel('Position')
    ax_scatter.set_ylabel('Latent Time')
    ax_scatter.set_xlim([0, 1])
    ax_scatter.set_ylim([0, 1])
    ax_scatter.set_facecolor('white')

    # Set plot title
    fig.suptitle(f"{gene} (cut sites = {2 * n_cutsites})", fontsize=14)

    plt.savefig(folder_data_preproc / f'plots/cutsites_histo/{gene}.png')

print(f"Done! Plots saved to {folder_data_preproc / 'plots/cutsites_histo'}")


# TODO
# select hspc marker genes
# decrease margins
# increase font size

#%%
# create a figure with 4 rows and 4 columns
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(60, 60))

# iterate over the axes and plot each gene's cut site data
for i, ax in enumerate(axes.flat):
    if i >= promoters.shape[0]: # if there are fewer genes than axes, hide the extra axes
        ax.axis('off')
        continue
    
    # pick a gene
    gene = promoters.index[i]

    # find out the gene_ix for chosen gene
    gene_ix = fragments.var.loc[gene]['ix']

    # use gene_ix to filter mapping and coordinates
    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]
    n_cutsites = coordinates_sub.shape[0]

    # create df
    tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
    df = pd.DataFrame(tens.numpy())
    df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
    df['height'] = 1

    # join latent time
    df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)

    # reshape
    df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'rank', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')
    df = df_long.rename(columns={'position': 'x', 'rank': 'y'})

    # Create scatter plot with rectangular markers
    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')

    # Set plot title and axis labels
    ax.set_title(f"{gene} (cut sites = {2 * n_cutsites})", fontsize=12)
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Latent Time', fontsize=10)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Set plot background color to white
    ax.set_facecolor('white')

# adjust spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# save the figure
plt.savefig(folder_data_preproc / f'plots/cutsites_subplot.png')


#%%
image_dir = folder_data_preproc / 'plots/cutsites_histo'
output_gif = folder_data_preproc / 'plots/cutsites_histo.gif'

# Create a list of all the PNG images in the directory
images = []
for filename in os.listdir(image_dir)[:100]:
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        images.append(imageio.imread(image_path))

# Create the GIF from the list of images
imageio.mimsave(output_gif, images, fps=5)

# %%
tens = torch.tensor(df_long[['rank', 'latent_time', 'position']].values)
torch.save(tens, folder_data_preproc / 'tens_lt.pt')
