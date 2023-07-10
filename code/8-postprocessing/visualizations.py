#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import torch
import imageio
import numpy as np
import pandas as pd
import chromatinhd as chd
import chromatinhd_manuscript.plot_functions as pf

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# %%
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc_backup"
dataset_name = "simulated"
dataset_name = "myeloid"
fragment_dir = folder_data_preproc / f"fragments_{dataset_name}"
df_latent_file = folder_data_preproc / f"MV2_latent_time_{dataset_name}.csv"

# load data
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_file = promoter_name + "_simulated" if dataset_name == "simulated" else promoter_name
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_file + ".csv"), index_col = 0)

genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
s_genes = info_genes_cells['s_genes'].dropna().tolist()
g2m_genes = info_genes_cells['g2m_genes'].dropna().tolist()
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()

# fragments and latent time are for 2-lt_discrete
fragments = chd.data.Fragments(fragment_dir / promoter_name)
fragments.window = window
fragments.create_cut_data()

latent_time = pd.read_csv(folder_data_preproc / f'MV2_latent_time_{dataset_name}.csv')
latent_time['rank_raw'] = latent_time['latent_time'].rank()
latent_time['rank'] = latent_time['rank_raw'] / latent_time.shape[0]
latent_time['quantile'] = pd.qcut(latent_time['latent_time'], q=10, labels=False)

latent = pd.get_dummies(latent_time['quantile'], prefix='quantile')
latent_torch = torch.from_numpy(latent.values).to(torch.float)

# %%
coordinates = fragments.coordinates
coordinates = coordinates + 10000
coordinates = coordinates / 20000

mapping = fragments.mapping
mapping_cutsites = torch.bincount(mapping[:, 1]) * 2

#%%
# find dirs of interest
nbins = (32, )
nbins = (1024, )
nbins = (128, )
nbins = (128, 64, 32, )
data_source = "_simulated" if dataset_name == "simulated" else ""
pattern = f"likelihood_continuous{data_source}_{'_'.join(str(n) for n in nbins)}_fold_"
dirs = sorted([file for file in os.listdir(folder_data_preproc) if pattern in file])

#%%
for x in range(promoters.shape[0]):
    gene = promoters.index[x]
    gene_ix = fragments.var.loc[gene]['ix']
    print(x, gene, gene_ix)

    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]

    tens1 = torch.cat((coordinates_sub[:, 0].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens2 = torch.cat((coordinates_sub[:, 1].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens = torch.cat((tens1, tens2), dim=0)

    df = pd.DataFrame(tens.numpy())
    df.columns = ['x', 'cell_ix']
    df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)
    df = df.rename(columns={'rank': 'y'})
    
    directory = folder_data_preproc / 'plots' / f"cutsites_{dataset_name}"
    pf.cutsites(gene, df, directory, show=False)

    # tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
    # df = pd.DataFrame(tens.numpy())
    # df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
    # df['height'] = 1

    # df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)
    # df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'rank', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')
    # df_long = df_long.rename(columns={'position': 'x', 'rank': 'y'})
    
    # directory = folder_data_preproc / 'plots' / f"cutsites_histo_{dataset_name}"
    # pf.cutsites_histo(gene, df, df_long, n_fragments, directory, show=False)
    # directory = folder_data_preproc / f"likelihood_quantile_{dataset_name}"
    # pf.model_quantile(x, latent_torch, fragments, directory, show=False)

#%%
# calculate the range that contains 90% of the data
# sorted_tensor, _ = torch.sort(mapping_cutsites)
# ten_percent = mapping_cutsites.numel() // 10
# min_val, max_val = sorted_tensor[ten_percent], sorted_tensor[-ten_percent]

# values, bins, _ = plt.hist(mapping_cutsites.numpy(), bins=50, color="blue", alpha=0.75)
# percentages = values / np.sum(values) * 100

# sns.set_style("white")
# sns.set_context("paper", font_scale=1.4)
# fig, ax = plt.subplots(dpi=300)
# ax.bar(bins[:-1], percentages, width=np.diff(bins), color="blue", alpha=0.75)
# ax.set_title("Percentage of values per bin")
# ax.set_xlabel("Number of cut sites")
# ax.set_ylabel("%")
# ax.axvline(min_val, color='r', linestyle='--')
# ax.axvline(max_val, color='r', linestyle='--')

# sns.despine()

# fig.savefig(folder_data_preproc / f'plots/n_cutsites.png')

#%%
def plot_cutsites_model_continuous(df, gene, n_fragments, dir_csv, directory):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [1, 1]})

    # Modify subplot configuration
    plt.subplots_adjust(hspace=0.1, bottom=0.12)

    # Plot cut sites
    ax1.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax1.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Latent Time', fontsize=12)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_facecolor('white')

    # Plot evaluated probabilities
    file_name = dir_csv / f"{gene}.csv"
    probsx = np.loadtxt(file_name, delimiter=',')

    heatmap = ax2.imshow(probsx, cmap='RdBu_r', aspect='auto')

    x_ticks1 = np.linspace(0, 1, num=6)
    y_ticks1 = np.linspace(0, 1, num=6)
    x_tick_labels1 = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
    y_tick_labels1 = ['0', '0.2', '0.4', '0.6', '0.8', '1.0'][::-1]

    ax2.set_xticks(x_ticks1 * (probsx.shape[1] - 1))
    ax2.set_yticks(y_ticks1 * (probsx.shape[0] - 1))
    ax2.set_xticklabels(x_tick_labels1)
    ax2.set_yticklabels(y_tick_labels1)

    ax2.set_xlabel('Position')
    ax2.set_ylabel('Latent Time')

    cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Probability of finding a cut site at given position and latent time')

    file_name = directory / f"{gene}.png"
    plt.savefig(file_name, dpi=300)

#%%
# TODO
# select hspc marker genes

# #%%
# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(60, 60))

# for i, ax in enumerate(axes.flat):
#     if i >= promoters.shape[0]:
#         # if there are fewer genes than axes, hide the extra axes
#         ax.axis('off')
#         continue
    
#     gene = promoters.index[i]
#     gene_ix = fragments.var.loc[gene]['ix']
#     mask = mapping[:,1] == gene_ix
#     mapping_sub = mapping[mask]
#     coordinates_sub = coordinates[mask]
#     n_fragments = coordinates_sub.shape[0]

#     tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
#     df = pd.DataFrame(tens.numpy())
#     df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
#     df['height'] = 1


#     df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)
#     df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'rank', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')
#     df_long = df_long.rename(columns={'position': 'x', 'rank': 'y'})

#     ax.scatter(df_long['x'], df_long['y'], s=1, marker='s', color='black')
#     ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=12)
#     ax.set_xlabel('Position', fontsize=10)
#     ax.set_ylabel('Latent Time', fontsize=10)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_facecolor('white')

# # adjust spacing between subplots
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.savefig(folder_data_preproc / f'plots/cutsites_subplot.png')
 
# #%%
# dir_plot_continuous = folder_data_preproc / "plots/evaluate_pseudo_continuous_3D"

# for gene_oi in range(promoters.shape[0]):
#     file_name = dir_csv / f"tensor_gene_oi_{gene_oi}.csv"
#     probsx = np.loadtxt(file_name, delimiter=',')

#     fig = go.Figure(data=[go.Surface(z=probsx)])
#     fig.update_layout(title=f'Probs for gene_oi = {gene_oi}', template='plotly_white')
#     fig.show()
#     if gene_oi == 3:
#         break
# %%
