#%%
import os
import torch
import imageio
import numpy as np
import pandas as pd
import chromatinhd as chd

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# %%
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc_backup"
dataset_name = "myeloid"
dataset_name = "simulated"
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

latent_time = pd.read_csv(folder_data_preproc / 'MV2_latent_time_myeloid.csv')
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
dir_csv = folder_data_preproc / "likelihood_continuous_128"
dir_plot_quantile = folder_data_preproc / "plots/likelihood_quantile"
# dir_plot_continuous = folder_data_preproc / "plots/likelihood_continuous"
# dir_plot_continuous_128 = folder_data_preproc / "plots/likelihood_continuous_128"
# dir_plot_combined = folder_data_preproc / "plots/cutsites_likelihood_continuous"

# global_vars = globals()
# dirs = {key: value for key, value in global_vars.items() if key.startswith('dir_')}
# [os.makedirs(value, exist_ok=True) for value in dirs.values()]

def plot_cutsites(df, gene, n_fragments, directory):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Latent Time', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')

    directory = folder_data_preproc / 'plots' / f"cutsites_{directory}"
    os.makedirs(directory, exist_ok=True)

    plt.savefig(directory / f'{gene}.png')

def plot_cutsites_histo(df, df_long, gene, n_fragments):
    fig, axs = plt.subplots(figsize=(15, 10), ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    ax_hist = axs[0]
    ax_hist.hist(df['rank'], bins=100, orientation='horizontal')
    ax_hist.set_xlabel('n cells')
    ax_hist.set_ylabel('Rank')
    ax_hist.set_ylim([0, 1])
    ax_hist.invert_xaxis()

    ax_scatter = axs[1]
    ax_scatter.scatter(df_long['x'], df_long['y'], s=1, marker='s', color='black')
    ax_scatter.set_xlabel('Position')
    ax_scatter.set_ylabel('Latent Time')
    ax_scatter.set_xlim([0, 1])
    ax_scatter.set_ylim([0, 1])
    ax_scatter.set_facecolor('white')

    fig.suptitle(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)

    plt.savefig(folder_data_preproc / f'plots/cutsites_histo/{gene}.png')

def plot_model_continuous(gene, dir_data):
    file_name = folder_data_preproc / dir_data / f"{gene}.csv"
    probsx = np.loadtxt(file_name, delimiter=',')

    fig, ax = plt.subplots(figsize=(5, 5))
    heatmap = ax.imshow(probsx, cmap='RdBu_r', aspect='auto')
    cbar = plt.colorbar(heatmap)

    ax.set_title(f'Probs for gene_oi = {gene}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Latent Time')

    dir_plot_full = folder_data_preproc / "plots" / dir_data
    os.makedirs(dir_plot_full, exist_ok=True)
    file_name = dir_plot_full / f"{gene}.png"
    plt.savefig(file_name, dpi=200)

def plot_cutsites_model_continuous(df, gene, n_fragments, directory):
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
    x_tick_labels1 = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
    y_ticks1 = np.linspace(0, 1, num=6)
    y_tick_labels1 = ['0', '0.2', '0.4', '0.6', '0.8', '1.0'][::-1]

    ax2.set_xticks(x_ticks1 * (probsx.shape[1] - 1))
    ax2.set_xticklabels(x_tick_labels1)
    ax2.set_yticks(y_ticks1 * (probsx.shape[0] - 1))
    ax2.set_yticklabels(y_tick_labels1)

    ax2.set_xlabel('Position')
    ax2.set_ylabel('Latent Time')

    cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Probability of finding a cut site at given position and latent time')

    file_name = directory / f"{gene}.png"
    plt.savefig(file_name, dpi=300)

def plot_model_quantile(latent_torch, gene_oi, fragments):
    gene_id = fragments.var.index[gene_oi]
    probs = pd.read_csv(folder_data_preproc / f"likelihood_quantile/{gene_id}.csv")

    bins = np.linspace(0, 1, 500)
    binmids = (bins[1:] + bins[:-1])/2
    binsize = binmids[1] - binmids[0]
    pseudocoordinates = torch.linspace(0, 1, 1000)

    fig, axes = plt.subplots(probs.shape[0], 1, figsize=(20, 1*probs.shape[0]), sharex = True, sharey = True)
    for i, ax in zip(reversed(range(probs.shape[0])), axes):
        n_cells = latent_torch[:, i].sum()

        fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi)
        bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins)
        freq = round((fragments_oi.sum()/n_cells).item(), 3)

        ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0)
        ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#0000FF", lw = 2, zorder = 20)
        ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#FFFFFF", lw = 3, zorder = 10)
        ax.set_ylabel(f"{probs.iloc[i]['cluster']}\n freq={freq}", rotation = 0, ha = "right", va = "center")

    plt.savefig(dir_plot_quantile / (gene_id + ".png"))
    plt.close()

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
    print(x)
    gene = promoters.index[x]
    gene_ix = fragments.var.loc[gene]['ix']
    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]
    n_fragments = coordinates_sub.shape[0]

    tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
    df = pd.DataFrame(tens.numpy())
    df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
    df['height'] = 1

    df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)
    df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'rank', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')
    df_long = df_long.rename(columns={'position': 'x', 'rank': 'y'})
    
    plot_cutsites(df_long, gene, n_fragments, dataset_name)

    # for directory in dirs:
    #     plot_model_continuous(gene, directory)

        # plot_cutsites_histo(df, df_long, gene, n_fragments)
        # plot_cutsites_model_continuous(df_long, gene, n_fragments)
        # plot_model_quantile(latent_torch, x, fragments)

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
