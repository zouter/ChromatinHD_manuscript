# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import torch
import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib.pyplot as plt
import chromatinhd_manuscript.plot_functions as pf

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"
dataset_name = 'myeloid'

info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
lin_myeloid = info_genes_cells['lin_myeloid'].dropna().tolist()
lin_erythroid = info_genes_cells['lin_erythroid'].dropna().tolist()
lin_platelet = info_genes_cells['lin_platelet'].dropna().tolist()

promoter_name, window = "10k10k", np.array([-10000, 10000])

dir_myeloid = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_myeloid_128_64_32_fold_0"
dir_erythroid = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_erythroid_128_64_32_fold_0"
dir_platelet = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_platelet_128_64_32_fold_0"

#%%
df_myeloid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_myeloid.csv", index_col = 0)
df_erythroid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_erythroid.csv", index_col = 0)
df_platelet = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_platelet.csv", index_col = 0)

#%%
transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

# %%
adata_myeloid = adata[list(df_myeloid.index)]
adata_erythroid = adata[list(df_erythroid.index)]
adata_platelet = adata[list(df_platelet.index)]

# %%
fragments_myeloid = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_myeloid/10k10k")
fragments_erythroid = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_erythroid/10k10k")
fragments_platelet = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_platelet/10k10k")

annotation = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

# %%
gene_name = 'MPO'
gene_id = adata.var.loc[gene_name]['Accession']
gene_ix = fragments_myeloid.var.index.get_loc(gene_id)

# %%
probs = pd.read_csv(dir_myeloid / (gene_id + '.csv'), index_col=0)

fragments_myeloid.window = window
fragments_myeloid.create_cut_data()

latent = pd.get_dummies(df_myeloid['celltype'])
latent = latent[lin_myeloid]
latent_torch = torch.from_numpy(latent.values).to(torch.float)

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]
pseudocoordinates = torch.linspace(0, 1, 1000)

# %%
adata_gene = adata_myeloid[:, gene_name]
adata_gene = pd.DataFrame(adata_gene.X, index=adata_myeloid.obs.index, columns=[gene_name])
df_myeloid[gene_name] = adata_gene
exp_xmin, exp_xmax = df_myeloid[gene_name].min(), df_myeloid[gene_name].max()

# %%
# Create the grid layout
height, width = 10, 10
fig = plt.figure(figsize=(height, width))

rows, cols = 12, 20
grid = GridSpec(rows, cols, figure=fig)

# Calculate the number of subplots required
num_plots_mye = len(lin_myeloid)
num_plots_ery = len(lin_erythroid)
num_plots_pla = len(lin_platelet)

# Define the positions and sizes of subplots
subplot_positions = {
    "ax_1": [grid[i, 1:10] for i in range(num_plots_mye)],
    "ax_2": [grid[i, 11:15] for i in range(num_plots_mye)],
    "ax_3": [grid[i, 16:20] for i in range(num_plots_mye)],

    "ax_4": [grid[i + num_plots_mye + 1, 1:10] for i in range(num_plots_ery)],
    "ax_5": [grid[i + num_plots_mye + 1, 11:15] for i in range(num_plots_ery)],
    "ax_6": [grid[i + num_plots_mye + 1, 16:20] for i in range(num_plots_ery)],

    "ax_7": [grid[i + num_plots_mye + num_plots_ery + 2, 1:10] for i in range(num_plots_pla)],
    "ax_8": [grid[i + num_plots_mye + num_plots_ery + 2, 11:15] for i in range(num_plots_pla)],
    "ax_9": [grid[i + num_plots_mye + num_plots_ery + 2, 16:20] for i in range(num_plots_pla)],
}

# Create the shared ax_1 outside the loop
ax_1 = None

# Loop through each cell type and populate its corresponding subplot
for i, celltype in enumerate(probs.index):
    print(i, celltype)

    # Get the corresponding subplots for the current cell type
    if i == 0:
        ax_1 = fig.add_subplot(subplot_positions["ax_1"][num_plots_mye - 1 - i])
    else:
        ax_1 = fig.add_subplot(subplot_positions["ax_1"][num_plots_mye - 1 - i], sharey=ax_1)

    ax_2 = fig.add_subplot(subplot_positions["ax_2"][num_plots_mye - 1 - i])
    ax_3 = fig.add_subplot(subplot_positions["ax_3"][num_plots_mye - 1 - i])

    fragments_oi = (latent_torch[fragments_myeloid.cut_local_cell_ix, i] != 0) & (fragments_myeloid.cut_local_gene_ix == gene_ix)
    bincounts, _ = np.histogram(fragments_myeloid.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
    n_cells = latent_torch[:, i].sum()

    ax_1.bar(binmids, bincounts / n_cells * len(bins), width=binsize, color="#888888", lw=0)
    ax_1.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label=i, color=annotation[celltype], lw=2, zorder=20)
    ax_1.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label=i, color="#FFFFFF", lw=3, zorder=10)
    ax_1.set_ylabel(f"{probs.index[i]}  \n n={int(n_cells)}  ", rotation=0, ha="right", va="center")

    medianprops = dict(color=annotation[celltype], linewidth=1)
    expression = df_myeloid.loc[df_myeloid['celltype'] == celltype, gene_name].values
    ax_2.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_2.set_xlim(exp_xmin*1.05, exp_xmax*1.05)

    lt = df_myeloid.loc[df_myeloid['celltype'] == celltype, 'latent_time'].values
    ax_3.boxplot(lt, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_3.set_xlim(-0.05, 1.05)

    # Remove the y-axis for ax_2 and ax_3
    ax_2.get_yaxis().set_visible(False)
    ax_3.get_yaxis().set_visible(False)

    # Remove the top and right spines for ax_1
    ax_1.spines['top'].set_visible(False)
    ax_1.spines['right'].set_visible(False)

    # Remove the top, right, and left spines for ax_2
    ax_2.spines['top'].set_visible(False)
    ax_2.spines['right'].set_visible(False)
    ax_2.spines['left'].set_visible(False)

    # Remove the top, right, and left spines for ax_3
    ax_3.spines['top'].set_visible(False)
    ax_3.spines['right'].set_visible(False)
    ax_3.spines['left'].set_visible(False)

    if i > 0:
        ax_1.set_xticklabels([])
        ax_2.set_xticklabels([])
        ax_3.set_xticklabels([])
    
    if i == num_plots_mye - 1:
        ax_1.set_title(f"{gene_name}: Accessibility")
        ax_2.set_title(f"{gene_name}: Expression")
        ax_3.set_title("Latent Time")

fig.text(0.15, 0.95, 'A', fontsize=16, fontweight='bold', va='top')
fig.text(0.55, 0.95, 'B', fontsize=16, fontweight='bold', va='top')
fig.text(0.74, 0.95, 'C', fontsize=16, fontweight='bold', va='top')

fig.savefig(folder_data_preproc / 'plots' / "fig2.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()

# %%
# fig, axes = plt.subplots(probs.shape[0], 1, figsize=(20, 1*probs.shape[0]), sharex = True, sharey = True)
# for i, ax in zip(reversed(range(probs.shape[0])), axes):
#     print(i, ax)

#     fragments_oi = (latent_torch[fragments_myeloid.cut_local_cell_ix, i] != 0) & (fragments_myeloid.cut_local_gene_ix == gene_ix)
#     bincounts, _ = np.histogram(fragments_myeloid.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins)
#     n_cells = latent_torch[:, i].sum()

#     ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0)
#     ax.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label = i, color = annotation[probs.index[i]], lw = 2, zorder = 20)
#     ax.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label = i, color = "#FFFFFF", lw = 3, zorder = 10)
#     ax.set_ylabel(f"{probs.index[i]}\n n={int(n_cells)}", rotation = 0, ha = "right", va = "center")
