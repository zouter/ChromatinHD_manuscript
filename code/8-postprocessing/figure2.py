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

from matplotlib.gridspec import GridSpec

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
annotation = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]
pseudocoordinates = torch.linspace(0, 1, 1000)
#%%
lineage_gene = {'lin_myeloid': 'MPO', 'lin_erythroid': 'HBB', 'lin_platelet': 'CD74'}
lineage_objects = {}

for lineage_name, gene_name in lineage_gene.items():

    dataset_name = lineage_name.replace('lin_', '')
    gene_name = gene_name

    fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}/10k10k")
    fragments.window = window
    fragments.create_cut_data()
    gene_id = adata.var.loc[gene_name]['Accession']
    gene_ix = fragments.var.index.get_loc(gene_id)

    lineage = info_genes_cells[f'lin_{dataset_name}'].dropna().tolist()
    df_latent = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv", index_col = 0)
    latent = pd.get_dummies(df_latent['celltype'])
    latent = latent[lineage]
    latent_torch = torch.from_numpy(latent.values).to(torch.float)
    adata_oi = adata[list(df_latent.index), gene_name]
    df_latent[gene_name] = pd.DataFrame(adata_oi.X, index=adata_oi.obs.index, columns=[gene_name])
    exp_xmin, exp_xmax = df_latent[gene_name].min(), df_latent[gene_name].max()

    dir_likelihood = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_{dataset_name}_128_64_32_fold_0"
    probs = pd.read_csv(dir_likelihood / (gene_id + '.csv'), index_col=0)

    df_bincounts = pd.DataFrame()
    for i, celltype in enumerate(latent.columns):
        fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_ix)
        bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
        n_cells = latent_torch[:, i].sum()
        bincounts = bincounts / n_cells * len(bins)

        df_bincounts[celltype] = bincounts

    print(lineage_name)
    lineage_objects[lineage_name] = {
        'df_latent': df_latent,
        'latent_torch': latent_torch,
        'fragments': fragments,
        'celltypes': lineage,
        'gene_name': gene_name,
        'gene_ix': gene_ix,
        'probs': probs,
        'df_bincounts': df_bincounts,
        'exp_xmin': exp_xmin,
        'exp_xmax': exp_xmax,
    }

# %%

# Create the grid layout
height, width = 10, 10
fig = plt.figure(figsize=(height, width))

rows, cols = 12, 20
grid = GridSpec(rows, cols, figure=fig)

lineages = [x for x in info_genes_cells.columns if 'lin' in x]
lineages_dict = {x: info_genes_cells[x].dropna().tolist() for x in lineages}
df = pd.DataFrame([(lineage, cell_type) for lineage, cell_types in lineages_dict.items() for cell_type in cell_types], columns=['lineage', 'cell type'])
df['group'] = df['lineage'].map({lineage: i for i, lineage in enumerate(lineages)})
df['index'] = df.groupby('group').cumcount()
df['plot_row'] = range(df.shape[0]) + df['group']
df['plot_row'] = df.groupby('group')['plot_row'].transform(lambda x: x.sort_values(ascending=False).values)
df = df.sort_values('plot_row').reset_index(drop=True)
# column with yes or no. yes for min values in index per group
df['xaxis'] = df.groupby('group')['index'].transform(lambda x: x == x.min())
df['title'] = df.groupby('group')['index'].transform(lambda x: x == x.max())

col_accessibility = [grid[i, 1:10] for i in df['plot_row']]
col_expression = [grid[i, 11:15] for i in df['plot_row']]
col_lt = [grid[i, 16:20] for i in df['plot_row']]

def plot_accessibility(fig, gridspec, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation, ymax):
    ax_object = fig.add_subplot(gridspec)
    ax_object.bar(binmids, bincounts, width=binsize, color="#888888", lw=0)
    ax_object.plot(pseudocoordinates.numpy(), data['probs'].iloc[index, :], label=index, color=annotation[data['celltypes'][index]], lw=2, zorder=20)
    ax_object.plot(pseudocoordinates.numpy(), data['probs'].iloc[index, :], label=index, color="#FFFFFF", lw=3, zorder=10)
    ax_object.set_ylabel(f"{data['probs'].index[index]}  \n n={int(n_cells)}  ", rotation=0, ha="right", va="center")
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.set_ylim(0, ymax)
    return ax_object

def plot_expression(fig, gridspec, expression, celltype, annotation, exp_xmin, exp_xmax):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(exp_xmin * 1.05, exp_xmax * 1.05)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    ax_object.get_yaxis().set_visible(False)
    return ax_object

def plot_lt(fig, gridspec, lt, celltype, annotation):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(lt, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(-0.05, 1.05)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    ax_object.get_yaxis().set_visible(False)
    return ax_object

for lineage, celltype, index, plot_row, xaxis, title, c1, c2, c3 in zip(df['lineage'], df['cell type'], df['index'], df['plot_row'], df['xaxis'], df['title'], col_accessibility, col_expression, col_lt):
    print(lineage, celltype, index, plot_row, c1, c2, c3)

    data = lineage_objects[lineage]
    print(data['gene_name'], data['celltypes'])

    # 1. data for accessibility
    # fragments_oi = (data['latent_torch'][data['fragments'].cut_local_cell_ix, index] != 0) & (data['fragments'].cut_local_gene_ix == data['gene_ix'])
    # bincounts, _ = np.histogram(data['fragments'].cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
    bincounts = data['df_bincounts'][celltype]
    ymax = data['df_bincounts'].max().max()
    print(ymax)
    n_cells = data['latent_torch'][:, index].sum()

    # 2. data for expression
    expression = data['df_latent'].loc[data['df_latent']['celltype'] == data['celltypes'][index], data['gene_name']].values
    
    # 3. data for latent time
    lt = data['df_latent'].loc[data['df_latent']['celltype'] == celltype, 'latent_time'].values

    ax_1 = plot_accessibility(fig, c1, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation, ymax)
    ax_2 = plot_expression(fig, c2, expression, data['celltypes'][index], annotation, data['exp_xmin'], data['exp_xmax'])
    ax_3 = plot_lt(fig, c3, lt, data['celltypes'][index], annotation)

    if title == True:
        ax_1.set_title(f"{lineage_gene[lineage]}: Accessibility")
        ax_2.set_title(f"{lineage_gene[lineage]}: Expression")
        ax_3.set_title("Latent Time")
    
    if xaxis == False:
        ax_1.set_xticklabels([])
        ax_2.set_xticklabels([])
        ax_3.set_xticklabels([])

x1, x2, x3 = 0.05, 0.52, 0.73
y1, y2, y3 = 0.905, 0.58, 0.32

fig.text(x1, y1, 'A', fontsize=16, fontweight='bold', va='top')
fig.text(x2, y1, 'B', fontsize=16, fontweight='bold', va='top')
fig.text(x3, y1, 'C', fontsize=16, fontweight='bold', va='top')

fig.text(x1, y2, 'D', fontsize=16, fontweight='bold', va='top')
fig.text(x2, y2, 'E', fontsize=16, fontweight='bold', va='top')
fig.text(x3, y2, 'F', fontsize=16, fontweight='bold', va='top')

fig.text(x1, y3, 'G', fontsize=16, fontweight='bold', va='top')
fig.text(x2, y3, 'H', fontsize=16, fontweight='bold', va='top')
fig.text(x3, y3, 'I', fontsize=16, fontweight='bold', va='top')

fig.savefig(folder_data_preproc / 'plots' / "fig2.pdf", bbox_inches='tight', pad_inches=0.01)

# %%
# import IPython
# if IPython.get_ipython() is not None:
#     IPython.get_ipython().magic('load_ext autoreload')
#     IPython.get_ipython().magic('autoreload 2')

# import torch
# import pickle
# import numpy as np
# import pandas as pd
# import chromatinhd as chd
# import matplotlib.pyplot as plt
# import chromatinhd_manuscript.plot_functions as pf

# from matplotlib import cm
# from matplotlib.gridspec import GridSpec
# from matplotlib.colors import ListedColormap
# from matplotlib.patches import FancyArrowPatch

# # %%
# folder_root = chd.get_output()
# folder_data_preproc = folder_root / "data" / "hspc"
# dataset_name_sub = "MV2"
# dataset_name = 'myeloid'

# info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
# lin_myeloid = info_genes_cells['lin_myeloid'].dropna().tolist()
# lin_erythroid = info_genes_cells['lin_erythroid'].dropna().tolist()
# lin_platelet = info_genes_cells['lin_platelet'].dropna().tolist()

# promoter_name, window = "10k10k", np.array([-10000, 10000])

# dir_myeloid = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_myeloid_128_64_32_fold_0"
# dir_erythroid = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_erythroid_128_64_32_fold_0"
# dir_platelet = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_platelet_128_64_32_fold_0"

# #%%
# df_myeloid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_myeloid.csv", index_col = 0)
# df_erythroid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_erythroid.csv", index_col = 0)
# df_platelet = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_platelet.csv", index_col = 0)

# #%%
# transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
# adata = transcriptome.adata

# # %%
# adata_myeloid = adata[list(df_myeloid.index)]
# adata_erythroid = adata[list(df_erythroid.index)]
# adata_platelet = adata[list(df_platelet.index)]

# # %%
# fragments_myeloid = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_myeloid/10k10k")
# fragments_erythroid = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_erythroid/10k10k")
# fragments_platelet = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_platelet/10k10k")

# annotation = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

# # %%
# gene_name = 'MPO'
# gene_id = adata.var.loc[gene_name]['Accession']
# gene_ix = fragments_myeloid.var.index.get_loc(gene_id)

# # %%
# probs = pd.read_csv(dir_myeloid / (gene_id + '.csv'), index_col=0)

# fragments_myeloid.window = window
# fragments_myeloid.create_cut_data()

# latent = pd.get_dummies(df_myeloid['celltype'])
# latent = latent[lin_myeloid]
# latent_torch = torch.from_numpy(latent.values).to(torch.float)

# bins = np.linspace(0, 1, 500)
# binmids = (bins[1:] + bins[:-1])/2
# binsize = binmids[1] - binmids[0]
# pseudocoordinates = torch.linspace(0, 1, 1000)

# # %%
# adata_gene = adata_myeloid[:, gene_name]
# adata_gene = pd.DataFrame(adata_gene.X, index=adata_myeloid.obs.index, columns=[gene_name])
# df_myeloid[gene_name] = adata_gene
# exp_xmin, exp_xmax = df_myeloid[gene_name].min(), df_myeloid[gene_name].max()

# # %%
# # Create the grid layout
# height, width = 10, 10
# fig = plt.figure(figsize=(height, width))

# rows, cols = 12, 20
# grid = GridSpec(rows, cols, figure=fig)

# # Calculate the number of subplots required
# num_plots_mye = len(lin_myeloid)
# num_plots_ery = len(lin_erythroid)
# num_plots_pla = len(lin_platelet)

# # Define the positions and sizes of subplots
# subplot_positions = {
#     "ax_1": [grid[i, 1:10] for i in range(num_plots_mye)],
#     "ax_2": [grid[i, 11:15] for i in range(num_plots_mye)],
#     "ax_3": [grid[i, 16:20] for i in range(num_plots_mye)],

#     "ax_4": [grid[i + num_plots_mye + 1, 1:10] for i in range(num_plots_ery)],
#     "ax_5": [grid[i + num_plots_mye + 1, 11:15] for i in range(num_plots_ery)],
#     "ax_6": [grid[i + num_plots_mye + 1, 16:20] for i in range(num_plots_ery)],

#     "ax_7": [grid[i + num_plots_mye + num_plots_ery + 2, 1:10] for i in range(num_plots_pla)],
#     "ax_8": [grid[i + num_plots_mye + num_plots_ery + 2, 11:15] for i in range(num_plots_pla)],
#     "ax_9": [grid[i + num_plots_mye + num_plots_ery + 2, 16:20] for i in range(num_plots_pla)],
# }

# # Create the shared ax_1 outside the loop
# ax_1 = None

# # Loop through each cell type and populate its corresponding subplot
# for i, celltype in enumerate(probs.index):
#     print(i, celltype)

#     # Get the corresponding subplots for the current cell type
#     if i == 0:
#         ax_1 = fig.add_subplot(subplot_positions["ax_1"][num_plots_mye - 1 - i])
#     else:
#         ax_1 = fig.add_subplot(subplot_positions["ax_1"][num_plots_mye - 1 - i], sharey=ax_1)

#     ax_2 = fig.add_subplot(subplot_positions["ax_2"][num_plots_mye - 1 - i])
#     ax_3 = fig.add_subplot(subplot_positions["ax_3"][num_plots_mye - 1 - i])

#     fragments_oi = (latent_torch[fragments_myeloid.cut_local_cell_ix, i] != 0) & (fragments_myeloid.cut_local_gene_ix == gene_ix)
#     bincounts, _ = np.histogram(fragments_myeloid.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
#     n_cells = latent_torch[:, i].sum()

#     ax_1.bar(binmids, bincounts / n_cells * len(bins), width=binsize, color="#888888", lw=0)
#     ax_1.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label=i, color=annotation[celltype], lw=2, zorder=20)
#     ax_1.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label=i, color="#FFFFFF", lw=3, zorder=10)
#     ax_1.set_ylabel(f"{probs.index[i]}  \n n={int(n_cells)}  ", rotation=0, ha="right", va="center")

#     medianprops = dict(color=annotation[celltype], linewidth=1)
#     expression = df_myeloid.loc[df_myeloid['celltype'] == celltype, gene_name].values
#     ax_2.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
#     ax_2.set_xlim(exp_xmin*1.05, exp_xmax*1.05)

#     lt = df_myeloid.loc[df_myeloid['celltype'] == celltype, 'latent_time'].values
#     ax_3.boxplot(lt, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
#     ax_3.set_xlim(-0.05, 1.05)

#     # Remove the y-axis for ax_2 and ax_3
#     ax_2.get_yaxis().set_visible(False)
#     ax_3.get_yaxis().set_visible(False)

#     # Remove the top and right spines for ax_1
#     ax_1.spines['top'].set_visible(False)
#     ax_1.spines['right'].set_visible(False)

#     # Remove the top, right, and left spines for ax_2
#     ax_2.spines['top'].set_visible(False)
#     ax_2.spines['right'].set_visible(False)
#     ax_2.spines['left'].set_visible(False)

#     # Remove the top, right, and left spines for ax_3
#     ax_3.spines['top'].set_visible(False)
#     ax_3.spines['right'].set_visible(False)
#     ax_3.spines['left'].set_visible(False)

#     if i > 0:
#         ax_1.set_xticklabels([])
#         ax_2.set_xticklabels([])
#         ax_3.set_xticklabels([])
    
#     if i == num_plots_mye - 1:
#         ax_1.set_title(f"{gene_name}: Accessibility")
#         ax_2.set_title(f"{gene_name}: Expression")
#         ax_3.set_title("Latent Time")

# fig.text(0.15, 0.95, 'A', fontsize=16, fontweight='bold', va='top')
# fig.text(0.55, 0.95, 'B', fontsize=16, fontweight='bold', va='top')
# fig.text(0.74, 0.95, 'C', fontsize=16, fontweight='bold', va='top')

# fig.savefig(folder_data_preproc / 'plots' / "fig2_old.pdf", bbox_inches='tight', pad_inches=0.01)
# plt.show()