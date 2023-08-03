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
from matplotlib.colors import ListedColormap

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
annotation = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

#%%
lineage_gene = {'lin_myeloid': 'MPO', 'lin_erythroid': 'HBB', 'lin_platelet': 'CD74'}
lineage_objects = {}

for lineage_name, gene_name in lineage_gene.items():

    dataset_name = lineage_name.replace('lin_', '')

    fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}/10k10k")
    fragments.window = window
    fragments.create_cut_data()
    gene_id = adata.var.loc[gene_name]['Accession']
    gene_ix = fragments.var.index.get_loc(gene_id)

    lineage = info_genes_cells[f'lin_{dataset_name}'].dropna().tolist()
    df_latent = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv", index_col = 0)
    df_latent['rank'] = (df_latent['latent_time'].rank() - 1) / (len(df_latent) - 1)

    adata_oi = adata[list(df_latent.index), gene_name]
    df_latent[gene_name] = pd.DataFrame(adata_oi.X, index=adata_oi.obs.index, columns=[gene_name])

    dir_likelihood = folder_data_preproc / f"{dataset_name_sub}_LC/lc_{dataset_name}_128_64_32_fold_0"
    probs = pd.read_csv(dir_likelihood / (gene_id + '.csv'), header=None)

    mapping = fragments.mapping
    coordinates = fragments.coordinates
    coordinates = (coordinates + 10000) / 20000

    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]

    tens1 = torch.cat((coordinates_sub[:, 0].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens2 = torch.cat((coordinates_sub[:, 1].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens = torch.cat((tens1, tens2), dim=0)

    df_latent['cell_ix'] = range(df_latent.shape[0])
    df_cutsites = pd.DataFrame(tens.numpy())
    df_cutsites.columns = ['x', 'cell_ix']
    df_cutsites = pd.merge(df_cutsites, df_latent, left_on='cell_ix', right_on='cell_ix')
    df_cutsites = df_cutsites.rename(columns={'rank': 'y'})
    df_cutsites = df_cutsites[['x', 'y']]

    print(lineage_name)
    lineage_objects[lineage_name] = {
        'df_latent': df_latent,
        'fragments': fragments,
        'cutsites': df_cutsites,
        'celltypes': lineage,
        'gene_name': gene_name,
        'gene_ix': gene_ix,
        'probs': probs,
    }

# %%
def plot_lt_heatmap(fig, gridspec, data, annotation_lin):
    ax_object = fig.add_subplot(gridspec)
    ax_object.imshow(data, cmap=ListedColormap(list(annotation_lin.values())), aspect='auto', interpolation='none')
    tick_positions = np.linspace(0, len(data)-1, 6)
    tick_labels = np.linspace(0, 1, 6)[::-1]
    tick_labels = [f'{tick:.1f}' for tick in tick_labels]
    ax_object.set_xticks([])
    ax_object.set_yticks(tick_positions, tick_labels)
    ax_object.yaxis.tick_right()
    ax_object.yaxis.set_label_position("right")
    return ax_object

def plot_exp_heatmap(fig, gridspec, data):
    ax_object = fig.add_subplot(gridspec)
    ax_object.imshow(data, cmap='RdBu_r', aspect='auto', interpolation='none')
    tick_positions = np.linspace(0, len(data)-1, 6)
    tick_labels = np.linspace(0, 1, 6)[::-1]
    tick_labels = [f'{tick:.1f}' for tick in tick_labels]
    ax_object.set_yticks(tick_positions, tick_labels)
    ax_object.set_yticklabels([])
    ax_object.set_xticks([])
    return ax_object

def plot_cutsites(fig, gridspec, df, title):
    ax_object = fig.add_subplot(gridspec)
    ax_object.scatter(df['x'], df['y'], s=0.3, edgecolors='none', color='black')
    ax_object.set_xlim([0, 1])
    ax_object.set_ylim([0, 1])
    tick_positions = np.linspace(0, 1, 5)
    tick_labels = ['-10k', '-5k', '0', '5k', '10k']
    ax_object.set_xticks(tick_positions, tick_labels)
    ax_object.set_title(title)
    return ax_object

def plot_likelihood(fig, gridspec, df, title):
    ax_object = fig.add_subplot(gridspec)
    ax_object.imshow(df, cmap='Blues', aspect='auto', interpolation='none')
    ax_object.set_yticklabels([])
    tick_positions = np.linspace(0, df.shape[1]-1, 5)
    tick_labels = ['-10k', '-5k', '0', '5k', '10k']
    ax_object.set_xticks(tick_positions, tick_labels)
    ax_object.set_title(title)
    return ax_object

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_legend(fig, gridspec, annotation_lin, celltypes, heatmap_ax, likelihood_ax):
    ax_object = fig.add_subplot(gridspec)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=annotation_lin[celltype], markersize=10, label=celltype) for celltype in celltypes[::-1]]
    ax_object.legend(handles=legend_elements, loc='upper center')

    im = heatmap_ax.get_images()[0]
    colorbar_ax = inset_axes(ax_object, width="50%", height="5%", loc='center')
    cbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
    cbar.set_label('Expression', size=12, labelpad=-50)

    im2 = likelihood_ax.get_images()[0]
    im2_ax = inset_axes(ax_object, width="50%", height="5%", loc='lower center')
    im2_cbar = fig.colorbar(im2, cax=im2_ax, orientation='horizontal')
    im2_cbar.set_label('Likelihood', size=12, labelpad=-50)

    ax_object.axis('off') 

    return ax_object

# Create the grid layout
height, width = 17, 15
fig = plt.figure(figsize=(width, height))

rows, cols = 68, 60
grid = GridSpec(rows, cols, figure=fig)

lineages = [x for x in info_genes_cells.columns if 'lin' in x]
df = pd.DataFrame(lineages, columns=['lineage'])
df['group'] = df['lineage'].map({lineage: i for i, lineage in enumerate(lineages)})
df['index'] = df.groupby('group').cumcount()
df['plot_row'] = df['group'] * 24

col_cutsites = [grid[i:i+19, 1:20] for i in df['plot_row']]
col_likelihood = [grid[i:i+19, 22:41] for i in df['plot_row']]
col_expression = [grid[i:i+19, 43:44] for i in df['plot_row']]
col_latent = [grid[i:i+19, 46:47] for i in df['plot_row']]
col_legend = [grid[i:i+19, 49:68] for i in df['plot_row']]

#%%
for lineage, c1, c2, c3, c4, c5 in zip(df['lineage'], col_cutsites, col_likelihood, col_expression, col_latent, col_legend):
    print(lineage, c1, c2, c3, c4, c5)

    data = lineage_objects[lineage]
    print(data['gene_name'], data['celltypes'])

    gene_name = data['gene_name']
    celltypes = data['celltypes']
    annotation_lin = {celltype: annotation[celltype] for celltype in celltypes}
    dfll = data['df_latent']
    dfll = dfll.sort_values('latent_time', ascending=False)
    dfll['celltype_numerical'] = dfll['celltype'].map({celltype: i for i, celltype in enumerate(celltypes)})

    ax_1 = plot_cutsites(fig, c1, data['cutsites'], f"{gene_name}: cut sites")
    ax_2 = plot_likelihood(fig, c2, data['probs'], f"{gene_name}: likelihood")
    ax_3 = plot_exp_heatmap(fig, c3, dfll[[gene_name]])
    ax_4 = plot_lt_heatmap(fig, c4, dfll[['celltype_numerical']], annotation_lin)
    ax_5 = plot_legend(fig, c5, annotation_lin, celltypes, ax_3, ax_2)

x1, x2, x3, x4 = 0.1, 0.4, 0.68, 0.72
y1, y2, y3 = 0.905, 0.63, 0.36

fig.text(x1, y1, 'A', fontsize=16, fontweight='bold', va='top')
fig.text(x2, y1, 'B', fontsize=16, fontweight='bold', va='top')
fig.text(x3, y1, 'C', fontsize=16, fontweight='bold', va='top')
fig.text(x4, y1, 'D', fontsize=16, fontweight='bold', va='top')

fig.text(x1, y2, 'E', fontsize=16, fontweight='bold', va='top')
fig.text(x2, y2, 'F', fontsize=16, fontweight='bold', va='top')
fig.text(x3, y2, 'G', fontsize=16, fontweight='bold', va='top')
fig.text(x4, y2, 'H', fontsize=16, fontweight='bold', va='top')

fig.text(x1, y3, 'I', fontsize=16, fontweight='bold', va='top')
fig.text(x2, y3, 'J', fontsize=16, fontweight='bold', va='top')
fig.text(x3, y3, 'K', fontsize=16, fontweight='bold', va='top')
fig.text(x4, y3, 'L', fontsize=16, fontweight='bold', va='top')

fig.savefig(folder_data_preproc / 'plots' / "fig4.pdf", bbox_inches='tight', pad_inches=0.01)
fig

#%%
import matplotlib.pyplot as plt

colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
colormaps = ['Greys', 'Purples', 'Blues', 'Reds', 'PuRd', 'BuPu']

df =  data['probs']
for cmap in colormaps:
    fig, ax = plt.subplots()
    ax.imshow(df, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_title(f'Colormap: {cmap}')
    plt.show()

# %%
