# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import torch
import pickle
import itertools
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
hspc_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

#%%
variables = {
    'lineages': ["myeloid", "erythroid", "platelet"],
    'genes': hspc_genes,
}

lineage_gene = {
    '_'.join(str(v) for v in dict(zip(variables.keys(), values)).values()): None
    for values in itertools.product(*variables.values())
}

for key in lineage_gene.keys():
    dataset_name, gene_name = key.split('_')

    fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}/10k10k")
    fragments.window = window
    fragments.create_cut_data()
    gene_id = adata.var.loc[gene_name]['Accession']
    gene_ix = fragments.var.index.get_loc(gene_id)

    df_latent = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv", index_col = 0)
    df_latent['rank'] = (df_latent['latent_time'].rank() - 1) / (len(df_latent) - 1)

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

    lineage_gene[key] = df_cutsites

#%%
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

for lineage in ['myeloid', 'erythroid', 'platelet']:
    lineage_objects = {key: value for key, value in lineage_gene.items() if lineage in key}

    height, width = 15, 15
    fig = plt.figure(figsize=(width, height))

    rows, cols = 4, 4
    grid = GridSpec(rows, cols, figure=fig)

    col_positions = list(range(1, rows+1)) * int((len(hspc_genes) / rows))
    row_positions = sorted(col_positions)

    grid_positions = [grid[i, j] for i in range(cols) for j in range(rows)]

    for index, (key, value) in enumerate(lineage_objects.items()):
        ax_1 = plot_cutsites(fig, grid_positions[index], value, key)

        if col_positions[index] > 1:
            ax_1.set_yticklabels([])

        if row_positions[index] < cols:
            ax_1.set_xticklabels([])

    fig.savefig(folder_data_preproc / 'plots' / f"figS1_{lineage}.pdf", bbox_inches='tight', pad_inches=0.01)

# %%