# %%
import scanpy as sc
import numpy as np
import pandas as pd
import chromatinhd as chd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sns.set_style('ticks')

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

#%%
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
s_genes = info_genes_cells['s_genes'].dropna().tolist()
g2m_genes = info_genes_cells['g2m_genes'].dropna().tolist()
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()
lin_myeloid = info_genes_cells['lin_myeloid'].dropna().tolist()
lin_erythroid = info_genes_cells['lin_erythroid'].dropna().tolist()
lin_platelet = info_genes_cells['lin_platelet'].dropna().tolist()

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])

#%%
lt_myeloid = pd.read_csv(folder_data_preproc / 'MV2_latent_time_myeloid.csv')
lt_myeloid.index = lt_myeloid["cell"]
del lt_myeloid["cell"]
lt_myeloid.sort_values('latent_time', inplace=True)

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

#%%
transcriptome = transcriptome.adata[lt_myeloid.index, :]
#%%
transcriptome.obs.index.name = 'cell'
transcriptome.obs = transcriptome.obs.join(lt_myeloid, how='left')

genes = sorted(hspc_marker_genes)

heatmap_data = transcriptome[:, genes]
heatmap_data = heatmap_data.X

# %%
for gene in genes:
    print('try', gene)

    gene_id = transcriptome.var.loc[gene]['Accession']
    try:
        gene_ix = fragments.var.loc[gene_id]["ix"]
        print('success', gene)

    except KeyError:
        print('failed', gene)
        continue

    # sc.pl.umap(transcriptome, color = [gene])

    coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix].numpy()
    mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix].numpy()
    outcome = lt_myeloid

    cell_order = outcome.index
    n_cells = len(cell_order)

    obs = fragments.obs.copy()
    obs = obs[obs.index.isin(cell_order)]
    obs.index = obs.index.astype(str)
    obs = obs.join(outcome, how='left')
    obs = obs.loc[cell_order]
    obs = obs.set_index("ix")
    obs["y"] = np.arange(obs.shape[0])

    fig, (ax_gex, ax_heatmap, ax_fragments) = plt.subplots(1, 3, figsize = (30, n_cells/300), sharey = True, width_ratios = [0.5, 1, 1.5])

    ax_gex.plot(obs["latent_time"], obs["y"])
    ax_gex.set_xlabel('Latent Time', fontsize=20)
    ax_gex.set_xticks([0, 0.25, 0.5, 0.75, 1])

    im = ax_heatmap.imshow(heatmap_data, cmap='coolwarm', aspect='auto', origin='lower', extent=[0, heatmap_data.shape[1], 0, heatmap_data.shape[0]], interpolation='none')

    ax_heatmap.set_xlabel('Gene Expression', fontsize=20)
    ax_heatmap.set_xticks(np.arange(heatmap_data.shape[1]) + 0.5)
    ax_heatmap.set_xticklabels(genes)
    ax_heatmap.tick_params(axis='x', rotation=270)

    highlighted_column = gene
    highlighted_column_index = genes.index(highlighted_column)
    rect = Rectangle((highlighted_column_index, 0), 1, heatmap_data.shape[0], edgecolor='black', facecolor='none', linewidth=3)
    ax_heatmap.add_patch(rect)

    highlighted_label = ax_heatmap.xaxis.get_ticklabels()[highlighted_column_index]
    highlighted_label.set_weight('bold')
    highlighted_label.set_backgroundcolor('lightgray')

    # cbar = ax_heatmap.figure.colorbar(im, ax=ax_heatmap)

    ax_fragments.set_xlim(*window)
    ax_fragments.set_ylim(0, n_cells)

    for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
        if cell_ix in obs.index:
            rectangles = [
                mpl.patches.Rectangle((start, obs.loc[cell_ix, "y"]), end - start, 10, fc = "#33333333", ec = None, linewidth = 0),
                mpl.patches.Rectangle((start-10, obs.loc[cell_ix, "y"]), 10, 10, fc = "red", ec = None, linewidth = 0),
                mpl.patches.Rectangle((end-10, obs.loc[cell_ix, "y"]), 10, 10, fc = "red", ec = None, linewidth = 0)
            ]
            for rect in rectangles:
                ax_fragments.add_patch(rect)

    ax_fragments.set_xlabel(f"Distance from TSS ({gene})", fontsize=20)
    ax_fragments.set_xticks([-10000, -5000, 0, 5000, 10000])
    
    for ax1 in [ax_gex, ax_heatmap, ax_fragments]:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel(ax1.get_xlabel())
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(ax1.get_xticklabels())

        if ax1 == ax_heatmap:
            ax2.tick_params(axis='x', rotation=90)
            highlighted_label = ax2.xaxis.get_ticklabels()[highlighted_column_index]
            highlighted_label.set_weight('bold')
            highlighted_label.set_backgroundcolor('lightgray')

    fig.savefig(folder_data_preproc / f"plots/fragments_{gene}.png", transparent=False, dpi = 300)

# %%
