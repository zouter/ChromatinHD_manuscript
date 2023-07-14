# %%
import numpy as np
import pandas as pd
import chromatinhd as chd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"
dataset_name = 'myeloid'

#%%
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()
genes = sorted(hspc_marker_genes)

# %%
lt_myeloid = pd.read_csv(folder_data_preproc / f'{dataset_name_sub}_latent_time_myeloid.csv')
lt_myeloid.sort_values('latent_time', inplace=True)
lt_myeloid.index = lt_myeloid["cell"]
del lt_myeloid["cell"]
del lt_myeloid["celltype"]

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}" / promoter_name)
coordinates = fragments.coordinates
mapping = fragments.mapping

#%%
transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
transcriptome = transcriptome.adata[lt_myeloid.index, :]
transcriptome.obs.index.name = 'cell'
transcriptome.obs = transcriptome.obs.join(lt_myeloid, how='left')

# %%
heatmap_data = transcriptome[:, genes]
heatmap_data = heatmap_data.X

# %%
for gene in genes:
    gene_id = transcriptome.var.loc[gene]['Accession']
    gene_ix = fragments.var.loc[gene_id]["ix"]
    print(gene, gene_id, gene_ix)
    
    coordinates_sub = coordinates[mapping[:, 1] == gene_ix]
    mapping_sub = mapping[mapping[:, 1] == gene_ix]

    obs = fragments.obs.copy()
    obs = obs.join(lt_myeloid, how='left')
    obs = obs.loc[lt_myeloid.index]
    obs = obs.set_index("ix")
    obs["y"] = np.arange(obs.shape[0])

    fig, (ax_gex, ax_heatmap, ax_fragments) = plt.subplots(1, 3, figsize = (30, len(obs)/300), sharey = True, width_ratios = [0.5, 1, 1.5])

    ax_gex.plot(obs["latent_time"], obs["y"])
    ax_gex.set_xlabel('Latent Time', fontsize=20)
    ax_gex.set_xticks([0, 0.25, 0.5, 0.75, 1])

    im = ax_heatmap.imshow(heatmap_data, cmap='coolwarm', aspect='auto', origin='lower', extent=[0, heatmap_data.shape[1], 0, heatmap_data.shape[0]], interpolation='none')

    ax_heatmap.set_xlabel('Gene Expression', fontsize=20)
    ax_heatmap.set_xticks(np.arange(heatmap_data.shape[1]) + 0.5)
    ax_heatmap.set_xticklabels(genes)
    ax_heatmap.tick_params(axis='x', rotation=270)

    highlighted_column_index = genes.index(gene)
    rect = Rectangle((highlighted_column_index, 0), 1, heatmap_data.shape[0], edgecolor='black', facecolor='none', linewidth=3)
    ax_heatmap.add_patch(rect)

    highlighted_label = ax_heatmap.xaxis.get_ticklabels()[highlighted_column_index]
    highlighted_label.set_weight('bold')
    highlighted_label.set_backgroundcolor('lightgray')

    ax_fragments.set_xlim(*window)
    ax_fragments.set_ylim(0, len(obs))

    for (start, end, cell_ix) in zip(coordinates_sub[:, 0], coordinates_sub[:, 1], mapping_sub[:, 0]):
        cell_ix = int(cell_ix)
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

    fig.show()
    # fig.savefig(folder_data_preproc / f"plots/fragments_{gene}.png", transparent=False, dpi = 300)

# %%
