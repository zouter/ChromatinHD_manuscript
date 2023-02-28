# %%
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import chromatinhd as chd

sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

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

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

#%%
lt_myeloid = pd.read_csv(folder_data_preproc / 'MV2_latent_time_myeloid.csv')
lt_myeloid.index = lt_myeloid["cell"]
del lt_myeloid["cell"]
lt_myeloid.sort_values('latent_time', inplace=True)

transcriptome = transcriptome.adata[lt_myeloid.index, :]
# fragments = fragments.adata[lt_myeloid.index, :]
#%%
transcriptome.obs.index.name = 'cell'
transcriptome.obs = transcriptome.obs.join(lt_myeloid, how='left')

# %%
gene = 'MPO'

for gene in hspc_marker_genes:
    gene_id = transcriptome.var.loc[gene]['Accession']
    try:
       gene_ix = fragments.var.loc[gene_id]["ix"]
    except KeyError:
        break

    sc.pl.umap(transcriptome, color = [gene])

    # %%
    cells_oi = range(0, 4000)

    coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix].numpy()#[cells_oi]
    mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix].numpy()#[cells_oi]
    # outcome = transcriptome.adata.obs["celltype"].cat.codes#[cells_oi]
    outcome = lt_myeloid

    cell_order = outcome.index
    n_cells = len(cell_order)

    #%%
    obs = fragments.obs.copy()
    obs = obs[obs.index.isin(cell_order)]
    obs.index = obs.index.astype(str)
    obs = obs.join(outcome, how='left')
    obs = obs.rename(columns={'latent_time': 'gex'})
    obs = obs.loc[cell_order]
    obs["y"] = np.arange(obs.shape[0])
    obs = obs.set_index("ix")

    # %%
    fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/300), sharey = True, width_ratios = [2, 0.5])
    ax_fragments.set_xlim(*window)
    ax_fragments.set_ylim(0, n_cells)

    for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
        if cell_ix in obs.index:
            color = "black"
            color = "#33333333"
            rect = mpl.patches.Rectangle((start, obs.loc[cell_ix, "y"]), end - start, 10, fc = "#33333333", ec = None, linewidth = 0)
            ax_fragments.add_patch(rect)
            
            rect = mpl.patches.Rectangle((start-10, obs.loc[cell_ix, "y"]), 10, 10, fc = "red", ec = None, linewidth = 0)
            ax_fragments.add_patch(rect)
            rect = mpl.patches.Rectangle((end-10, obs.loc[cell_ix, "y"]), 10, 10, fc = "red", ec = None, linewidth = 0)
            ax_fragments.add_patch(rect)
            
    ax_gex.plot(obs["gex"], obs["y"])
    ax_gex.set_xlabel('latent time')

    ax_fragments.set_xlabel("Distance from TSS")
    ax_fragments.set_xticks([4400])

    for ax1 in [ax_gex, ax_fragments]:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel(ax1.get_xlabel())
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(ax1.get_xticklabels())

    # %%
    fig.savefig(folder_data_preproc / f"plots/{gene}_fragments.png", transparent=False, dpi = 300)

# %%
