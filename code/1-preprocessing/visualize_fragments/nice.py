# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize the fragments around a gene


# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    get_ipython().run_line_magic("config", "InlineBackend.figure_format='retina'")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import scanpy as sc

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_clustered"
# dataset_name = "e18brain"
# dataset_name = "lymphoma+pbmc10k"
# dataset_name = "MSGN1_7"
# dataset_name = "CDX1_7"
# dataset_name = "morf_20"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_name, window = "100k100k", np.array([-100000, 100000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
settings = {
    "local_QKI_-1300": {
        "name": "local_QKI_-1300",
        "gene": "QKI",
        "window": [-1250, -1200],
        "cells": 1000,
    },
    "mid_IL1B_-5kb": {
        "name": "mid_IL1B_-5kb",
        "gene": "IL1B",
        "window": [-5500, -4500],
        "cells": 1000,
    },
    "long_AAK1_TSS": {
        "name": "long_AAK1_TSS",
        "gene": "AAK1",
        "window": [-4000, -0],
        "cells": 1000,
    },
    "long_ZEB1_TSS": {
        "name": "long_ZEB1_TSS",
        "gene": "ZEB1",
        "window": [-1500, 3500],
        "cells": 1000,
    },
}
setting = settings["local_QKI_-1300"]
setting = settings["mid_IL1B_-5kb"]
setting = settings["long_ZEB1_TSS"]
setting = None

# %%
# gene_id = transcriptome.gene_id("IL1B")
# gene_id = transcriptome.gene_id("NKG7")
# gene_id = transcriptome.gene_id("QKI")
# gene_id = transcriptome.gene_id("FOSB")
# gene_id = transcriptome.gene_id("CCL4")
# gene_id = transcriptome.gene_id("CCDC138")
# gene_id = transcriptome.gene_id("UBE3D")
# gene_id = transcriptome.gene_id("ZEB1")
gene_id = transcriptome.gene_id("RHEX")

if setting is not None:
    gene_id = transcriptome.gene_id(setting["gene"])

# %%
gene_ix = fragments.var.loc[gene_id]["ix"]

# %%
# cells_oi = range(0, 4000)
cells_oi = range(0, fragments.n_cells)
# cells_oi = np.random.choice(fragments.n_cells, 1000, replace=False)

# %%
coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix].numpy()
mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix].numpy()
coordinates = coordinates[np.isin(mapping[:, 0], cells_oi)]
mapping = mapping[np.isin(mapping[:, 0], cells_oi)]

# %%
expression = sc.get.obs_df(transcriptome.adata, gene_id)[cells_oi]
expression = sc.get.obs_df(transcriptome.adata, gene_id, layer="magic")[cells_oi]
expression[expression < (expression.max() / 10)] = 0.0
outcome = expression
# outcome = transcriptome.adata.obs["oi"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["overexpressed"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["leiden"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["gene_overexpressed"].cat.codes[cells_oi]
outcome2 = transcriptome.adata.obs["celltype"].cat.codes[cells_oi]
cell_order = (outcome + outcome2 / 10000).sort_values().index

n_cells = len(cell_order)

obs = transcriptome.obs.copy()
obs["ix"] = np.arange(obs.shape[0])
obs.index = obs.index.astype(str)
obs["gex"] = outcome[cell_order]
obs = obs.loc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
celltype_info = pd.DataFrame(
    {"celltype": transcriptome.adata.obs["celltype"].cat.categories}
).set_index("celltype")
celltype_info["color"] = sns.color_palette("tab20", n_colors=len(celltype_info))

# %%
window_oi = window
# window_oi = [-2975, -2775]
# window_oi = [-18000, -14000]
# window_oi = np.array([-10000, 10000])
# window_oi = [-10000, 10000]
window_oi = [-10000, 10000]
# window_oi = [-4000, 0]
# window_oi = [-5500, -4500]
# window_oi = [-2500, -1000]
# window_oi = [-2500, -1000]
# window_oi = [-2500, -1000]
# window_oi = [-1250, -1200]
# window_oi = [-500, 0]
# window_oi = [-100000, 100000]
# window_oi = [60000, 90000]
if setting is not None:
    window_oi = setting["window"]

promoter = promoters.loc[gene_id]
promoter_oi = promoter.copy()
if promoter["strand"] == 1:
    promoter_oi["start"] = promoter_oi["tss"] + window_oi[0]
    promoter_oi["end"] = promoter_oi["tss"] + window_oi[1]
else:
    promoter_oi["end"] = promoter_oi["tss"] - window_oi[0]
    promoter_oi["start"] = promoter_oi["tss"] - window_oi[1]

# %%
fig = chd.grid.Figure(chd.grid.Grid())

# panel_width = 2.
panel_width = 2.0

# genes

genome_folder = folder_data_preproc / "genome"
panel_genes = chd.plot.genome.Genes.from_region(
    promoter_oi,
    width=panel_width,
    label_genome=True,
    symbol=transcriptome.symbol(gene_id),
)
panel_genes = fig.main.add_under(panel_genes)

panel_height = n_cells / 2500

# fragments
panel_fragments, ax_fragments = fig.main.add_under(
    chdm.plotting.fragments.Fragments(
        coordinates,
        mapping,
        window=window_oi,
        height=panel_height,
        width=panel_width,
        obs=obs,
        connect=(setting is None) or (setting["name"] != "local_QKI_-1300"),
    ),
    padding=0.1,
)

# expression
panel, ax_gex = fig.main.add_right(
    chd.grid.Panel((0.5, panel_height)), padding=0.1, row=panel_fragments
)
ax_gex.set_ylim(0, obs["y"].max())
ax_gex.set_yticks([])
ax_gex.plot(obs["gex"], obs["y"], color="#333333")
ax_gex.set_xlim(0)

ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")


# celltype
panel, ax_celltype = fig.main.add_right(
    chd.grid.Panel((0.2, panel_height)), padding=0.1, row=panel_fragments
)
patches = []
for _, obs_row in obs.iterrows():
    patches.append(
        mpl.patches.Rectangle(
            (0, obs_row["y"] - 0.5),
            1,
            1,
            fc=celltype_info.loc[obs_row["celltype"], "color"],
            ec="None",
            lw=0,
        )
    )
ax_celltype.add_collection(
    mpl.collections.PatchCollection(patches, match_original=True)
)
ax_celltype.set_yticks([])
ax_celltype.set_xticks([])
ax_celltype.set_ylim(ax_fragments.get_ylim())

panel_celltypelabel, ax_celltypelabel = fig.main.add_right(
    chd.grid.Panel((5, panel_height)), padding=0.0, row=panel_fragments
)
ax_celltypelabel.set_yticks([])
ax_celltypelabel.set_xticks([])
ax_celltypelabel.set_ylim(ax_fragments.get_ylim())
plotdata = obs.groupby("celltype").median().sort_values("y")

texts = []
for i, (celltype, row) in enumerate(plotdata.iterrows()):
    texts.append(
        ax_celltypelabel.annotate(
            celltype,
            (0.0, row["y"]),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=celltype_info.loc[celltype, "color"],
            fontsize=8,
            arrowprops=dict(
                arrowstyle="-",
                color=celltype_info.loc[celltype, "color"],
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0",
            ),
        )
    )
ax_celltypelabel.axis("off")
peaks_folder = chd.get_output() / "peaks" / dataset_name
peaks_panel = fig.main.add_under(
    chdm.plotting.Peaks(
        promoter_oi,
        peaks_folder,
        window=window_oi,
        width=panel_width,
        row_height=0.8,
        label_methods_side="left",
    )
)

fig.plot()

adjust = True
if adjust:
    import adjustText

    adjustText.adjust_text(
        texts,
        ax=ax_celltypelabel,
        ha="left",
        autoalign=False,
        only_move={"text": "y"},
        avoid_self=False,
        ensure_inside_axes=False,
    )

# %%
if setting is not None:
    manuscript.save_figure(fig, "1", "fragments_" + setting["name"], dpi=300)

# %%
fig_colorbar = plt.figure(figsize=(3.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chdm.plotting.fragments.length_norm, cmap=chdm.plotting.fragments.length_cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal")
colorbar.set_label("Fragment length")
manuscript.save_figure(fig_colorbar, "1", "colorbar_length")

# %% [markdown]
# ## Find interesting regions

# %%
prediction_name = "v20_initdefault"
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / "permutations_5fold5repeat"
    / prediction_name
)

# %%
gene = transcriptome.gene_id("IL1B")

# %%
sc.pl.umap(transcriptome.adata, color=[gene])

# %%
scores_folder_window = prediction.path / "scoring" / "window_gene" / gene
window_scoring = chd.scoring.prediction.Scoring.load(scores_folder_window)

# %%
scores = (
    window_scoring.genescores.sel(gene=gene)
    .mean("model")
    .sel(phase=["validation", "test"])
    .mean("phase")
    .to_pandas()
)

# %%
scores.sort_values("deltacor")

# %%
gene = transcriptome.gene_id("IL1B")

# %%
sc.pl.umap(transcriptome.adata, color=[gene_id])

# %%
yup = []
for gene in tqdm.tqdm(transcriptome.var.index):
    scores_folder_window = prediction.path / "scoring" / "window_gene" / gene
    window_scoring = chd.scoring.prediction.Scoring.load(scores_folder_window)
    scores = (
        window_scoring.genescores.sel(gene=gene)
        .mean("model")
        .sel(phase=["validation", "test"])
        .mean("phase")
        .to_pandas()
    )
    scores = scores.loc[(scores.index > -10000) & (scores.index < 0)]

    scores["oi"] = scores["deltacor"] < -0.01
    if scores["oi"].sum() > 1:
        true_indices = np.where(scores["oi"])[0]
        diffs = np.diff(true_indices)
        max_diff_index = np.argmax(diffs)
        length = diffs[max_diff_index]
        start = true_indices[max_diff_index]
        end = true_indices[max_diff_index + 1] - 1

        yup.append(
            {
                "gene": transcriptome.symbol(gene),
                "length": length,
                "start": start,
                "end": end,
            }
        )

# %%
yup = pd.DataFrame(yup).set_index("gene")

# %%
yup.sort_values("length", ascending=False).query("length == 30")

# %%
