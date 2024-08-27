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
# # Visualize a gene fragments

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import torch_scatter
import torch

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "liver"
# dataset_name = "pbmc10k"
# dataset_name = "liver"
# dataset_name = "hspc"
# dataset_name = "pbmc10k/subsets/mono_t_a"
# dataset_name = "hspc_gmp"
# dataset_name = "e18brain"
folder_dataset = chd.get_output() / "datasets" / dataset_name

# %%
# promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_name, window = "100k100k", np.array([-100000, 100000])

# %%
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")
fragments = chd.data.Fragments(folder_dataset / "fragments" / promoter_name)
clustering = chd.data.Clustering(folder_dataset / "latent" / "leiden_0.1")

# %%
celltype_expression = {}
for celltype in transcriptome.adata.obs["celltype"].unique():
    celltype_expression[celltype] = np.array(transcriptome.X[:])[
        transcriptome.adata.obs["celltype"] == celltype
    ].mean(0)
celltype_expression = pd.DataFrame(celltype_expression).T

# %%
celltype_expression.columns = transcriptome.var.index

# %%
# scores = pd.DataFrame(
#     {
#         "mine": celltype_expression.loc[["KC"]].min(),
#         "dispersions_norm": transcriptome.var.dispersions_norm,
#     }
# )
# scores["symbol"] = transcriptome.var.symbol
# scores.query("mine > 1").head(10)

# %%
gene_id = transcriptome.gene_id("Gdf2")
# gene_id = transcriptome.gene_id("IRF1")
# gene_id = transcriptome.gene_id("KLF7")
# gene_id = transcriptome.gene_id("Clec4f")
# gene_id = transcriptome.gene_id("Glul")
# gene_id = transcriptome.gene_id("EBF1")
# gene_id = "ENSMUSG00000026814"

# %%
region = fragments.regions.coordinates.loc[gene_id]

# %%
adata2 = transcriptome.adata.copy()

# %%
# sc.external.pp.magic(adata2)

# %%
sc.pl.umap(transcriptome.adata, color=[gene_id], use_raw = False)

# %%
gene_ix = fragments.var.loc[gene_id]["ix"]

# %%
# cells_oi = range(0, 1000)
cells_oi = range(0, fragments.n_cells)

# %%
coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix]
mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix]
coordinates = coordinates[np.isin(mapping[:, 0], cells_oi)]
mapping = mapping[np.isin(mapping[:, 0], cells_oi)]

# %%
expression = sc.get.obs_df(transcriptome.adata, gene_id)[cells_oi]
expression = sc.get.obs_df(transcriptome.adata, gene_id, layer="magic")[cells_oi]
outcome = expression
# outcome = transcriptome.adata.obs["oi"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["overexpressed"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["leiden"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["gene_overexpressed"].cat.codes[cells_oi]
# outcome = -clustering.labels.cat.codes[cells_oi]
# outcome = -transcriptome.adata.obs["celltype"].cat.codes[cells_oi]
cell_order = outcome.sort_values().index

n_cells = len(cell_order)

obs = transcriptome.obs.copy()
obs["ix"] = np.arange(obs.shape[0])
obs.index = obs.index.astype(str)
obs["gex"] = outcome[cell_order]
obs = obs.loc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
window_oi = fragments.regions.window
resolution = 2000

# window_oi = [33250, 33400]
# resolution = 100
# window_oi = [33000, 33150]
# resolution = 100
# window_oi = [52000, 54000]
# resolution = 250
# window_oi = [32750, 34000]
# resolution = 500
# window_oi = [30000, 35000]
# resolution = 2000

# %%
32750 + (33500-32750) / 2

# %%
segments = np.stack([
    np.stack([
        coordinates[:, 0],
        coordinates[:, 1],
    ]),
    np.stack([
        obs.loc[mapping[:, 0]]["y"],
        obs.loc[mapping[:, 0]]["y"],
    ]),
]).transpose(2, 1, 0)

# %%
celltype_colors = pd.Series(sns.color_palette("husl", n_colors=clustering.n_clusters).as_hex(), index=clustering.cluster_info.index)

# %%
width = (window_oi[1] - window_oi[0]) // resolution

# %%
# fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/10), sharey = True, width_ratios = [2, 0.5])

fig = polyptich.grid.Figure(polyptich.grid.Grid())

main = fig.main.add_under(polyptich.grid.Grid())

panel_fragments, ax_fragments = main.add_right(polyptich.grid.Panel((width, n_cells/1000)))
panel_gex, ax_gex = main.add_right(polyptich.grid.Panel((1, n_cells/1000)))

ax_fragments.set_xlim(*window_oi)
ax_fragments.set_ylim(0, n_cells)

c = celltype_colors[obs.loc[mapping[:, 0], "celltype"]]

lc = mpl.collections.LineCollection(segments, linewidths=1, color = c, alpha = 0.1)
ax_fragments.add_collection(lc)

ax_fragments.scatter(coordinates[:, 0], obs.loc[mapping[:, 0]]["y"], s= 0.5, c = c, alpha = 1.0)
ax_fragments.scatter(coordinates[:, 1], obs.loc[mapping[:, 0]]["y"], s = 0.5, c = c, alpha = 1.0)

ax_gex.plot(obs["gex"], obs["y"])
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
ax_gex.set_ylim(0, n_cells)
ax_gex.xaxis.set_label_position('top')
ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
ax_fragments.xaxis.set_label_position('top')
ax_fragments.xaxis.tick_top()

panel_legend, ax_legend = fig.main.add_under(polyptich.grid.Panel((9, 1)))
ax_legend.axis("off")

for celltype in celltype_colors.index:
    ax_legend.scatter([], [], s=100, c=celltype_colors[celltype], label=celltype)
ax_legend.legend(ncols = 8)

fig.plot()

# %%
var = transcriptome.adata.raw.to_adata().var
var.query("symbol == 'NR1H3'")

# %%
sc.pl.umap(transcriptome.adata, color = ["ENSG00000025434"])

# %%
obs.loc[mapping[:, 0], "y"].max()

# %% [markdown]
# ## Nice

# %%
regions = pd.DataFrame({
    "start":[32950, 32750, 30000],
    "end":[33200, 34000, 35000],
    "resolution":[250, 500, 1000]
})

# %%
# window_oi1 = [33250, 33400]
window_oi1 = [32950, 33200]
# window_oi1 = [33000, 33150]
resolution1 = 200
width1 = (window_oi1[1] - window_oi1[0]) // resolution1

# window_oi2 = [32750, 33500]
window_oi2 = [32750, 34000]
resolution2 = 750
width2 = (window_oi2[1] - window_oi2[0]) // resolution2

window_oi3 = [30000, 35000]
resolution3 = 2000
width3 = (window_oi3[1] - window_oi3[0]) // resolution3

# %%
celltype_info = pd.DataFrame(
    {"celltype": transcriptome.adata.obs["celltype"].cat.categories}
).set_index("celltype")
celltype_info["color"] = sns.color_palette("husl", n_colors=len(celltype_info))

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height = 0.05))

main = fig.main.add_under(polyptich.grid.Grid(padding_height = 0.15))

height = n_cells / 2000

# celltype
panel_celltypelabel, ax_celltypelabel = main.add_right(
    polyptich.grid.Panel((1, height)), padding=0.0, row=0
)
ax_celltypelabel.set_yticks([])
ax_celltypelabel.set_xticks([])
ax_celltypelabel.set_ylim(ax_fragments.get_ylim())
plotdata = obs.groupby("celltype").median().sort_values("y")
ax_celltypelabel.set_xlim(-1, 0)

texts = []
for i, (celltype, row) in enumerate(plotdata.iterrows()):
    texts.append(
        ax_celltypelabel.annotate(
            celltype,
            (0.0, row["y"]),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
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

panel, ax_celltype = main.add_right(
    polyptich.grid.Panel((0.2, height)), padding=0.0, row=0
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

# expression
panel_gex, ax_gex = main.add_right(polyptich.grid.Panel((1, height)), padding=0.25)
ax_gex.plot(obs["gex"], obs["y"], color = "#333")
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
ax_gex.set_ylim(0, n_cells)
ax_gex.xaxis.set_label_position("top")
ax_gex.xaxis.tick_top()
ax_gex.set_yticks([])
ax_gex.set_xlim(0)

# fragments
prev_windows = []
for i, width, window_oi, resolution in [
    [0, width1, window_oi1, resolution1],
    [1, width2, window_oi2, resolution2],
    [2, width3, window_oi3, resolution3],
]:
    panel_fragments, ax_fragments = main.add_right(polyptich.grid.Panel((width, height)), padding=0.15)

    coordinates_oi = coordinates[
        (coordinates[:, 0] >= window_oi[0])
        & (coordinates[:, 1] <= window_oi[1])
    ]
    mapping_oi = mapping[
        (coordinates[:, 0] >= window_oi[0])
        & (coordinates[:, 1] <= window_oi[1])
    ]
    segments = np.stack([
        np.stack([
            coordinates_oi[:, 0],
            coordinates_oi[:, 1],
        ]),
        np.stack([
            obs.loc[mapping_oi[:, 0]]["y"],
            obs.loc[mapping_oi[:, 0]]["y"],
        ]),
    ]).transpose(2, 1, 0)

    ax_fragments.set_xlim(*window_oi)
    ax_fragments.set_ylim(0, n_cells)

    c = celltype_info.loc[obs.loc[mapping_oi[:, 0], "celltype"], "color"]

    lc = mpl.collections.LineCollection(segments, linewidths=1, color=c, alpha=0.1)
    ax_fragments.add_collection(lc)

    ax_fragments.scatter(coordinates_oi[:, 0], obs.loc[mapping_oi[:, 0]]["y"], s=0.5, c=c, alpha=1.0)
    ax_fragments.scatter(coordinates_oi[:, 1], obs.loc[mapping_oi[:, 0]]["y"], s=0.5, c=c, alpha=1.0)
    ax_fragments.set_xlabel("Distance from TSS")
    ax_fragments.xaxis.set_label_position("top")
    ax_fragments.xaxis.tick_top()
    ax_fragments.set_yticks([])
    ax_fragments.set_xticks(window_oi)
    ax_fragments.get_xticklabels()[0].set_horizontalalignment("left")
    ax_fragments.get_xticklabels()[1].set_horizontalalignment("right")
    ax_fragments.set_xlabel("")

    for prev_window in prev_windows:
        ax_fragments.axvspan(prev_window[0], prev_window[1], color="black", alpha=0.1)

    # peaks
    peaks_folder = chd.get_output() / "peaks" / dataset_name
    peaks_panel = main[1, i+3] = chdm.plotting.Peaks(
            region,
            peaks_folder,
            window=window_oi,
            width=width,
            row_height=0.6,
            label_methods = i == 0,
            label_rows = i == 0,
            label_methods_side="left",
        )


    for prev_window in prev_windows:
        peaks_panel.ax.axvspan(prev_window[0], prev_window[1], color="black", alpha=0.1)

    prev_windows.append(window_oi)

# panel_legend, ax_legend = fig.main.add_under(polyptich.grid.Panel((9, 1)))
# ax_legend.axis("off")
# for celltype in celltype_colors.index:
#     ax_legend.scatter([], [], s=100, c=celltype_colors[celltype], label=celltype)
# ax_legend.legend(ncols=8)

fig.plot()

symbol = transcriptome.symbol(gene_id)
manuscript.save_figure(fig, "1", f"example_{symbol}")

# %% [markdown]
# ## Large

# %%
expression = sc.get.obs_df(transcriptome.adata, gene_id)

# %%
cells_oi = np.concatenate([expression.index[expression < 1.3], np.random.choice(expression.index[expression > 1.3], 500, replace = False)])
cells_oi = np.unique(cells_oi)
cells_oi = fragments.obs.index.get_indexer(cells_oi)

# %%
coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix]
mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix]
coordinates = coordinates[np.isin(mapping[:, 0], cells_oi)]
mapping = mapping[np.isin(mapping[:, 0], cells_oi)]

# %%
expression = sc.get.obs_df(transcriptome.adata, gene_id)[cells_oi]
expression = sc.get.obs_df(transcriptome.adata, gene_id, layer="magic")[cells_oi]
outcome = expression
# outcome = transcriptome.adata.obs["oi"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["overexpressed"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["leiden"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["gene_overexpressed"].cat.codes[cells_oi]
# outcome = -clustering.labels.cat.codes[cells_oi]
# outcome = -transcriptome.adata.obs["celltype"].cat.codes[cells_oi]
cell_order = outcome.sort_values().index

n_cells = len(cell_order)

obs = transcriptome.obs.copy()
obs["ix"] = np.arange(obs.shape[0])
obs.index = obs.index.astype(str)
obs["gex"] = outcome[cell_order]
obs = obs.loc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
window_oi = [-20000, 20000]
resolution = 2500

# %%
segments = np.stack([
    np.stack([
        coordinates[:, 0],
        coordinates[:, 1],
    ]),
    np.stack([
        obs.loc[mapping[:, 0]]["y"],
        obs.loc[mapping[:, 0]]["y"],
    ]),
]).transpose(2, 1, 0)

# %%
celltype_colors = pd.Series(sns.color_palette("husl", n_colors=clustering.n_clusters).as_hex(), index=clustering.cluster_info.index)

# %%
width = (window_oi[1] - window_oi[0]) // resolution

# %%
# fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/10), sharey = True, width_ratios = [2, 0.5])

fig = polyptich.grid.Figure(polyptich.grid.Grid())

main = fig.main.add_under(polyptich.grid.Grid())

panel_fragments, ax_fragments = main.add_right(polyptich.grid.Panel((width, n_cells/350)))
panel_gex, ax_gex = main.add_right(polyptich.grid.Panel((1, n_cells/350)))

ax_fragments.set_xlim(*window_oi)
ax_fragments.set_ylim(0, n_cells)

lc = mpl.collections.LineCollection(segments, linewidths=1, color = "#333", alpha = 0.1)
ax_fragments.add_collection(lc)

ax_fragments.scatter(coordinates[:, 0], obs.loc[mapping[:, 0]]["y"], s= 0.5, c = "#555", alpha = 1.0)
ax_fragments.scatter(coordinates[:, 1], obs.loc[mapping[:, 0]]["y"], s = 0.5, c = "#555", alpha = 1.0)

ax_gex.plot(obs["gex"], obs["y"], color = "#333")
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
ax_gex.set_ylim(0, n_cells)
ax_gex.xaxis.set_label_position('top')
ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
ax_fragments.xaxis.set_label_position('top')
ax_fragments.xaxis.tick_top()

ax_fragments.set_yticks([])
ax_gex.set_yticks([])

fig.plot()

# %%
