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

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
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
scores = pd.DataFrame(
    {
        "mine": celltype_expression.loc[["cDCs"]].min(),
        "dispersions_norm": transcriptome.var.dispersions_norm,
    }
)
scores["symbol"] = transcriptome.var.symbol
scores.query("mine > 1").head(10)

# %%
# gene_id = transcriptome.gene_id("GZMH")
gene_id = transcriptome.gene_id("CCL4")
# gene_id = transcriptome.gene_id("EBF1")
# gene_id = "ENSG00000136250"

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
outcome = -transcriptome.adata.obs["celltype"].cat.codes[cells_oi]
cell_order = outcome.sort_values().index

n_cells = len(cell_order)

obs = fragments.obs.copy()
obs.index = obs.index.astype(str)
obs["gex"] = outcome[cell_order]
obs = obs.loc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
window_oi = fragments.regions.window
# window = [-10000, 6000]
# window_oi = [-18000, -14000]
# window_oi = [-10000, 20000]
# window_oi = [-22000, -20000]
# window_oi = [60000, 90000]
# window_oi = [2500, 5000]

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
celltype_colors = pd.Series(sns.color_palette("tab20", n_colors=clustering.n_clusters).as_hex(), index=clustering.cluster_info.index)

# %%
# fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/10), sharey = True, width_ratios = [2, 0.5])

fig = chd.grid.Figure(chd.grid.Grid())

main = fig.main.add_under(chd.grid.Grid())

panel_fragments, ax_fragments = main.add_right(chd.grid.Panel((8, n_cells/2000)))
panel_gex, ax_gex = main.add_right(chd.grid.Panel((1, n_cells/2000)))

ax_fragments.set_xlim(*window_oi)
ax_fragments.set_ylim(0, n_cells)

c = celltype_colors[obs.loc[mapping[:, 0], "celltype"]]

# lc = mpl.collections.LineCollection(segments, linewidths=1, color = "#33333333")
lc = mpl.collections.LineCollection(segments, linewidths=1, color = c, alpha = 0.5)
ax_fragments.add_collection(lc)

ax_fragments.scatter(coordinates[:, 0], obs.loc[mapping[:, 0]]["y"], s=0.5, c = c, alpha = 0.5)
ax_fragments.scatter(coordinates[:, 1], obs.loc[mapping[:, 0]]["y"], s = 0.5, c = c, alpha = 0.5)

ax_gex.plot(obs["gex"], obs["y"])
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
ax_gex.set_ylim(0, n_cells)
# ax_gex.xaxis.set_label_position('top')
# ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
# ax_fragments.xaxis.set_label_position('top')
# ax_fragments.xaxis.tick_top()

panel_legend, ax_legend = fig.main.add_under(chd.grid.Panel((9, 1)))
ax_legend.axis("off")

for celltype in celltype_colors.index:
    ax_legend.scatter([], [], s=100, c=celltype_colors[celltype], label=celltype)
ax_legend.legend(ncols = 8)

fig.plot()

# %%
obs.loc[mapping[:, 0], "y"].max()

# %%
