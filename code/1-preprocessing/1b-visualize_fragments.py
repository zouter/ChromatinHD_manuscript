# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %% [markdown]
# Interesting examples:
# - *ITK* and *HLA-B* in the lymphoma dataset: one of the many non-monotonic
# - *IL1B* in pmbc10k: a clear "mononucleotide" peak

# %%
# transcriptome.var.query("means > 1").sort_values("dispersions_norm", ascending = False).head(20).index

# %%
celltype_expression = {}
for celltype in transcriptome.adata.obs["celltype"].unique():
    celltype_expression[celltype] = np.array(transcriptome.X.to_scipy_csr().todense())[
        transcriptome.adata.obs["celltype"] == celltype
    ].mean(0)
celltype_expression = pd.DataFrame(celltype_expression).T

# %%
celltype_expression.columns = transcriptome.var.index

# %%
scores = pd.DataFrame(
    {
        "mine": celltype_expression.loc[["memory B", "cDCs"]].min(),
        "dispersions_norm": transcriptome.var.dispersions_norm,
    }
)
scores["symbol"] = transcriptome.var.symbol
scores.query("mine > 1").head(10)

# %%
# gene_id = transcriptome.gene_id("Satb2")
# gene_id = transcriptome.gene_id("PAX5")
# gene_id = transcriptome.gene_id("CCL4")
# gene_id = transcriptome.gene_id("PTGDS")
# gene_id = transcriptome.gene_id("IL1B")
# gene_id = transcriptome.gene_id("ITK")
# gene_id = transcriptome.gene_id("PLEKHD1")
# gene_id = transcriptome.gene_id("PTTG1")
# gene_id = transcriptome.gene_id("AAK1")
# gene_id = transcriptome.gene_id("BCL11B")
# gene_id = transcriptome.gene_id("NFKBIA")
# gene_id = transcriptome.gene_id("QKI")
# gene_id = transcriptome.gene_id("BACH2")
# gene_id = transcriptome.gene_id("HOXD4")
# gene_id = transcriptome.gene_id("JCHAIN")
# gene_id = transcriptome.gene_id("CD74")
# gene_id = transcriptome.gene_id("IPP")
# gene_id = transcriptome.gene_id("LYN")
# gene_id = transcriptome.gene_id("BCL2")
gene_id = transcriptome.gene_id("CD74")
# gene_id = transcriptome.gene_id("RPS14")
# gene_id = transcriptome.gene_id("CCL4")

# %%
adata2 = transcriptome.adata.copy()

# %%
# sc.external.pp.magic(adata2)

# %%
sc.pl.umap(transcriptome.adata, color=[gene_id])

# %%
gene_ix = fragments.var.loc[gene_id]["ix"]

# %%
# cells_oi = range(0, 4000)
cells_oi = range(0, fragments.n_cells)

# %%
coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix].numpy()
mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix].numpy()
coordinates = coordinates[np.isin(mapping[:, 0], cells_oi)]
mapping = mapping[np.isin(mapping[:, 0], cells_oi)]

# %%
expression = sc.get.obs_df(transcriptome.adata, gene_id)[cells_oi]
expression = sc.get.obs_df(transcriptome.adata, gene_id, layer="magic")[cells_oi]
# outcome = expression
# outcome = transcriptome.adata.obs["oi"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["overexpressed"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["leiden"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["gene_overexpressed"].cat.codes[cells_oi]
outcome = transcriptome.adata.obs["celltype"].cat.codes[cells_oi]
cell_order = outcome.sort_values().index

n_cells = len(cell_order)

obs = fragments.obs.copy()
obs.index = obs.index.astype(str)
obs["gex"] = outcome[cell_order]
obs = obs.loc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
window_oi = window
# window_oi = [0, 10000]
# window_oi = [-18000, -14000]
# window_oi = [-10000, 10000]
window_oi = [-100000, 100000]
# window_oi = [60000, 90000]

# %%
fig, (ax_fragments, ax_gex) = plt.subplots(
    1, 2, figsize=(15, n_cells / 300), sharey=True, width_ratios=[2, 0.5]
)
ax_fragments.set_xlim(*window_oi)
ax_fragments.set_ylim(0, n_cells)

for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
    if start > window_oi[1] or end < window_oi[0]:
        continue
    color = "black"
    color = "#33333333"
    rect = mpl.patches.Rectangle(
        (start, obs.loc[cell_ix, "y"]),
        end - start,
        10,
        fc="#33333333",
        ec=None,
        linewidth=0,
    )
    ax_fragments.add_patch(rect)

    rect = mpl.patches.Rectangle(
        (start - 10, obs.loc[cell_ix, "y"]), 10, 10, fc="red", ec=None, linewidth=0
    )
    ax_fragments.add_patch(rect)
    rect = mpl.patches.Rectangle(
        (end - 10, obs.loc[cell_ix, "y"]), 10, 10, fc="red", ec=None, linewidth=0
    )
    ax_fragments.add_patch(rect)

ax_gex.plot(obs["gex"], obs["y"])
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
# ax_gex.xaxis.set_label_position('top')
# ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
# ax_fragments.xaxis.set_label_position('top')
# ax_fragments.xaxis.tick_top()

# just copy the xaxis so it is repeated on top
for ax1 in [ax_gex, ax_fragments]:
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(ax1.get_xticklabels())

# %%
fig.savefig("fragments.png", transparent=False, dpi=300)

# %%
# fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/10), sharey = True, width_ratios = [2, 0.5])
fig, (ax_fragments, ax_gex) = plt.subplots(
    1, 2, figsize=(15, n_cells / 300), sharey=True, width_ratios=[2, 0.5]
)
ax_fragments.set_xlim(*window)
ax_fragments.set_ylim(0, n_cells)

# for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
ax_fragments.scatter(coordinates[:, 0], obs.loc[mapping[:, 0]]["y"])
ax_fragments.scatter(coordinates[:, 1], obs.loc[mapping[:, 0]]["y"])

ax_gex.plot(obs["gex"], obs["y"])
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
# ax_gex.xaxis.set_label_position('top')
# ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
# ax_fragments.xaxis.set_label_position('top')
# ax_fragments.xaxis.tick_top()

for ax1 in [ax_gex, ax_fragments]:
    ax2 = ax1.twiny()
    # ax2.xaxis.set_label_position('bottom')
    # ax2.xaxis.tick_bottom()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(ax1.get_xticklabels())

# %%
obs.loc[mapping[:, 0], "y"].max()

# %%
