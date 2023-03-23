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
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
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

dataset_name = "pbmc10k_leiden_0.1"
folder_data_preproc = folder_data / dataset_name

# %%
fragments = chd.data.fragments.ChunkedFragments(folder_data_preproc / "fragments")

# %%
chr = "chr6"; position = 89899164
window = [position - 10**5, position + 10**5]

# %%
cells_oi = range(0, 4000)

# %%
chromosome_start = fragments.chromosomes.loc[chr]["position_start"]

# %%
window_chunks = (
    (chromosome_start + window[0]) // fragments.chunk_size,
    (chromosome_start + window[1]) // fragments.chunk_size
)

# %%
chunks_from, chunks_to = fragments.chunkcoords_indptr[window_chunks[0]], fragments.chunkcoords_indptr[window_chunks[1]]

# %%
coordinates = (
    fragments.chunkcoords[chunks_from:chunks_to].to(torch.int64) * fragments.chunk_size
    + fragments.relcoords[chunks_from:chunks_to]
    - chromosome_start
)
clusters = fragments.clusters[chunks_from:chunks_to]

# %%
cluster_info = pickle.load((folder_data_preproc / ("cluster_info.pkl")).open("rb"))

# %%
bins = np.linspace(*window, 500)

# %%
main = chd.grid.Grid(padding_height=0)
fig = chd.grid.Figure(main)

for cluster_ix, cluster in enumerate(cluster_info.index):
    main[cluster_ix, 0] = panel = chd.grid.Panel((10, 0.5))
    ax = panel.ax
    ax.set_ylabel(cluster, rotation = 0, ha = "right", va = "center")
    ax.set_xticks([])
    
    coordinates_cluster = coordinates[clusters == cluster_ix]
    ax.hist(coordinates_cluster, bins = bins, lw = 0)
    ax.set_ylim(0.1 * cluster_info.loc[cluster, "n_cells"])
    ax.axvline(89899164)
main[0, -1]
fig.plot()

# %%
main[-1, 0] = 

# %%
coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix].numpy()#[cells_oi]
mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix].numpy()#[cells_oi]

# %%
# outcome = sc.get.obs_df(transcriptome.adata, gene_id)[cells_oi]
# outcome = transcriptome.adata.obs["oi"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["overexpressed"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["leiden"].cat.codes[cells_oi]
outcome = transcriptome.adata.obs["gene_overexpressed"].cat.codes[cells_oi]
# outcome = transcriptome.adata.obs["celltype"].cat.codes[cells_oi]
cell_order = outcome.sort_values().index

n_cells = len(cell_order)

obs = fragments.obs.copy()
obs.index = obs.index.astype(str)
obs["gex"] = outcome[cell_order]
obs = obs.loc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
# fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/10), sharey = True, width_ratios = [2, 0.5])
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
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
# ax_gex.xaxis.set_label_position('top')
# ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
# ax_fragments.xaxis.set_label_position('top')
# ax_fragments.xaxis.tick_top()
ax_fragments.set_xticks([4400])

for ax1 in [ax_gex, ax_fragments]:
    ax2 = ax1.twiny()
    # ax2.xaxis.set_label_position('bottom')
    # ax2.xaxis.tick_bottom()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(ax1.get_xticklabels())

# %%
fig.savefig("fragments.png", transparent=False, dpi = 300)

# %%
# fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/10), sharey = True, width_ratios = [2, 0.5])
fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (15, n_cells/300), sharey = True, width_ratios = [2, 0.5])
ax_fragments.set_xlim(*window)
ax_fragments.set_ylim(0, n_cells)

for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
    if cell_ix in obs.index:
        color = "black"
        color = "#33333333"
        rect = mpl.patches.Rectangle((start-10, obs.loc[cell_ix, "y"]), 10, 10, fc = "#33333333", ec = None, linewidth = 0)
        ax_fragments.add_patch(rect)
        rect = mpl.patches.Rectangle((end-10, obs.loc[cell_ix, "y"]), 10, 10, fc = "#33333333", ec = None, linewidth = 0)
        ax_fragments.add_patch(rect)
        
ax_gex.plot(obs["gex"], obs["y"])
ax_gex.set_xlabel(transcriptome.symbol(gene_id) + " expression")
ax_gex.xaxis.set_label_position('top')
ax_gex.xaxis.tick_top()

ax_fragments.set_xlabel("Distance from TSS")
ax_fragments.xaxis.set_label_position('top')
ax_fragments.xaxis.tick_top()

# %%

# %%

# %%
