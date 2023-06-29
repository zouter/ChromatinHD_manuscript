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

# %%
import IPython

if IPython.get_ipython():
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
    IPython.get_ipython().run_line_magic(
        "config", "InlineBackend.figure_format='retina'"
    )

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
from example import Example

chd.differential.plot.get_cmap_atac_diff()

# %%
# colorbar
# %%
fig_colorbar = plt.figure(figsize=(3.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chd.differential.plot.get_norm_atac_diff(),
    cmap=chd.differential.plot.get_cmap_atac_diff(),
)
colorbar = plt.colorbar(
    mappable, cax=ax_colorbar, orientation="horizontal", extend="both"
)
colorbar.set_label("Differential accessibility")
colorbar.set_ticks(np.log([0.25, 0.5, 1, 2, 4]))
colorbar.set_ticklabels(["¼", "½", "1", "2", "4"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_atac_diff")

# %%
fig_colorbar = plt.figure(figsize=(0.1, 1.0))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chd.differential.plot.get_norm_atac_diff(),
    cmap=chd.differential.plot.get_cmap_atac_diff(),
)
colorbar = plt.colorbar(
    mappable,
    cax=ax_colorbar,
    extend="both",
    orientation="vertical",
)
colorbar.set_label("Differential\naccessibility", va="center", rotation=0, ha="left ")
colorbar.set_ticks(np.log([0.25, 0.5, 1, 2, 4]))
colorbar.set_ticklabels(["¼", "½", "1", "2", "4"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_atac_diff_vertical")

# %%
motifs_to_merge = [["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"]]
example = Example(
    "lymphoma",
    "10k10k",
    "celltype",
    "v9_128-64-32",
    "cutoff_0001",
    "BCL2",
    motifs_to_merge,
    subset_clusters=["Lymphoma cycling", "B"],
)

example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_BCL2")

# %%
motifs_to_merge = [["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"]]
example = Example(
    "lymphoma",
    "10k10k",
    "celltype",
    "v9_128-64-32",
    "cutoff_0001",
    "SIAH2",
    motifs_to_merge,
    subset_clusters=["Lymphoma cycling", "B"],
)

example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_SIAH2")

# %%
motifs_to_merge = [["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"]]
example = Example(
    "lymphoma",
    "10k10k",
    "celltype",
    "v9_128-64-32",
    "cutoff_0001",
    "FCRL2",
    motifs_to_merge,
    subset_clusters=["Lymphoma cycling", "B"],
)

example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_FCRL2")

# %%
motifs_to_merge = [["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"]]
example = Example(
    "lymphoma",
    "10k10k",
    "celltype",
    "v9_128-64-32",
    "cutoff_0001",
    "RHEX",
    motifs_to_merge,
    subset_clusters=["Lymphoma cycling", "B"],
)

example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_RHEX")

# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
    ["RUNX3_HUMAN.H11MO.0.A", "RUNX1_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "AAK1",
    subset_clusters=["B", "CD4 T", "Monocytes", "NK", "cDCs"],
)

# motifs_to_merge = [["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"]]
# example = Example("lymphoma", "10k10k", "celltype", "v9_128-64-32", "cutoff_0001", "CTLA4", motifs_to_merge, subset_clusters = ["Lymphoma cycling", "B"])

# motifs_to_merge = [["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"]]
# example = Example("pbmc10k", "10k10k", "leiden_0.1", "v9_128-64-32", "cutoff_0001", "AAK1", motifs_to_merge)

example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_AAK1")


# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "NKG7",
    motifs_to_merge,
    subset_clusters=["NK"],
)
example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_NKG7")

# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "NKG7",
    motifs_to_merge,
    subset_clusters=["NK"],
)
example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_NKG7")

# %%
motifs_to_merge = []
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "IRF7",
    motifs_to_merge,
    subset_clusters=["pDCs", "Monocytes"],
)
example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_IRF7")

# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "CD74",
    motifs_to_merge,
    subset_clusters=["B", "Monocytes", "cDCs", "Plasma"],
)
example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_CD74")


# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "LYN",
    motifs_to_merge,
    subset_clusters=["B", "Monocytes", "cDCs", "Plasma"],
)
example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_CD74")

# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "TNFAIP2",
    motifs_to_merge,
    subset_clusters=["B", "Monocytes", "cDCs", "Plasma"],
)
example.fig.plot()
example.fig.show()
manuscript.save_figure(example.fig, "3", "example_differential_TNFAIP2")
# %%
motifs_to_merge = [
    ["PO5F1_HUMAN.H11MO.0.A", "PO2F2_HUMAN.H11MO.0.A"],
]
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "HLA-DQA1",
    motifs_to_merge,
    subset_clusters=["cDCs"],
)
example.fig.plot()
example.fig.show()

manuscript.save_figure(example.fig, "3", "example_differential_HLADQA1")

# %%
