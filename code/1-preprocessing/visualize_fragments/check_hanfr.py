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
fragments.coordinates[:]