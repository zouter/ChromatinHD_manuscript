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

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %%
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

# %% [markdown]
# ## Get the dataset

# %% [markdown]
# ### Full real dataset

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %% [markdown]
# ## Create loaders

# %%
fragments.window = window
fragments.create_cut_data()

# %%
cells_train = np.arange(0, int(fragments.n_cells * 9 / 10))
cells_validation = np.arange(int(fragments.n_cells * 9 / 10), fragments.n_cells)

# %%
import chromatinhd.loaders.fragments

# n_cells_step = 1000
# n_regions_step = 1000

n_cells_step = 100
n_regions_step = 5000

# n_cells_step = 2000
# n_regions_step = 500

loaders_train = chromatinhd.loaders.pool.LoaderPoolOld(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments": fragments, "cellxregion_batch_size": n_cells_step * n_regions_step},
    n_workers=20,
    shuffle_on_iter=True,
)
minibatches_train = chd.loaders.minibatching.create_bins_random(
    cells_train,
    np.arange(fragments.n_regions),
    fragments.n_genes,
    n_regions_step=n_regions_step,
    n_cells_step=n_cells_step,
    use_all=True,
    permute_regions=False,
)
loaders_train.initialize(minibatches_train)

loaders_validation = chromatinhd.loaders.pool.LoaderPoolOld(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments": fragments, "cellxregion_batch_size": n_cells_step * n_regions_step},
    n_workers=5,
)
minibatches_validation = chd.loaders.minibatching.create_bins_random(
    cells_validation,
    np.arange(fragments.n_regions),
    fragments.n_genes,
    n_regions_step=n_regions_step,
    n_cells_step=n_cells_step,
    use_all=True,
    permute_regions=False,
)
loaders_validation.initialize(minibatches_validation)

# %%
data = loaders_train.pull()

# %%
data.cut_coordinates

# %%
data.cut_local_cell_ix

# %%
data.cut_local_gene_ix

# %%
data.cut_local_cellxgene_ix

# %% [markdown]
# ## Model

# %% [markdown]
# ### Load latent space

# %%
# torch works by default in float32
latent = np.random.rand(fragments.n_cells).astype(np.float32)

# %% [markdown]
# ### Create model

# %%
import chromatinhd.models.likelihood_pseudotime.v1 as likelihood_model

model = likelihood_model.Decoding(fragments, torch.from_numpy(latent), nbins=(32,))

# %%
model.latent

# %%
model.forward(data)
