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
#     display_name: Python 3
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

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")


# %%
device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

#
promoter_name, window = "10k10k", np.array([-10000, 10000])

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
counts = torch.bincount(
    fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1],
    minlength=fragments.n_genes * fragments.n_cells,
)

# %%
fig, ax = plt.subplots()
ax.hist(counts[:1000], bins=np.arange(0, 10, 1))


# %%
bincounts = torch.bincount(counts, minlength=10)

# %%
plt.plot(bincounts[1:] / bincounts[:-1])

# %%
bincounts[1].sum() / bincounts.sum()

# %%
bincounts[2:].sum() / bincounts.sum()
# %%
bincounts[2:].sum() / bincounts[1].sum()
# %%
