# ---
# jupyter:
#   jupytext:
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
# # Preprocess

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
dataset_name = "hspc"
# dataset_name = "hspc_gmp"

dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
# promoter_name = "10k10k"
promoter_name = "100k100k"

# %%
transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")
fragments = chd.data.Fragments(dataset_folder / "fragments" / promoter_name)

# %%
(fragments.path / "folds").mkdir(exist_ok = True)

# %%
for dataset_name in ["pbmc10k", "hspc", "e18brain", "lymphoma"]:
    dataset_folder = chd.get_output() / "datasets" / dataset_name
    fragments = chd.data.Fragments(dataset_folder / "fragments" / promoter_name)
    folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x1", reset = True)
    folds.sample_cells(fragments, 5)

    folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x5", reset = True)
    folds.sample_cells(fragments, 5, 5)
