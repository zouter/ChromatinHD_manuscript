# ---
# jupyter:
#   jupytext:
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

import chromatinhd as chd

# %%
import tobias

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "pbmc10k_gran"

# dataset_name = "CDX1_7"
# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"
# dataset_name = "morf_20"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
