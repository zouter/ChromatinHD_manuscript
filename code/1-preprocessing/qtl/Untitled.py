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

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests
import xarray as xr

# %%
data = np.zeros((100, 100))
cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(-2, 2)
data[:50, :50] = np.random.normal(np.random.normal(-1, 0.4, size = (50, 1)), 0.5, size = (50, 50))
data[50:, :50] = np.random.normal(np.random.normal(1, 0.4, size = (50, 1)), 0.5, size = (50, 50))
data[:50, 50:] = np.random.normal(np.random.normal(1, 0.4, size = (50, 1)), 0.5, size = (50, 50))
data[50:, 50:] = np.random.normal(np.random.normal(-1, 0.4, size = (50, 1)), 0.5, size = (50, 50))

fig, ax = plt.subplots()
ax.matshow(data, cmap = cmap, norm = norm)
