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

import pickle

import tqdm.auto as tqdm

import pathlib

import polars as pl

# %%
import peakfreeatac as pfa

# %%
folder_qtl = pfa.get_output() / "data" / "qtl" / "hs" / "gtex_caviar"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %%
# !curl --location https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_v8_finemapping_CAVIAR.tar > {folder_qtl}/GTEx_v8_finemapping_CAVIAR.tar

# %%
# !tar -xf {folder_qtl}/GTEx_v8_finemapping_CAVIAR.tar -C {folder_qtl}

# %%
# !head -n 10 {folder_qtl}/GTEx_v8_finemapping_CAVIAR/README.txt

# %%
# !head -n 5 {folder_qtl}/full.tsv
