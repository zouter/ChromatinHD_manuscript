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

dataset_name = "3dram"
genome = "GRCh38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %%
# !mkdir -p /home/wsaelens/SVRAW1/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/3dram

# %%
# !ln -s /home/wsaelens/SVRAW1/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/3dram {chd.get_output() / "data/3dram"}

# %%
# !rm -r /home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/3dram

# %%
# !ls {chd.get_output() / "data/3dram"}

# %%
if not (folder_data_preproc / "GSE211736_RAW.tar").exists():
    # ! wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE211nnn/GSE211736/suppl/GSE211736_RAW.tar -O {folder_data_preproc / "GSE211736_RAW.tar"}

# %%
if not (folder_data_preproc / "GSE211736_RAW").exists():
    # ! tar -xvf {folder_data_preproc / "GSE211736_RAW.tar"} -C {folder_data_preproc}
