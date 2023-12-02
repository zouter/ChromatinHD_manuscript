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

import torch

import tqdm.auto as tqdm

# %% [markdown]
# https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html#core_15state

# %% [markdown]
# https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/download/

# %%
import googleapiclient

# %%
SHEET_ID = "1yikGx4MsO9Ei36b64yOy9Vb6oPC5IBGlFbYEt-N6gOM"
SHEET_NAME = "Consolidated_EpigenomeIDs_summary_Table"

# %%
url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'

# %%
metadata = pd.read_csv(url, skiprows = [1, 2])

# %%
metadata["file"] = [f"{epigenome_id}_15_coreMarks_hg38lift_mnemonics.bed.gz" for epigenome_id in metadata["Epigenome ID (EID)"]]
metadata["url"] = ["https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/download/" + file for file in metadata["file"]]

# %%
biosamples_oi = pd.DataFrame([
    ["Primary T helper naive cells from peripheral blood", "CD4 T"],
    ["Primary T CD8+ naive cells from peripheral blood", "CD8 T"],
    ["Primary monocytes from peripheral blood", "Monocytes"],
    ["Primary B cells from peripheral blood", "B"],
    ["Primary Natural Killer cells from peripheral blood", "NK"]
], columns = ["Standardized Epigenome name", "cluster"])

# %%
metadata_oi = metadata.loc[metadata["Standardized Epigenome name"].isin(biosamples_oi["Standardized Epigenome name"])]

# %%
import chromatinhd as chd

# %%
raw_folder = chd.get_output() / "data" / "chmm" / "raw"
raw_folder.mkdir(exist_ok = True, parents = True)

# %%
for _, metadata_row in metadata_oi.iterrows():
    urllib.request.urlretrieve(metadata_row["url"], filename = raw_folder / metadata_row["file"])

# %%
pd.merge(metadata_oi, biosamples_oi).to_csv(raw_folder / "metadata.csv")
