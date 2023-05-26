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

# %% [markdown]
# # Get Hi-C data

# %%
# %load_ext autoreload
# %autoreload 2

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

import chromatinhd as chd
import chromatinhd_manuscript as chdm

import cooler

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"

promoter_name, promoter = "100k100k", np.array([-100000, 100000])
# promoter_name, promoter = "10k10k", np.array([-10000, 10000])

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

cool_name = "rao_2014_1kb"

if cool_name == "rao_2014_1kb":
    c = cooler.Cooler(
        str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000"
    )

hic_file = folder_data_preproc / "hic" / promoter_name / f"{cool_name}.pkl"
hic_file.parent.mkdir(exist_ok=True, parents=True)

# %%
# load or create gene hics
import pickle
import pathlib

if not hic_file.exists():
    gene_hics = {}
    for gene in tqdm.tqdm(promoters.index):
        promoter = promoters.loc[gene]
        promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"

        import cooler

        try:
            hic, bins_hic = chdm.hic.extract_hic(promoter, c=c)
        except ValueError:
            print(f"Could not extract Hi-C for {gene}")
            continue

        gene_hics[gene] = (hic, bins_hic)
    pickle.dump(gene_hics, open(hic_file, "wb"))
