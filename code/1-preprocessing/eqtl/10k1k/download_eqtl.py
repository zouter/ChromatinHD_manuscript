# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

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
import pathlib

# %% tags=[]
# !ls -lh {chd.get_output()}/"data"/eqtl/onek1k

# %% tags=[]
raw_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k" / "raw"
eqtl_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k" / "eqtl"
eqtl_data_folder.mkdir(exist_ok = True)

# %% [markdown]
# ### Download

# %% tags=[]
# !wget -P {raw_data_folder} https://onek1k.s3.ap-southeast-2.amazonaws.com/onek1k_eqtl_dataset.zip

# %% tags=[]
# !rm {eqtl_data_folder}/onek1k_eqtl_dataset.tsv

# %% tags=[]
# !unzip -d {eqtl_data_folder} {eqtl_data_folder}/onek1k_eqtl_dataset.zip

# %% tags=[]
# !ls -lh {eqtl_data_folder}/onek1k_eqtl_dataset.tsv

# %% [markdown]
# Something weird is going on with this tsv given that the first line contains both the header and the first line of data...

# %% tags=[]
# !head -n 3 {eqtl_data_folder}/onek1k_eqtl_dataset.tsv

# %% tags=[]
eqtl_file = eqtl_data_folder / "onek1k_eqtl_dataset.tsv"

# %% tags=[]
# with eqtl_file.open("r") as original: data = original.read()
# a, b = data.split("\n", 1)
# a1, a2 = a.split("bin")
# a2 = "bin" + a2
# with eqtl_file.open("w") as modified: modified.write(a1 + "\n" + a2 + "\n" + b)

# %% tags=[]
# !head -n 3 {eqtl_file}

# %% tags=[]
# !awk 'NR==390136' {eqtl_file}

# %% tags=[]
# Inconsistency!!! This SNP should be significant for CTLA4, but we only get data for a nearby gene WDR12, and it is on top of that not significant for that! There are other SNPs for CTLA4, which are significant, but this one seems to be missing
# There is therefore clearly something wrong with the SNP data...
# !grep rs3087243 {eqtl_file}

# %% tags=[]
# !grep CTLA4 {eqtl_file} > test.csv

# %% tags=[]
pd.read_table("test.csv", names = eqtl.columns).query("p_value < 0.1").sort_values("p_value").head(20)

# %% [markdown] tags=[]
# Something else is weird about the data: some genomics positions are not stored as full integers, but rather in scientific notation, e.g. `7.2e+07`. I hope this truly means that the position is 72,000,000 or otherwise we're a bit screwed. Just as a check, not all positions of this size are stored like this, so I think we're safe. It just means there literally was a SNP at 72e6 and the author's software thought it was funny to safe it as such...

# %%
# !head -n 100000 {eqtl_file} | tail -n 1

# %%
import polars as pl

# %%
eqtl = pl.read_csv(
    eqtl_file,
    sep = "\t",
    has_header = True,
    dtypes = {
        "POS":pl.Float64,
        "CELL_ID":pl.Categorical,
        "CELL_TYPE":pl.Categorical,
        "GENE":pl.Categorical,
        "GENE_ID":pl.Categorical,
        "GENOTYPED":pl.Categorical
    }
)
eqtl = eqtl.with_column(pl.col("POS").cast(pl.Int64))
eqtl = eqtl.rename(dict(zip(eqtl.columns, [col.lower() for col in eqtl.columns])))

# %% [markdown]
# Let's store this in a more efficient format than a 13GB tsv 🙄

# %% tags=[]
eqtl["cell_id"].value_counts()

# %%
eqtl.filter(pl.col("q_value") < 0.05)["cell_id"].value_counts().to_pandas().set_index("cell_id").plot(kind = "bar")

# %% [markdown]
# Corresponds quite well to figure 2B (bottom right):
# ![](https://www.science.org/cms/10.1126/science.abf3041/asset/55db8185-d844-4f4d-90dd-e86b32693427/assets/images/large/science.abf3041-f2.jpg)

# %% [markdown]
# It's important to note that the dataframe is sorted by celltype, chromosome and position. We will resort it here by chromosome and position, ignoring the celltype. This is useful for any binary search later

# %%
eqtl = eqtl.sort([pl.col("chr"), pl.col("pos")])

# %%
eqtl_file2 = raw_data_folder / "onek1k_eqtl_dataset.parquet"

# %%
eqtl.write_parquet(eqtl_file2)

# %% [markdown]
# ## Explore

# %% tags=[]
eqtl_file2 = raw_data_folder / "onek1k_eqtl_dataset.parquet"

# %% tags=[]
pip install pyarrow

# %% tags=[]
import polars as pl
import pyarrow

# %% tags=[]
eqtl = pl.read_parquet(eqtl_file2)

# %% tags=[]
eqtl_filtered = eqtl.filter(pl.col("fdr") < 0.05).to_pandas()

# %% tags=[]
eqtl_filtered.groupby(["rsid", "gene"], observed = True).size().sort_values(ascending = False).head(20)

# %% tags=[]
sns.histplot(eqtl.head(1000000)["spearmans_rho"].to_pandas())
sns.histplot(eqtl.filter(pl.col("fdr") < 0.05).head(1000000)["spearmans_rho"].to_pandas())

# %% tags=[]
eqtl.filter(pl.col("gene_id") == "ENSG00000203747")

# %% tags=[]
eqtl.filter(pl.col("rsid") == "rs207253")

# %%
eqtl.filter(pl.col("gene") == "CTLA4").to_pandas().query("q_value < 0.1").sort_values("p_value")

# %% tags=[]
eqtl.filter(pl.col("rsid") == "rs3087243")

# %% tags=[]
# !grep rs3087243 {eqtl_file}
