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
import chromatinhd as chd

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gtex_caveman"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %%
# download hhttps://storage.cloud.google.com/adult-gtex/bulk-qtl/v8/fine-mapping-cis-eqtl/GTEx_v8_finemapping_CaVEMaN.tar and copy url here
# !wget "https://ff466be5e5f50e7a6977cd62ee865ec0e3a400b29494d70b6ef7459-apidata.googleusercontent.com/download/storage/v1/b/adult-gtex/o/bulk-qtl%2Fv8%2Ffine-mapping-cis-eqtl%2FGTEx_v8_finemapping_CaVEMaN.tar?jk=AanfhSAre1fBau4eS4sH30lKxuRxWkqYWKnUMvhobgZQqT4jz4HSbkL0DnJJBoAl9V7qy-5VFygdGh3W3aagbWESFYz8McKoXrXg5r_-n7gM4TMdG07ijbibgBmDHr1itpvIriYI5wrCGbgAD4l8230E9ZWGui6IDUgz9wBF0qU7mhNbdHQABv8Z33l2PeCcR8YE6Stx8a3k-6LVkbsn2cCye5DWJgro6uRK-TnIYGadzkZ0uW2kCf6ySYyMEh3ih9gcMlZVSjnHh9kEyOrl2tdhXiqxpSfp7VRA4GmRgDJeJnqI-uXvbca-V_-ELT4iztuYInwRtp5tEZFeJPiVtmut5u82SBzIYZ5DH2jTnC7hVac8nHP1r9h8d6dD-m1YUI7fB-4_zgsFS9hn1VKJj1IzfreJqwPiUcOtofylMSGyxYqWQsnMKweJrqDL_yW1HFI3UGcbcNT1gDMEWNU2oY_aDShCxCgzPJUyO0z2Qy5jQcBmQ2ncmhg6hOSO-4mi5aAyy7WcBX3OerlrqM92aZvU9X_K4mFq4874wOh1CTxxoXFKO-G7vTdd9HuTtKPn29Tjxaq9RPghTimSyX-_LN6Al3TmnYqIFLpKEuvSOhjtIpBwWdmx9GnvE6t0a_Zh5WiEdi89mapBa8TBBDGEcYA0LzvQxUEnrCThB9puySB65E6vw7l4rwKoYWEwqWq3TgZStIuvNa4DZ98nJlimAjKmr1yIQVRpjd7CYMWGz6x0faKYs14uO-KgKBpyhHBJaXuAKIndAHInrZMxTjISRYdg4mcmR2qnTiAUEmu45Xtkfjm7QKiPEPdx3s5aU_z6fAr0l6BHLAAymBnsvVkr2VXMG6EHKpf4GNxT3lvg1JzWucQXSI5YfjomvG7K8E6LzIA-EG3bEHt5Hfhst6Dx4JUf-a4f54pKmpahijsLA-2Du5M1NpGWT7uzTz-sxpua5Hjw-DootIeBd4pk6Tlqc5986CGVMHkGgvWVoWPhQV64q3yEFxotuPEyJdnYqADXK1UnwGghrY5fumf6tzyPESOTy0tq2HykBD9GZeW_raq5FaJH88_d8LkkAyOAslni9Dk6fEMgo6u5RjktkGGqbt4T96gj_yQFic89X1zp1voRT67fGFfsjkZDOtWBDItYkrvrnKIRAAxoCDXl7MriBjYvTTLucYUc_3-46xuOq17jn4DnfXDpf9cEFmES1yILk67fRKgiF8q5m4Q1b8hNKUO80Pz1f7jO2Q8p6sLFfJue_u4LGA4_Ee6kpObD1nNGGB2l&isca=1" -O {folder_qtl}/GTEx_v8_finemapping_CaVEMaN.tar

# %%
# !tar -xf {folder_qtl}/GTEx_v8_finemapping_CaVEMaN.tar -C {folder_qtl}

# %%
# !head -n 30 {folder_qtl}/GTEx_v8_finemapping_CaVEMaN/README.txt

# %%
# !ls {folder_qtl}/GTEx_v8_finemapping_CaVEMaN

# %% [markdown]
# ## Process

# %%
qtl = pd.read_table(folder_qtl / "GTEx_v8_finemapping_CaVEMaN"/ "GTEx_v8_finemapping_CaVEMaN.txt.gz")
# qtl["TISSUE"].value_counts()

# %%
motifscan_name = "gtex_immune"; tissue = "Whole_Blood"
motifscan_name = "gtex_liver"; tissue = "Liver"
# motifscan_name = "gtex_cerebellum"; tissue = "Brain_Cerebellum"

# %%
qtl_oi = qtl.loc[qtl["TISSUE"] == tissue].copy().rename(columns = {"TISSUE": "tissue", "GENE": "gene", "eQTL": "snp", "CHROM": "chr", "POS": "pos", "Probability": "prob"})
qtl_oi["snp_main"] = qtl_oi["snp"]
qtl_oi["gene"] = qtl_oi["gene"].str.split(".").str[0]

# %%
qtl_oi.to_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))

# %%
