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
import polyptich as pp
pp.setup_ipython()

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
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gtex_caviar"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %%
# download https://storage.cloud.google.com/adult-gtex/bulk-qtl/v8/fine-mapping-cis-eqtl/GTEx_v8_finemapping_CAVIAR.tar and copy url here
# !wget "https://ff5c046f8f1875da2608499ab27fca100e7cfa1d83f6f221e9ca809-apidata.googleusercontent.com/download/storage/v1/b/adult-gtex/o/bulk-qtl%2Fv8%2Ffine-mapping-cis-eqtl%2FGTEx_v8_finemapping_CAVIAR.tar?jk=AanfhSA2jHTSBddSaD8Mn5fYGciUZ2xVsxwtyT0UyxUxSMJAq0AOQ2g4qvWinB8ahuHPMGZgiOZHb4713ETkHjVx-qdl0ctw57VWcvi-kGNqgfjPIf_olNqb-qlcIXiCTSP4GkxszrhzoJZmYoHpgaVAl-JZRJdcHHwfqHxN6tsMe_JmfBjS92ySNwv6QBuoAY2Ogc8lNjFTiugGKlsqDx6DQWkPeyEkTW75mennuktNAObEDOP30DmOAhl9PsAfIJ-tReJOqcy5sUlGyltKk4LvSMTKjvb0wNzkhW19HyMUNi0iJZ9e4FiptzrZGyVbWCw6sjE5tpZviXSGOD5TB5NSKEfpdduTD2u6iP0iBpKPh1L6-2NfBCdZYfF6liXPL8mPMts2SX4XlvSuIxJMNtevG794_qnNpyxXXC6bwZrFdvfKGpoVqaH9DElRylhcD7pA7jZVnoOKiGelo6KU2ytb46IZqYLJVjqXErsdlxZ_mNSK042MfwlEv3wwmnkQ-b1Osxid1OdG_1LEUf9Tf7jCMdOJHYwYlRoHEkpouCHXFuLIuNUeOyBV88cbSQMJuyvd9-HjM3X8rNCvmR2MvtQ6QhcRBaeN9fwZXTwjKIDDd-9RAV-V86tVuXKC8T9nH2kG7MjaGD__gIxWnjTOO2TtAmrm807umKpkjyhWoxuD2ahtV3ZnES2N_mzXP1ygnKl1Xgfn9yvNbyDTe1NkQ6NGN9Sqtj5xESpLJwNV-LsPXUwBcAu_dWhIvMw-3xlwjY4FaPh5AMsOG0gLKtVoUSUSAkRALsMvIOBUbsMxbFu11mPd5L7cIJk19OwwPf05bjBlSvH0wj8fGKnlm68kPhPdo2-MXOpSCai3YD9rKX4TSxc2nvrLgu6ssW2WXwnNsqoAnQq7jsm4XItMtdgb3z2jyJXJqviYmoF13IZNmyZ0WHId80616jTKZroB9_ToUke4HT8MNv6vhfg0-XWxtboLWmgi6qvoLrADi2dSG4Bm6REgwG2kDopcInu3Ng33wkr57kVO1u0niDsz_B9q-dSAPljwbgLSfCtaYFYZ5HRRaDchhLj9kfWUBUD1F5hAfQ9B1X1CpOBl7r6NBi8ra2GpTLFoPwmuqrkJoJYjkAzYAedPN2gg0gTTu4SLCtV1i3yIVQ8HYbRIQLL6zT-7AVKJeEhPzUux3_9vVzXOVnfvQTr8l-od7ZEb7nloRTCFMokPdu74bKFqFoZd05znQv6W1-vpgaah3R-YmvFrvpY46ON8peSYLofyjJ1X-sMaLKM&isca=1" -O {folder_qtl}/GTEx_v8_finemapping_CAVIAR.tar

# %%
# !tar -xf {folder_qtl}/GTEx_v8_finemapping_CAVIAR.tar -C {folder_qtl}

# %%
# !head -n 30 {folder_qtl}/GTEx_v8_finemapping_CAVIAR/README.txt

# %% [markdown]
# ## Process

# %%
qtl = pd.read_table(folder_qtl / "GTEx_v8_finemapping_CAVIAR"/ "CAVIAR_Results_v8_GTEx_LD_HighConfidentVariants.gz")
# qtl["TISSUE"].value_counts()

# %%
qtl["TISSUE"].value_counts()

# %%
motifscan_name = "gtex_immune"; tissue = "Whole_Blood"
motifscan_name = "gtex_liver"; tissue = "Liver"
motifscan_name = "gtex_cerebellum"; tissue = "Brain_Cerebellum"

# %%
qtl_oi = qtl.loc[qtl["TISSUE"] == tissue].copy().rename(columns = {"TISSUE": "tissue", "GENE": "gene", "eQTL": "snp", "CHROM": "chr", "POS": "pos", "Probability": "prob"})
qtl_oi["snp_main"] = qtl_oi["snp"]
qtl_oi["chr"] = "chr" + qtl_oi["chr"].astype(str)
qtl_oi["gene"] = qtl_oi["gene"].str.split(".").str[0]

# %%
qtl_oi.to_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))

# %%
# !head -n 5 {folder_qtl}/full.tsv
