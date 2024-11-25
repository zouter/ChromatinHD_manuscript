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
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gtex"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %%
# !curl --location https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_Analysis_v8_eQTL.tar > {folder_qtl}/GTEx_Analysis_v8_eQTL.tar

# %%
# !tar -tvf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar

# %%
# !tar -xf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar -C {folder_qtl} GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz

# %%
# !tar -xf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar -C {folder_qtl} GTEx_Analysis_v8_eQTL/Brain_Cerebellum.v8.signif_variant_gene_pairs.txt.gz

# %%
# !tar -xf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar -C {folder_qtl} GTEx_Analysis_v8_eQTL/Liver.v8.signif_variant_gene_pairs.txt.gz

# %%
# motifscan_name = "gtex_immune"; files = [
#     ["Whole_Blood.v8.signif_variant_gene_pairs.txt.gz", "whole_blood"]
# ]

motifscan_name = "gtex_cerebellum"; files = [
    ["Brain_Cerebellum.v8.signif_variant_gene_pairs.txt.gz", "cerebellum"]
]

# motifscan_name = "gtex_liver"; files = [
#     ["Liver.v8.signif_variant_gene_pairs.txt.gz", "liver"]
# ]

# %%
qtl = pd.concat([
    pd.read_table(folder_qtl/"GTEx_Analysis_v8_eQTL"/file).assign(tissue = tissue) for file, tissue in files
])

# %%
qtl["chr"], qtl["pos"] = qtl["variant_id"].str.split("_").str[0], qtl["variant_id"].str.split("_").str[1].astype(int)

# %%
qtl["abs_tss_distance"] = np.abs(qtl["tss_distance"])

# %%
qtl_mapped = qtl.copy()
qtl_mapped["snp_main"] = qtl_mapped["variant_id"]
qtl_mapped["snp"] = qtl_mapped["variant_id"]

# %%
qtl_mapped.to_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
