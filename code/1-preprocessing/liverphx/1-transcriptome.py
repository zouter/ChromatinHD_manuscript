# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
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

# %%
import chromatinhd as chd

# %%
data_folder = chd.get_output() / "data" / "liverphx" / "transcriptome" / "endothelial"
counts = pd.read_table(data_folder / "count_data.tsv")

# %%
obs = pd.read_table(data_folder / "cell_data.tsv")
var = pd.read_table(data_folder / "feature_data.tsv")

# %%
adata = sc.AnnData(X=counts.values, obs=obs, var=var)

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color = "time_experiment")

# %%
adata.var = adata.var.set_index("feature_id")
adata.var.index.name = "gene"

# %%
adata2 = adata[adata.obs["experiment"] == "phase_1"]

# %%
sc.pp.pca(adata2)
sc.pp.neighbors(adata2)
sc.tl.umap(adata2)

# %%
sc.pl.umap(adata2, color = ["time_experiment", "G2M.Score_2", "S.Score_2"])

# %%
sc.tl.rank_genes_groups(adata2, "time_experiment", method = "wilcoxon")

# %%
diffexp = sc.get.rank_genes_groups_df(adata2, None).query("group == '48_phase_1'").sort_values("scores", ascending = False)
diffexp["symbol"] = diffexp["names"].apply(lambda x: adata2.var.loc[x, "symbol"])
diffexp.index = diffexp["names"]

# %%
sc.pl.umap(adata, color = "time_experiment")

# %%
diffexp.query("pvals_adj < 1e-3")

# %%
fig, ax = plt.subplots(figsize = (4, 4))
ax.scatter(diffexp["scores"], diffexp["pvals_adj"], s = 1, alpha = 0.5)

# %%
diffexp.head(20)

# %%
genes_oi = diffexp.head(30).index
genes_oi = adata.var.reset_index().set_index("symbol").loc[["Mecom", "Dll4", "Dll1", "Hey1", "Hes1", "Sox18"]]["gene"]

# %%
sc.pl.umap(adata2, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"])

# %%
with mpl.rc_context({"font.size": 20}):
    sc.pl.umap(adata2, color = "time_experiment", legend_loc = "on data")

# %%
genes_oi = adata.var.reset_index().set_index("symbol").loc[["Dll4", "Dll1", "Fos", "Jund", "Wnt2", "Gata6", "Rspo3"]]["gene"]

sc.pl.umap(adata2, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"])

# %%
genes_oi = adata.var.loc[adata.var["symbol"].str.startswith("Adamts4")].index

sc.pl.umap(adata2, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"])
