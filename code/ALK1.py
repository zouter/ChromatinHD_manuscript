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

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
data_folder = chd.get_output() / "data" / "ALK1"
data_folder.mkdir(parents=True, exist_ok=True)
data_folder

# %%
# sftp -o "ProxyJump=liesbetm@cp0001.irc.ugent.be:22345" liesbetm@cn2031:/srv/data/liesbetm/Projects/u_mgu/Wouter/singleCell_Alk1
# lcd /home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/ALK1
# get seuratObj_zoom*

# %%
# library(Seurat)
# library(SeuratData)
# library(SeuratDisk)
# setwd("/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/ALK1")

# seu <- readRDS("seuratObj_zoomEndo.rds")
# Matrix::writeMM(seu@assays$RNA@counts, "zoomEndo.mm")
# write.table(seu@meta.data, "zoomEndo_obs.tsv")
# write.table(seu@assays$RNA@meta.features, "zoomEndo_var.tsv")


# seu <- readRDS("seuratObj_zoomHep.rds")
# Matrix::writeMM(seu@assays$RNA@counts, "zoomHep.mm")
# write.table(seu@meta.data, "zoomHep_obs.tsv")
# write.table(seu@assays$RNA@meta.features, "zoomHep_var.tsv")


# seu <- readRDS("seuratObj_zoomFibro.rds")
# Matrix::writeMM(seu@assays$RNA@counts, "zoomFibro.mm")
# write.table(seu@meta.data, "zoomFibro_obs.tsv")
# write.table(seu@assays$RNA@meta.features, "zoomFibro_var.tsv")

# %%
name = "zoomEndo"
# name = "zoomHep"
# name = "zoomFibro"

X = scipy.io.mmread(data_folder /f"{name}.mm").T.tocsr()
obs = pd.read_csv(data_folder /f"{name}_obs.tsv", sep=" ", header=0)
var = pd.read_csv(data_folder /f"{name}_var.tsv", sep=" ", header=0)

# %%
adata = anndata.AnnData(X=X.astype(np.float32), obs=obs, var=var)

# %%
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata

# %%
adata = adata[:, adata.var.highly_variable]
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color = ["condition", "sample", "exp", "Dll4", "Stab2", "Glul", "Rspo3"])

# %%
sc.pl.umap(adata, color = ["condition", "sample", "exp", "Dll4", "Dll1"])

# %%
sc.pl.dotplot(adata, ["Dll4"], groupby = "sample", dot_min = 0, dot_max = 0.3)

# %%
# find differential between exp1-3
adata_ctrl = adata[adata.obs["exp"].isin(["exp1", "exp2", "exp3"]), :]
sc.tl.rank_genes_groups(adata_ctrl, "sample", method="t-test")
diffexp = pd.DataFrame({
    "lfc_max":sc.get.rank_genes_groups_df(adata_ctrl, group=None).groupby("names")["logfoldchanges"].max(),
    "pval_min":sc.get.rank_genes_groups_df(adata_ctrl, group=None).groupby("names")["pvals_adj"].min(),
})
diffexp["significant"] = (diffexp["lfc_max"] > 0.7) & (diffexp["pval_min"] < 0.05)
genes_oi = diffexp.index[~diffexp["significant"]]

# %%
diffexp.query("significant").sort_values("pval_min", ascending = True).head(20)

# %%
sc.pl.dotplot(adata, ["Pck1", "Dbp", "Ass1", "Cry1", "Ttr", "Alb", "Mki67", "Nr1d1", "Arntl", "Npas2", "Rorc", "Per2"], groupby = "sample", dot_min = 0, dot_max = 0.3)

# %%
sc.pl.dotplot(adata, ["Alb", "mt-Co3", "Fos"], groupby = "sample", dot_min = 0, dot_max = 0.3)

# %%
# find differential between exp4
adata_ko = adata[adata.obs["exp"].isin(["exp4"]), :]
sc.tl.rank_genes_groups(adata_ko, "sample", method="t-test")
diffexp = sc.get.rank_genes_groups_df(adata_ko, group=None).query("group == 'CS199'")
diffexp["significant"] = (diffexp["logfoldchanges"].abs() > 0.7) & (diffexp["pvals_adj"] < 0.05)

# %%
diffexp.sort_values("scores", ascending = True).head(20)

# %%
sc.pl.dotplot(adata, ["Zbtb20", "Cyb5a", "Actb", "Acox1", "Aldh2", "Prdx1", "Cyp26b1", "Thrsp", "Acaa1b", "Alb"], groupby = "sample", dot_min = 0, dot_max = 0.3)

# %%
sc.pl.umap(adata, color = ["condition", "sample", "exp", "Zbtb20", "Actb", "Cyb5a", "Sox9", "Spp1"])
