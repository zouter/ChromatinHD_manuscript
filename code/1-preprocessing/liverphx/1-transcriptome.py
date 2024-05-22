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
# # %config InlineBackend.figure_format='retina'

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
cors = np.corrcoef(adata.X.T)

# %%
print("\n".join(pd.DataFrame(cors, adata.var["symbol"], adata.var["symbol"])["Myc"].sort_values(ascending = False).head(40).index.tolist()))

# %%
adata.X

# %%
np.corrcoef()

# %%
adata.var = adata.var.set_index("feature_id")
adata.var.index.name = "gene"

# %%
if not pathlib.Path("regev_lab_cell_cycle_genes.txt").exists():
    # !wget https://raw.githubusercontent.com/scverse/scanpy_usage/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt
cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
s_genes = [adata.var.index[adata.var["symbol"] == x.capitalize()][0] for x in s_genes if x.capitalize() in adata.var["symbol"].tolist()]
g2m_genes = [adata.var.index[adata.var["symbol"] == x.capitalize()][0] for x in g2m_genes if x.capitalize() in adata.var["symbol"].tolist()]
sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
sc.pl.umap(adata, color = ["phase"])

# %% [markdown]
# ## Subset phase 1

# %%
adata2 = adata[adata.obs["experiment"] == "phase_1"]

# %%
sc.pp.pca(adata2)
sc.pp.neighbors(adata2)
sc.tl.umap(adata2)

# %%
sc.tl.leiden(adata2)

# %%
sc.pl.umap(adata2, color = ["time_experiment", "leiden", "phase"])

# %%
sc.tl.rank_genes_groups(adata2, "time_experiment", method = "wilcoxon")
sc.tl.rank_genes_groups(adata2, "leiden", method = "wilcoxon", key_added = "leiden")

# %%
diffexp = sc.get.rank_genes_groups_df(adata2, None).query("group == '48_phase_1'").sort_values("scores", ascending = False)
diffexp["symbol"] = diffexp["names"].apply(lambda x: adata2.var.loc[x, "symbol"])
diffexp.index = diffexp["names"]

# %%
sc.pl.umap(adata2, color = "time_experiment")

# %%
diffexp = sc.get.rank_genes_groups_df(adata2, None, key = "leiden")
diffexp.index = diffexp["names"].values
diffexp["significant"] = diffexp["pvals_adj"] < 0.05
diffexp = diffexp.groupby("names").agg({"scores": "mean", "pvals_adj": "mean", "significant": "any"}).sort_values("scores", ascending = False)
diffexp["gene"] = diffexp.index.values
diffexp["symbol"] = diffexp["gene"].apply(lambda x: adata2.var.loc[x, "symbol"])
diffexp.to_csv(data_folder / "diffexp.csv")

# %%
adata.var.loc[adata.var["symbol"].str.startswith("Gata")]

# %%
adata.var.loc[adata.var["symbol"].str.startswith("Irf")]["symbol"].tolist()

# %%
symbol = "Irf4"
gene_id = adata.var.query("symbol == @symbol").index.values[0]
sc.pl.umap(adata2, color = gene_id, title = symbol)

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
genes_oi = adata.var.reset_index().set_index("symbol").loc[["Dll4", "Dll1", "Fos", "Jund", "Wnt2", "Gata6", "Rspo3", "Wnt9b", "Fosl2", "Efnb2", "Ltbp4", "Thbd", "Kit", "Cdh13", "Lyve1", "Fabp4"]]["gene"]

sc.pl.umap(adata2, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"])

# %% [markdown]
# ## Create central-portal data

# %%
adata_control = adata2[(adata2.obs["time_experiment"] == "0_phase_1") & (adata2.obs["phase"] == "G1")]
sc.pp.pca(adata_control)
sc.pp.neighbors(adata_control)
sc.tl.umap(adata_control)

# %%
adata_48h = adata2[(adata2.obs["time_experiment"] == "48_phase_1") & (adata2.obs["phase"] == "G1")]
sc.pp.pca(adata_48h)
sc.pp.neighbors(adata_48h)
sc.tl.umap(adata_48h)

# %%
sc.pl.umap(adata_control, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"])
sc.pl.umap(adata_48h, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"])

# %%
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata_control.X)
adata_control.layers["magic"] = X_smoothened

# %%
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata_48h.X)
adata_48h.layers["magic"] = X_smoothened

# %%
genes_oi = adata.var.reset_index().set_index("symbol").loc[["Dll4", "Dll1", "Wnt2", "Rspo3", "Wnt9b", "Efnb2", "Ltbp4", "Thbd", "Kit", "Cdh13", "Lyve1", "Fabp4"]]["gene"]

# %%
adata_control_limited = adata_control[:, genes_oi]
adata_control_limited.X = adata_control_limited.layers["magic"]
sc.pp.pca(adata_control_limited)

adata_48h_limited = adata_48h[:, genes_oi]
adata_48h_limited.X = adata_48h_limited.layers["magic"]
sc.pp.pca(adata_48h_limited)

# %%
FIGSIZE=(3,3)
mpl.rcParams['figure.figsize']=FIGSIZE

# %%
sc.pl.pca(adata_control_limited, color = genes_oi, title = adata.var.loc[genes_oi, "symbol"], ncols = 8)

# %%
sc.pl.pca(adata_48h_limited, color = genes_oi, layer = "magic", title = adata.var.loc[genes_oi, "symbol"])

# %%
labels = pd.Series("cycling", index = adata2.obs.index)
labels[adata_control_limited.obs.index[(adata_control_limited.obsm["X_pca"][:, 0] < 0)]] = "lsec-portal-sham"
labels[adata_control_limited.obs.index[(adata_control_limited.obsm["X_pca"][:, 0] >= 0)]] = "lsec-central-sham"
labels[adata_48h_limited.obs.index[(adata_48h_limited.obsm["X_pca"][:, 0] < 0)]] = "lsec-portal-48h"
labels[adata_48h_limited.obs.index[(adata_48h_limited.obsm["X_pca"][:, 0] >= 0)]] = "lsec-central-48h"

# %%
dataset_folder = chd.get_output() / "datasets" / "liverphx_48h"

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata2, path = dataset_folder / "transcriptome")
clustering = chd.data.clustering.Clustering.from_labels(labels, path = dataset_folder / "clusterings" / "portal_central", overwrite = True)
