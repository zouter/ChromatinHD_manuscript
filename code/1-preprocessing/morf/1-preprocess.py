# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
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

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
dataset_name = "morf"

# %%
import pathlib
folder_data_preproc = chd.get_output() / "data" / dataset_name
folder_data_preproc_NAS1 = pathlib.Path("/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output") / folder_data_preproc.relative_to(chd.get_output())
folder_data_preproc_NAS1.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Download

# %%
# !ls /home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/morf/

# %%
# # !rm -r {folder_data_preproc}
# # !rm -r {folder_data_preproc_NAS1}

# %%
if not folder_data_preproc.exists():
    # !ln -s {folder_data_preproc_NAS1} {folder_data_preproc}

# %%
# !echo wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217215/suppl/GSE217215_201218_ATAC.h5ad.gz -O {folder_data_preproc}/adata.h5ad.gz

# %%
# !echo wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217215/suppl/GSE217215_201218_ATAC_fragments.tsv.gz -O {folder_data_preproc.resolve()}/fragments.tsv.gz

# %%
import scanpy as sc

# %%
import gzip

# %%
# untar adata.h5ad.gz
# !gunzip -c {folder_data_preproc}/adata.h5ad.gz > {folder_data_preproc}/adata.h5ad

# %%
adata = sc.read_h5ad(folder_data_preproc / "adata.h5ad")

# %%
adata = adata[adata.obs["n_counts"]]

# %%
# sc.pp.filter_cells(data=adata, min_genes=200)
# sc.pp.filter_genes(data=adata, min_cells=3)

# %%
adata.var["n_counts"] = (adata.X > 0).sum(0)
adata.obs["n_counts"] = (adata.X > 0).sum(1)

# %%
adata = adata[(adata.obs["n_counts"] > 1000),adata.var["n_counts"] > 500]

# %%
import scipy.sparse

# %%
X = scipy.sparse.csr_matrix(adata.X)

# %%
import pickle
adata2 = sc.AnnData(X=X, obs=adata.obs, var=adata.var)
pickle.dump(adata2, open(folder_data_preproc / "adata2.pkl", "wb"))

# %% [markdown]
# ----

# %%
import pickle
adata2 = pickle.load(open(folder_data_preproc / "adata2.pkl", "rb"))

# %%
adata2.raw = adata2

# %%
import scanpy as sc
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)

# %%
sc.pp.pca(adata2, n_comps=50)
sc.pp.neighbors(adata2, n_neighbors=30, n_pcs=50)

# %%
sc.tl.umap(adata2)

# %%
adata2.obs = pd.DataFrame(adata2.obs)

# %%
adata2.obs["tf"] = adata2.obs["TF"].str.split("-").str[1:].str.join("-")

# %%
tf_colors = {
    "CDX1":"red",
    "POU2F2":"blue",
}
adata2.uns["tf_colors"] = ["white" if tf not in tf_colors else tf_colors[tf] for tf in sorted(adata2.obs["tf"].unique())]

# %%
sc.pl.umap(adata2, color = "tf")

# %%
pickle.dump(adata2, open(folder_data_preproc / "adata3.pkl", "wb"))

# %% [markdown]
# -------

# %%
import pickle
import scanpy as sc
adata3 = pickle.load(open(folder_data_preproc / "adata3.pkl", "rb"))

# %%
adata3 = adata3[(adata3.obs["tf"].value_counts() > 200)[adata3.obs["tf"]].values]

# %%
sc.tl.rank_genes_groups(adata3, "tf", reference = "mCherry")

# %%
n_diffexp = {}
for tf in adata3.obs["tf"].unique():
    try:
        diffexp = sc.get.rank_genes_groups_df(adata3, group = tf)
        diffexp = diffexp[diffexp["pvals_adj"] < 0.1]
        diffexp = diffexp[diffexp["logfoldchanges"] > 0.2]
        n_diffexp[tf] = len(diffexp)
    except KeyError:
        n_diffexp[tf] = 0
n_diffexp = pd.Series(n_diffexp)

# %%
n_diffexp.sort_values(ascending=False)

# %%
sc.get.rank_genes_groups_df(adata3, "MSGN1")
