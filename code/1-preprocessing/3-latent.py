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
# # Infer latent cellular spaces

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

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10k"
dataset_name = "e18brain"
# dataset_name = "lymphoma"
# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "pbmc10k_gran"
# dataset_name = "hspc"

# dataset_name = "CDX1_7"
# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"
# dataset_name = "morf_20"

# dataset_name = "GSE198467_H3K27ac"
# dataset_name = "GSE198467_single_modality_H3K27me3"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Clustering

# %%
import chromatinhd.data

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
sc.pp.neighbors(transcriptome.adata)

# %%
resolution = 0.1

# %%
sc.tl.leiden(transcriptome.adata, resolution=resolution)

# %%
latent = pd.get_dummies(transcriptome.adata.obs["leiden"])
latent.columns = pd.Series("leiden_" + latent.columns.astype(str))

# %%
latent_folder = folder_data_preproc / "latent"
latent_folder.mkdir(exist_ok=True)

# %%
latent_name = "leiden_" + str(resolution)

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
sc.pl.umap(transcriptome.adata, color="leiden")

# %%
transcriptome.obs["latent"] = pd.Categorical(
    pd.from_dummies(pd.DataFrame(latent)).values[:, 0]
)

# %%
cluster_info = pd.DataFrame(
    {
        "cluster": latent.columns,
        "dimension": np.arange(latent.shape[1]),
        "color": sns.color_palette("husl", latent.shape[1]),
    }
).set_index("cluster")

if dataset_name == "pbmc10k":
    proportions = (
        transcriptome.obs.groupby("latent")["celltype"].value_counts()
        / transcriptome.obs.groupby("latent").size()
    )
    cluster_info["label"] = (
        proportions.loc[proportions > 0.05]
        .reset_index()
        .groupby("latent")
        .apply(lambda x: "; ".join(x.celltype))
    )
else:
    cluster_info["label"] = np.arange(len(cluster_info))

# %%
cluster_info.to_pickle(latent_folder / (latent_name + "_info.pkl"))

# %% [markdown]
# ## Celltypes

# %%
import chromatinhd.data

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
latent = pd.get_dummies(transcriptome.adata.obs["celltype"])

# %%
latent_folder = folder_data_preproc / "latent"
latent_folder.mkdir(exist_ok=True)

# %%
latent_name = "celltype"

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
sc.pl.umap(transcriptome.adata, color="celltype")

# %%
cluster_info = pd.DataFrame(
    {"cluster": transcriptome.adata.obs["celltype"].cat.categories}
).set_index("cluster")
# cluster_info = cluster_info.reset_index().set_index("cluster")
cluster_info["label"] = cluster_info.index
cluster_info["dimension"] = np.arange(len(cluster_info))

# %%
cluster_info.to_pickle(latent_folder / (latent_name + "_info.pkl"))

# %% [markdown]
# ## Merging cell types for PBMC10k

# %%
import chromatinhd.data

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
cluster_info = pd.DataFrame(
    [
        ["CD4 T", ["CD4 memory T", "CD4 naive T"]],
        ["CD8 T", ["CD8 activated T", "CD8 naive T"]],
        ["MAIT", ["MAIT"]],
        ["Monocytes", ["CD14+ Monocytes", "FCGR3A+ Monocytes"]],
        ["Plasma", ["Plasma"]],
        ["cDCs", ["cDCs"]],
        ["B", ["memory B", "naive B"]],
        ["NK", ["NK"]],
        ["pDCs", ["pDCs"]],
    ],
    columns=["cluster", "celltypes"],
)

# %%
celltype_to_cluster = cluster_info.explode("celltypes").set_index("celltypes")[
    "cluster"
]

# %%
transcriptome.adata.obs["cluster"] = celltype_to_cluster[
    transcriptome.adata.obs["celltype"]
].values

# %%
latent = pd.get_dummies(transcriptome.adata.obs["cluster"])

# %%
latent_folder = folder_data_preproc / "latent"
latent_folder.mkdir(exist_ok=True)

# %%
resolution = 0.1

# %%
latent_name = "leiden_" + str(resolution)

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
sc.pl.umap(transcriptome.adata, color="cluster")

# %%
cluster_info = pd.DataFrame(
    {"cluster": transcriptome.adata.obs["cluster"].cat.categories}
).set_index("cluster")
# cluster_info = cluster_info.reset_index().set_index("cluster")
cluster_info["label"] = cluster_info.index
cluster_info["dimension"] = np.arange(len(cluster_info))

# %%
cluster_info.to_pickle(latent_folder / (latent_name + "_info.pkl"))

# %% [markdown]
# ## Overexpressed

# %%
import chromatinhd.data

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
# latent = pd.get_dummies(transcriptome.adata.obs["oi"])
# latent.columns = pd.Series("leiden_" + latent.columns.astype(str))

transcriptome.adata.obs["overexpressed"] = transcriptome.adata.obs["gene_overexpressed"]
latent = pd.get_dummies(transcriptome.adata.obs["overexpressed"])

# %%
latent_folder = folder_data_preproc / "latent"
latent_folder.mkdir(exist_ok=True)

# %%
latent_name = "overexpression"

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
sc.pl.umap(transcriptome.adata, color="overexpressed")

# %%
cluster_info = pd.DataFrame({"cluster": latent.columns}).set_index("cluster")
cluster_info["dimension"] = np.arange(len(cluster_info))
cluster_info["label"] = cluster_info.index

# %%
cluster_info.to_pickle(latent_folder / (latent_name + "_info.pkl"))

# %% [markdown]
# ## Given

# %%
import chromatinhd.data

# %%
fragments = chromatinhd.data.Fragments(folder_data_preproc / "fragments" / "10k10k")

# %%
latent_column = "idents_L2"
latent_column = "seurat_clusters"
latent_name = "leiden_0.1"

# %%
# latent = pd.get_dummies(transcriptome.adata.obs["oi"])
# latent.columns = pd.Series("leiden_" + latent.columns.astype(str))

fragments.obs[latent_name] = pd.Categorical(fragments.obs[latent_column])
latent = pd.get_dummies(fragments.obs[latent_name])

# %%
latent_folder = folder_data_preproc / "latent"
latent_folder.mkdir(exist_ok=True)

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
cluster_info = pd.DataFrame({"cluster": latent.columns}).set_index("cluster")
cluster_info["dimension"] = np.arange(len(cluster_info))
cluster_info["label"] = cluster_info.index

# %%
cluster_info.to_pickle(latent_folder / (latent_name + "_info.pkl"))

# %%
cluster_info

# %%
