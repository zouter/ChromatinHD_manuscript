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
sns.set_style('ticks')

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm

import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10kx"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "pbmc10k_gran"
# dataset_name = "hspc"
# dataset_name = "liver"
# dataset_name = "hspc_gmp"
dataset_name = "hspc_cycling"
# dataset_name = "pbmc20k"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Clustering2

# %%
import chromatinhd.data

# %%
for dataset_name in ["liverkia_lsecs"]:
    folder_dataset = chd.get_output() / "datasets" / dataset_name
    transcriptome = chromatinhd.data.Transcriptome(folder_dataset / "transcriptome")
    clustering = chd.data.Clustering.from_labels(transcriptome.adata.obs["celltype2"], path = folder_dataset / "latent" /  ("celltype2"), overwrite = True)
    clustering.cluster_info["dimension"] = np.arange(clustering.n_clusters)
    clustering.cluster_info = clustering.cluster_info

# %% [markdown]
# ## Clustering

# %%
import chromatinhd.data

# %%
for dataset_name in ["e18brain", "alzheimer"]:
    folder_dataset = chd.get_output() / "datasets" / dataset_name
    transcriptome = chromatinhd.data.Transcriptome(folder_dataset / "transcriptome")
    sc.pp.neighbors(transcriptome.adata)
    resolution = 0.1
    sc.tl.leiden(transcriptome.adata, resolution = resolution)
    clustering = chd.data.Clustering.from_labels(transcriptome.adata.obs["leiden"], path = folder_dataset / "latent" /  ("leiden_0.1"), overwrite = True)
    clustering.cluster_info["dimension"] = np.arange(clustering.n_clusters)
    clustering.cluster_info = clustering.cluster_info
    sc.pl.umap(transcriptome.adata, color = "leiden")

# %% [markdown]
# ## Merging cell types for PBMC10k

# %%
import chromatinhd.data

# %%
for dataset_name in [
    # "pbmc10k",
    # "pbmc3k",
    # "pbmc10kx",
    # "pbmc10k_gran",
    "liver",
    # "lymphoma",
    # "hepatocytes",
    # "hspc",
    # "pbmc20k",
]:
    folder_dataset = chd.get_output() / "datasets" / dataset_name
    transcriptome = chromatinhd.data.Transcriptome(folder_dataset / "transcriptome")
    cluster_info = pd.DataFrame(
        [[celltype, [celltype]] for celltype in transcriptome.adata.obs["celltype"].cat.categories],
        columns=["cluster", "celltypes"],
    )
    celltype_to_cluster = cluster_info.explode("celltypes").set_index("celltypes")["cluster"]
    transcriptome.adata.obs["cluster"] = celltype_to_cluster[transcriptome.adata.obs["celltype"]].values
    latent_folder = folder_dataset / "latent"
    latent_folder.mkdir(exist_ok=True)

    resolution = 0.1
    latent_name = "leiden_" + str(resolution)

    clustering = chd.data.Clustering.from_labels(
        transcriptome.adata.obs["cluster"],
        path=folder_dataset / "latent" / ("leiden_" + str(resolution)),
        overwrite=True,
    )

    sc.pl.umap(transcriptome.adata, color="celltype")

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
latent_folder.mkdir(exist_ok = True)

# %%
latent_name = "overexpression"

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
sc.pl.umap(transcriptome.adata, color = "overexpressed")

# %%
cluster_info = pd.DataFrame({"cluster":latent.columns}).set_index("cluster")
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
latent_folder.mkdir(exist_ok = True)

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %%
cluster_info = pd.DataFrame({"cluster":latent.columns}).set_index("cluster")
cluster_info["dimension"] = np.arange(len(cluster_info))
cluster_info["label"] = cluster_info.index

# %%
cluster_info.to_pickle(latent_folder / (latent_name + "_info.pkl"))

# %%
cluster_info

# %% [markdown]
# ## Cycling

# %%
for dataset_name in ["hspc_cycling", "hspc_meg_cycling", "hspc_gmp_cycling"]:
    folder_dataset = chd.get_output() / "datasets" / dataset_name
    transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")
    clustering = chd.data.Clustering.from_labels(transcriptome.adata.obs["phase"], path = folder_dataset / "latent" / "phase", overwrite = True)
    clustering.cluster_info["dimension"] = np.arange(clustering.n_clusters)
    clustering.cluster_info = clustering.cluster_info
