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
# # Preprocess

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

# %%
import peakfreeatac as pfa

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

first_dataset_name = "pbmc10k"
second_dataset_name = "pbmc3k"
organism = "hs"
first_dataset_name = "pbmc10k"
second_dataset_name = "lymphoma"
organism = "hs"
first_dataset_name = "lymphoma"
second_dataset_name = "pbmc10k"
organism = "hs"

dataset_name = first_dataset_name + "+" + second_dataset_name

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome_first = pfa.data.Transcriptome(folder_data / first_dataset_name / "transcriptome")

# %%
transcriptome = pfa.data.Transcriptome(folder_data / dataset_name / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
desired_genes = transcriptome_first.var.index

# %%
adatas = []

for dataset_ix, dataset_name_ in enumerate([first_dataset_name, second_dataset_name]):
    adata = sc.read_10x_h5(folder_data / dataset_name_ / "filtered_feature_bc_matrix.h5")
    transcriptome_ = pfa.data.Transcriptome(folder_data / dataset_name_ / "transcriptome")

    adata.var.index.name = "symbol"
    adata.var = adata.var.reset_index()
    adata.var.index = adata.var["gene_ids"]
    adata.var.index.name = "gene"
    adata.obs["original_dataset"] = dataset_name_
    adata.obs["cell_original"] = adata.obs.index

    adata = adata[transcriptome_.obs.index]

    adata.obs.index = adata.obs.index.str.split("-").str[0]
    print(adata.obs.index)

    adatas.append(adata)

# %%
adata = adatas[0].concatenate(adatas[1:])

# %%
transcriptome_ = pfa.data.Transcriptome(folder_data / "pbmc10k" / "transcriptome")

# %%
sc.pp.normalize_total(adata, target_sum=size_factor)
sc.pp.log1p(adata)
adata = adata[:, desired_genes]

# %%
sc.pp.pca(adata)

# %%
adata.var["n_cells"] = np.array((adata.X > 0).sum(0))[0]

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.var["chr"] = transcriptome_first.var["chr"]

# %%
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs

# %%
transcriptome.create_X()

# %%
transcriptome.var

# %%
transcriptome.var["symbol"] = transcriptome_first.var["symbol"]

# %%
genes_oi = transcriptome_first.adata.var.sort_values("dispersions_norm", ascending=False).index[:10]
sc.pl.umap(adata, color=genes_oi, title=transcriptome.symbol(genes_oi))

# %%
sc.pl.umap(adata, color=[transcriptome.gene_id("PAX5"), "original_dataset"])

# %%
sc.pl.umap(adata, color=[*transcriptome.gene_id(["CD3D", "CD14", "PAX5"]), "original_dataset"])

# %% [markdown]
# ### Creating promoters

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)

# %%
promoters = pd.read_csv(folder_data / first_dataset_name / ("promoters_" + promoter_name + ".csv"), index_col=0)
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %%
import tabix


# %%
def gather_relative_fragments(fragments_tabix, promoters, cell_to_cell_ix, gene_to_gene_ix):
    gene_to_fragments = [[] for i in var["ix"]]
    cell_to_fragments = [[] for i in obs["ix"]]

    coordinates_raw = []
    mapping_raw = []

    for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total=promoters.shape[0]):
        gene_ix = gene_to_gene_ix[gene]
        fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])

        for fragment in fragments_promoter:
            cell = fragment[3]

            # only store the fragment if the cell is actually of interest
            if cell in cell_to_cell_ix:
                # add raw data of fragment relative to tss
                coordinates_raw.append(
                    [
                        (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                        (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"],
                    ][:: promoter_info["strand"]]
                )

                # add mapping of cell/gene
                mapping_raw.append([cell_to_cell_ix[fragment[3]], gene_ix])

    coordinates = torch.tensor(np.array(coordinates_raw, dtype=np.int64))
    mapping = torch.tensor(np.array(mapping_raw), dtype=torch.int64)

    return coordinates, mapping


# %%
obs = transcriptome.adata.obs[["cell_original", "original_dataset"]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

var = pd.DataFrame(index=promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
coordinates = []
mapping = []

for dataset_name_ in transcriptome.obs["original_dataset"].unique():
    fragments_tabix = tabix.open(str(folder_data / dataset_name_ / "atac_fragments.tsv.gz"))

    gene_to_gene_ix = var["ix"]

    cell_to_cell_ix = obs.query("original_dataset == @dataset_name_").set_index("cell_original")["ix"]

    n_cells = obs.shape[0]

    coordinates_, mapping_ = gather_relative_fragments(fragments_tabix, promoters, cell_to_cell_ix, gene_to_gene_ix)

    coordinates.append(coordinates_)
    mapping.append(mapping_)

# %%
coordinates = torch.cat(coordinates)
mapping = torch.cat(mapping)

# %%
import pathlib

fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
var.shape[0]

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %%
torch.isin(mapping[:, 0], torch.tensor(obs["ix"][(obs["original_dataset"] == "pbmc10k")].values)).sum()

# %%
torch.isin(mapping[:, 0], torch.tensor(obs["ix"][(obs["original_dataset"] == "lymphoma")].values)).sum()

# %% [markdown]
# Check size

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(coordinates.size()) * 64 / 8 / 1024 / 1024

# %% [markdown]
# Store

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# Create cellxgene index pointers

# %%
cellxgene = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]
n_cellxgene = fragments.n_genes * fragments.n_cells

# %%
import torch_sparse

# %%
cellxgene_indptr = torch.ops.torch_sparse.ind2ptr(cellxgene, n_cellxgene)

# %%
assert fragments.coordinates.shape[0] == cellxgene_indptr[-1]

# %%
fragments.cellxgene_indptr = cellxgene_indptr

# %% [markdown]
# #### Create training folds

# %%
n_bins = 1

# %%
import pathlib

fragments_first = pfa.data.Fragments(folder_data / first_dataset_name / "fragments" / promoter_name)
folds_first = pickle.load(open(fragments_first.path / "folds.pkl", "rb"))

# %%
assert (fragments_first.var.index == fragments.var.index).all()

# %%
transcriptome.obs["ix"] = np.arange(transcriptome.obs.shape[0])

# %%
joined_obs = fragments_first.obs.rename(columns=lambda x: x + "_first").join(
    fragments.obs.query("original_dataset == @first_dataset_name").reset_index().set_index("cell_original")
)

# %%
cell_ix_first_to_cell_ix = joined_obs.set_index("ix_first")["ix"]

# %%
# train/test split
cells_all = np.arange(fragments.n_cells)
new_train_cell_ids = fragments.obs.loc[
    transcriptome.obs.index[transcriptome.obs["original_dataset"] != first_dataset_name]
]["ix"].values.tolist()
genes_all = np.arange(fragments.n_regions)

folds = []
for fold_first in folds_first:
    cells_train = np.array(
        cell_ix_first_to_cell_ix[fold_first["cells_train"].tolist()].values.tolist() + new_train_cell_ids
    )
    genes_train = fold_first["genes_train"]
    cells_validation = np.array(cell_ix_first_to_cell_ix[fold_first["cells_validation"]])
    genes_validation = fold_first["genes_validation"]

    folds.append(
        {
            "cells_train": cells_train,
            "genes_train": genes_train,
            "cells_validation": cells_validation,
            "genes_validation": genes_validation,
        }
    )
pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %% [markdown]
# ## Link baseline model

# %%
baseline_location = pfa.get_git_root() / (dataset_name + "_" + "baseline_model.pkl")

# %%
baseline_location_first = pfa.get_git_root() / (first_dataset_name + "_" + "baseline_model.pkl")

# %%
# !ln -s {baseline_location_first} {baseline_location}

# %% [markdown]
# ## Link motifscans

# %%
# !mkdir {pfa.get_output()}/motifscans/{dataset_name}/

# %%
# !ln -s {pfa.get_output()}/motifscans/{first_dataset_name}/{promoter_name}/ {pfa.get_output()}/motifscans/{dataset_name}/{promoter_name}

# %%
# !ls {pfa.get_output()}/motifscans/{dataset_name}/{promoter_name}/

# %%
