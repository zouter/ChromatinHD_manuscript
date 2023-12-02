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

# train_dataset_name = "pbmc10k"; test_dataset_name = "pbmc3k"; organism = "hs"
# train_dataset_name = "pbmc10k"; test_dataset_name = "lymphoma"; organism = "hs"
train_dataset_name = "pbmc10k"
dataset_name = "pbmc10k_small"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %%
dataset_name

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome_train = pfa.data.Transcriptome(folder_data / train_dataset_name / "transcriptome")

# %%
transcriptome = pfa.data.Transcriptome(folder_data / dataset_name / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = transcriptome_train.adata

# %%
sc.tl.leiden(adata, resolution=0.1)

# %%
sc.tl.rank_genes_groups(adata, "leiden")

# %%
gene_ids = set()
for group in adata.obs["leiden"].cat.categories:
    gene_ids.update(sc.get.rank_genes_groups_df(adata, group)["names"][:10])
gene_ids = list(gene_ids)

# %%
adata = adata[:, gene_ids]

# %%
sc.pp.normalize_per_cell(adata)

# %%
sc.pp.pca(adata)

# %%
adata.var["n_cells"] = np.array((adata.X > 0).sum(0))[0]

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.var["chr"] = transcriptome_train.var["chr"]

# %%
transcriptome.adata = adata

# %%
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs

# %%
transcriptome.create_X()

# %%
transcriptome.var

# %%
genes_oi = transcriptome.adata.var.sort_values("dispersions_norm", ascending=False).index[:10]
sc.pl.umap(adata, color=genes_oi, title=transcriptome.symbol(genes_oi))

# %% [markdown]
# ### Creating promoters

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data / train_dataset_name / "atac_fragments.tsv.gz"))

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)

# %%
promoters = pd.read_csv(folder_data / train_dataset_name / ("promoters_" + promoter_name + ".csv"), index_col=0)
promoters = promoters.loc[transcriptome.adata.var.index]
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %%
import pathlib

fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index=promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

if "cell_original" in transcriptome.adata.obs.columns:
    cell_ix_to_cell = transcriptome.adata.obs["cell_original"].explode()
    cell_to_cell_ix = pd.Series(cell_ix_to_cell.index.astype(int), cell_ix_to_cell.values)
else:
    cell_to_cell_ix = obs["ix"].to_dict()

n_cells = obs.shape[0]

# %%
gene_to_fragments = [[] for i in var["ix"]]
cell_to_fragments = [[] for i in obs["ix"]]

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total=promoters.shape[0]):
    gene_ix = var.loc[gene, "ix"]
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

# %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %%
coordinates = torch.tensor(np.array(coordinates_raw, dtype=np.int64))
mapping = torch.tensor(np.array(mapping_raw), dtype=torch.int64)

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

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

fragments_train = pfa.data.Fragments(folder_data / train_dataset_name / "fragments" / promoter_name)
folds_training = pickle.load(open(fragments_train.path / "folds.pkl", "rb"))

# %%
# train/test split
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_regions)

folds = []
for fold_training in folds_training:
    genes_test = fold_training["genes_validation"]
    cells_test = cells_all

    folds.append(
        {
            "cells_test": cells_all,
            "genes_test": fold_training["genes_validation"],
            "genes_train": fold_training["genes_train"],
        }
    )
pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %% [markdown]
# ## Copy baseline model

# %%
baseline_location = pfa.get_git_root() / (dataset_name + "_" + "baseline_model.pkl")

# %%
baseline_location_train = pfa.get_git_root() / (train_dataset_name + "_" + "baseline_model.pkl")

# %%
# !ln -s {baseline_location_train} {baseline_location}
