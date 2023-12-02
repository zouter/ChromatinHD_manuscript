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
sns.set_style('ticks')

import torch


import pickle

import scanpy as sc

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

train_dataset_name = "pbmc10k"; test_dataset_name = "pbmc3k"; organism = "hs"
train_dataset_name = "pbmc10k"; test_dataset_name = "lymphoma"; organism = "hs"
# train_dataset_name = "pbmc10k"; test_dataset_name = "pbmc10k_gran"; organism = "hs"

dataset_name = test_dataset_name + "-" + train_dataset_name

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %%
dataset_name

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome_train = chd.data.Transcriptome(folder_data / train_dataset_name / "transcriptome")

# %%
transcriptome = chd.data.Transcriptome(folder_data / dataset_name / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = sc.read_10x_h5(folder_data / test_dataset_name / "filtered_feature_bc_matrix.h5")

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

# %%
desired_genes =  transcriptome_train.var.index
common_genes = desired_genes.intersection(adata.var.index)

assert len(common_genes) == len(desired_genes)

# %%
adata = adata[:, common_genes]

# %%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts = 1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes= 200)
print(adata.obs.shape[0])

# %%
sc.external.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind = "hist")

# %%
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
sc.pp.normalize_per_cell(adata)

# %%
sc.pp.log1p(adata)

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
genes_oi = transcriptome_train.adata.var.sort_values("dispersions_norm", ascending = False).index[:10]
sc.pl.umap(adata, color=genes_oi, title = transcriptome.symbol(genes_oi))

# %% [markdown]
# ### Creating promoters

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data / test_dataset_name / "atac_fragments.tsv.gz"))

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)

# %%
promoters = pd.read_csv(folder_data / train_dataset_name / ("promoters_" + promoter_name + ".csv"), index_col = 0)
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %%
import pathlib
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
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

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    gene_ix = var.loc[gene, "ix"]
    fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])
    
    for fragment in fragments_promoter:
        cell = fragment[3]
        
        # only store the fragment if the cell is actually of interest
        if cell in cell_to_cell_ix:
            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([
                cell_to_cell_ix[fragment[3]],
                gene_ix
            ])

# %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %%
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int64))
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %% [markdown]
# Store

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# Create cellxgene index pointers

# %%
fragments.create_cellxgene_indptr()

# %% [markdown]
# ## Create training folds

# %%
(fragments.path / "folds").mkdir(exist_ok = True)

# %% [markdown]
# ### Default

# %% [markdown]
# ### All

# %%
n_bins = 1
splitter = "all"
folds = [{"cells_test":np.arange(fragments.n_cells)}]
pickle.dump(folds, (fragments.path / "folds" / (splitter + ".pkl")).open("wb"))

# %% [markdown]
# ## Copy peaks

# %%
folder_peaks_test = chd.get_output() / "peaks" / dataset_name
folder_peaks_train = chd.get_output() / "peaks" / train_dataset_name

# %%
# !mkdir -p {folder_peaks_test}

# %%
# !cp -r {folder_peaks_train}/* {folder_peaks_test}/

# %% [markdown]
# ## Softlink fragments

# %%
folder_data_preproc = folder_data / dataset_name
test_folder_data_preproc = folder_data / test_dataset_name

# %%
# !ln -s {test_folder_data_preproc}/atac_fragments.tsv.gz {folder_data_preproc}
# !ln -s {test_folder_data_preproc}/atac_fragments.tsv.gz.tbi {folder_data_preproc}

# %%
# !ls {test_folder_data_preproc}

# %%
