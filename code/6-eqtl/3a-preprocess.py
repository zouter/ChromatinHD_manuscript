# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

original_dataset_name = "pbmc10k"
dataset_name = "pbmc10k_eqtl"; genome = "GRCh38.107"

original_folder_data_preproc = folder_data / original_dataset_name
folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

if not (folder_data_preproc / "genome").exists():
    genome_folder = chd.get_output() / "data" / "genomes" / genome
    (folder_data_preproc / "genome").symlink_to(genome_folder)

# %%
# !ln -s {original_folder_data_preproc}/atac_fragments.tsv.gz {folder_data_preproc}/atac_fragments.tsv.gz
# !ln -s {original_folder_data_preproc}/atac_fragments.tsv.gz.tbi {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %%
import pathlib
scores = pickle.load(pathlib.Path("scores.pkl").open("rb"))
scores["significant"] = scores["bf"] > np.log(10)

# %%
variants_oi = scores.groupby("variant")["significant"].any()
variants_oi = variants_oi[variants_oi].index

# %%
fragment_dataset_name = "pbmc10k_leiden_0.1"
genotyped_dataset_name = fragment_dataset_name + "_gwas"
genotyped_dataset_folder = chd.get_output() / "datasets" / "genotyped" / genotyped_dataset_name
genotype = chd.data.genotype.genotype.Genotype(genotyped_dataset_folder / "genotype")
variants_info = genotype.variants_info.loc[variants_oi]

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Creating promoters

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "atac_fragments.tsv.gz"))

# %% [markdown]
# #### Define promoters

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

# %%
import pybedtools

# %%
promoters = pd.DataFrame({
    "chr":variants_info["chr"],
    "tss":variants_info["start"],
    "start":variants_info["start"] - padding_negative,
    "end":variants_info["end"] + padding_positive,
    "strand":+1
})
promoters.index.name = "gene"

# %%
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %% [markdown]
# #### Create fragments

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = chromatinhd.data.Transcriptome(original_folder_data_preproc / "transcriptome")

# %%
import pathlib
import chromatinhd.data
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
# Check size

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

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
fragments.create_cellxgene_indptr()

# %%
# !ls {original_folder_data_preproc}

# %%
# !ln -s {original_folder_data_preproc}/latent {folder_data_preproc}/latent
