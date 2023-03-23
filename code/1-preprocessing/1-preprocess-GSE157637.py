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

dataset_name = "GSE198467_single_modality_H3K27me3"; organism = "mm"; genome = "mm10"
# dataset_name = "GSE198467_H3K27ac"; organism = "mm"; genome = "mm10"
# dataset_name = "GSE198467_H3K27me3"; organism = "mm"; genome = "mm10"
# dataset_name = "GSE198467_ATAC"; organism = "mm"; genome = "mm10"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)
    
if not (folder_data_preproc / "genome").exists():
    genome_folder = chd.get_output() / "data" / "genomes" / genome
    (folder_data_preproc / "genome").symlink_to(genome_folder)

# %% [markdown]
# ## Download

# %% [markdown]
# For an overview on the output data format, see:
# https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/algorithms/overview

# %%
# ! echo mkdir -p {folder_data_preproc}
# ! echo mkdir -p {folder_data_preproc}/bam

# %%
main_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198467/suppl/"

# %%
# !wget {main_url}/{dataset_name}_fragments.tsv.gz -O {folder_data_preproc}/fragments.tsv.gz

# %%
# !wget {main_url}/{dataset_name}_01.clustering.h5seurat -O {folder_data_preproc}/seurat.h5seurat
# # !wget {main_url}/{dataset_name}_Seurat_object_clustered_renamed.h5seurat -O {folder_data_preproc}/seurat.h5seurat

# %% [markdown]
# ## Get obs

# %%
Rscript = """
# sudo apt-get install libhdf5-dev
# if (!requireNamespace("remotes", quietly = TRUE)) {
#   install.packages("remotes")
# }
# remotes::install_github("mojaveazure/seurat-disk")
# remotes::install_github("satijalab/seurat-data")
library(SeuratDisk)
seu <- SeuratDisk::LoadH5Seurat('""" + str(folder_data_preproc / "seurat.h5seurat") + """')

write.table(seu@meta.data, "obs.tsv")
"""
with open("/tmp/script.R", "w") as outfile:
    outfile.write(Rscript)

# %%
R_location = "/data/peak_free_atac/software/R-4.2.2/bin/Rscript"

# %%
# !{R_location} /tmp/script.R

# %%
obs = pd.read_table("obs.tsv", sep = " ")

# %%
obs.index.name = "cell"

# %%
obs.to_csv("obs.tsv", sep = "\t")

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Creating promoters

# %%
import tabix
import pysam

# %%
# !gunzip -t {folder_data_preproc}/fragments.tsv.gz

# %%
# create tabix index if not available
if not (folder_data_preproc / "fragments.tsv.gz.tbi").exists():
    pysam.tabix_index(
        str(folder_data_preproc / "fragments.tsv.gz"),
        seq_col=0,
        start_col=1,
        end_col=2,
        force=True,
    )

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "fragments.tsv.gz"))

# %% [markdown]
# #### Define promoters

# %%
genes = pd.read_csv(folder_data_preproc / "genome" / "genes.csv", index_col = 0)

# %%
promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

# %%
all_gene_ids = genes.index

# %%
promoters = pd.DataFrame(index = all_gene_ids)

# %%
promoters["tss"] = [genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"] for _, genes_row in genes.loc[promoters.index].iterrows()]
promoters["strand"] = genes["strand"]
promoters["positive_strand"] = (promoters["strand"] == 1).astype(int)
promoters["negative_strand"] = (promoters["strand"] == -1).astype(int)
promoters["chr"] = genes.loc[promoters.index, "chr"]
promoters["symbol"] = genes.loc[promoters.index, "symbol"]

# %%
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")

# %%
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %% [markdown]
# #### Create fragments

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
import pathlib
import chromatinhd.data
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = pd.read_table("obs.tsv", sep = "\t", index_col = 0)
obs["ix"] = np.arange(len(obs))

# %%
cell_to_cell_ix = dict(zip(obs.index, obs["ix"]))

# %%
fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])

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
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int32))
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int32)

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
fragments.create_cellxgene_indptr()

# %% [markdown]
# ## Fragment distribution

# %%
import gzip

# %%
sizes = []
with gzip.GzipFile(folder_data_preproc / "fragments.tsv.gz", "r") as fragment_file:
    i = 0
    for line in fragment_file:
        line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        split = line.split("\t")
        sizes.append(int(split[2]) - int(split[1]))
        i += 1
        if i > 1000000:
            break

# %%
sizes = np.array(sizes)

# %%
np.isnan(sizes).sum()

# %%
fig, ax = plt.subplots()
ax.hist(sizes, range = (0, 1000), bins = 100)
ax.set_xlim(0, 1000)

# %%

# %%
