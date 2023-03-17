# %%
import gzip
import torch
import tabix
import pickle
import pathlib
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import tqdm.auto as tqdm
import chromatinhd as chd
import chromatinhd.data

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
sc._settings.ScanpyConfig.figdir = pathlib.PosixPath('')

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

#%%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)

#%%
fragments_tabix = tabix.open(str(folder_data_preproc / "GSM6403411_3423-MV-2_atac_fragments.tsv.gz"))

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
myeloid = pd.read_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv", index_col = 0)

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
n_genes = var.shape[0]
var["ix"] = np.arange(n_genes)

# %%
obs = transcriptome.adata[myeloid.index].obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])
n_cells = obs.shape[0]

#%%
if "cell_original" in transcriptome.obs.columns:
    cell_ix_to_cell = transcriptome.obs["cell_original"].explode()
    cell_to_cell_ix = pd.Series(cell_ix_to_cell.index.astype(int), cell_ix_to_cell.values)
else:
    cell_to_cell_ix = obs["ix"].to_dict()

# %%
gene_to_fragments = [[] for i in var["ix"]]
cell_to_fragments = [[] for i in obs["ix"]]

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    gene_ix = var.loc[gene, "ix"]
    query = f"{promoter_info['chr']}:{promoter_info['start']}-{promoter_info['end']}"
    fragments_promoter = fragments_tabix.querys(query)
    
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

# %%
# Create fragments tensor
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int64))
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# %%
# Sort `coordinates` and `mapping` according to `mapping`
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %% 
# Check size
# np.product(mapping.size()) * 64 / 8 / 1024 / 1024
# np.product(coordinates.size()) * 64 / 8 / 1024 / 1024

# %% 
# Store
fragments.mapping = mapping
fragments.coordinates = coordinates

# %%
# Create cellxgene index pointers
fragments.create_cellxgene_indptr()

# %%
#Create training folds
n_bins = 5

# %%
# train/test split
# transcriptome.var.index.name = "symbol"
# transcriptome.var = transcriptome.var.reset_index()
transcriptome.var.index = transcriptome.var["Accession"]
transcriptome.var.index.name = "gene"

#%%
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

cell_bins = np.floor((np.arange(len(cells_all))/(len(cells_all)/n_bins)))

chromosome_gene_counts = transcriptome.var.groupby("Chromosome").size().sort_values(ascending = False)
chromosome_bins = np.cumsum(((np.cumsum(chromosome_gene_counts) % (chromosome_gene_counts.sum() / n_bins + 1)).diff() < 0))

gene_bins = chromosome_bins[transcriptome.var["Chromosome"]].values

#%%
n_folds = 5
folds = []
for i in range(n_folds):
    cells_train = cells_all[cell_bins != i]
    cells_validation = cells_all[cell_bins == i]

    chromosomes_train = chromosome_bins.index[~(chromosome_bins == i)]
    chromosomes_validation = chromosome_bins.index[chromosome_bins == i]
    
    genes_index = set(transcriptome.var.index[transcriptome.var["Chromosome"].isin(chromosomes_train)]).intersection(set(promoters.index))
    genes_train = fragments.var["ix"][genes_index].values

    genes_index = set(transcriptome.var.index[transcriptome.var["Chromosome"].isin(chromosomes_validation)]).intersection(set(promoters.index))
    genes_validation = fragments.var["ix"][genes_index].values
    
    folds.append({
        "cells_train":cells_train,
        "cells_validation":cells_validation,
        "genes_train":genes_train,
        "genes_validation":genes_validation
    })
    
pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %%
