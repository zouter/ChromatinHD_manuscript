# %%
import glob
import torch
import tabix
import pickle
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
dataset_name_sub = "MV2"
folder_data_preproc = folder_data / dataset_name

#%%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

# %%
fragments_tabix = tabix.open(next(iter(glob.glob(str(folder_data_preproc / dataset_name_sub / "*atac_fragments.tsv.gz"))), None))

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")

var = pd.DataFrame(index = promoters.index)
n_genes = var.shape[0]
var["ix"] = np.arange(n_genes)

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])
n_cells = obs.shape[0]
cell_to_cell_ix = obs["ix"].to_dict()

#%%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    query = f"{promoter_info['chr']}:{promoter_info['start']}-{promoter_info['end']}"
    fragments_promoter = fragments_tabix.querys(query)
    
    gene_ix = var.loc[gene, "ix"]
    for fragment in fragments_promoter:
        # only store the fragment if the cell is actually of interest
        cell = fragment[3]
        if cell in cell_to_cell_ix:

            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([cell_to_cell_ix[cell], gene_ix])

# Create fragments tensor
coordinates = torch.tensor(np.array(coordinates_raw), dtype = torch.int64)
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# Sort `coordinates` and `mapping` according to `mapping`
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

#%%
np.random.seed(410)
folds = []
for i in range(5):
    # randomly select 20% of cells
    cells_validation = np.random.choice(np.arange(n_cells), size=int(n_cells * 0.2), replace=False)
    # get the remaining cells
    cells_train = np.setdiff1d(np.arange(n_cells), cells_validation)
    # append to folds
    folds.append({'cells_train': cells_train, 'cells_validation': cells_validation})

# %% 
# create fragments dir
fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments" / promoter_name)
fragments.var = var
fragments.obs = obs
fragments.mapping = mapping
fragments.coordinates = coordinates
fragments.create_cellxgene_indptr()
pickle.dump(folds, open(fragments.path / 'folds.pkl', 'wb'))

# %%
# create fragments dir for myeloid lineage
myeloid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_myeloid.csv", index_col = 0)

obs = transcriptome.adata[myeloid.index].obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])
n_cells = obs.shape[0]
cell_to_cell_ix = obs["ix"].to_dict()

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    query = f"{promoter_info['chr']}:{promoter_info['start']}-{promoter_info['end']}"
    fragments_promoter = fragments_tabix.querys(query)
    
    gene_ix = var.loc[gene, "ix"]
    for fragment in fragments_promoter:
        # only store the fragment if the cell is actually of interest
        cell = fragment[3]
        if cell in cell_to_cell_ix:

            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([cell_to_cell_ix[cell], gene_ix])

# Create fragments tensor
coordinates = torch.tensor(np.array(coordinates_raw), dtype = torch.int64)
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# Sort `coordinates` and `mapping` according to `mapping`
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

#%%
np.random.seed(410)
folds = []
for i in range(5):
    # randomly select 20% of cells
    cells_validation = np.random.choice(np.arange(n_cells), size=int(n_cells * 0.2), replace=False)
    # get the remaining cells
    cells_train = np.setdiff1d(np.arange(n_cells), cells_validation)
    # append to folds
    folds.append({'cells_train': cells_train, 'cells_validation': cells_validation})


fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_myeloid" / promoter_name)
fragments.var = var
fragments.obs = obs
fragments.mapping = mapping
fragments.coordinates = coordinates
fragments.create_cellxgene_indptr()
pickle.dump(folds, open(fragments.path / 'folds.pkl', 'wb'))

# %%
# create fragments dir for erythroid lineage
erythroid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_erythroid.csv", index_col = 0)

obs = transcriptome.adata[erythroid.index].obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])
n_cells = obs.shape[0]
cell_to_cell_ix = obs["ix"].to_dict()

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    query = f"{promoter_info['chr']}:{promoter_info['start']}-{promoter_info['end']}"
    fragments_promoter = fragments_tabix.querys(query)
    
    gene_ix = var.loc[gene, "ix"]
    for fragment in fragments_promoter:
        # only store the fragment if the cell is actually of interest
        cell = fragment[3]
        if cell in cell_to_cell_ix:

            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([cell_to_cell_ix[cell], gene_ix])

# Create fragments tensor
coordinates = torch.tensor(np.array(coordinates_raw), dtype = torch.int64)
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# Sort `coordinates` and `mapping` according to `mapping`
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

#%%
np.random.seed(410)
folds = []
for i in range(5):
    # randomly select 20% of cells
    cells_validation = np.random.choice(np.arange(n_cells), size=int(n_cells * 0.2), replace=False)
    # get the remaining cells
    cells_train = np.setdiff1d(np.arange(n_cells), cells_validation)
    # append to folds
    folds.append({'cells_train': cells_train, 'cells_validation': cells_validation})


fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_erythroid" / promoter_name)
fragments.var = var
fragments.obs = obs
fragments.mapping = mapping
fragments.coordinates = coordinates
fragments.create_cellxgene_indptr()
pickle.dump(folds, open(fragments.path / 'folds.pkl', 'wb'))

# %%
# create fragments dir for platelet lineage
platelet = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_platelet.csv", index_col = 0)

obs = transcriptome.adata[platelet.index].obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])
n_cells = obs.shape[0]
cell_to_cell_ix = obs["ix"].to_dict()

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    query = f"{promoter_info['chr']}:{promoter_info['start']}-{promoter_info['end']}"
    fragments_promoter = fragments_tabix.querys(query)
    
    gene_ix = var.loc[gene, "ix"]
    for fragment in fragments_promoter:
        # only store the fragment if the cell is actually of interest
        cell = fragment[3]
        if cell in cell_to_cell_ix:

            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([cell_to_cell_ix[cell], gene_ix])

# Create fragments tensor
coordinates = torch.tensor(np.array(coordinates_raw), dtype = torch.int64)
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# Sort `coordinates` and `mapping` according to `mapping`
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

#%%
np.random.seed(410)
folds = []
for i in range(5):
    # randomly select 20% of cells
    cells_validation = np.random.choice(np.arange(n_cells), size=int(n_cells * 0.2), replace=False)
    # get the remaining cells
    cells_train = np.setdiff1d(np.arange(n_cells), cells_validation)
    # append to folds
    folds.append({'cells_train': cells_train, 'cells_validation': cells_validation})


fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_platelet" / promoter_name)
fragments.var = var
fragments.obs = obs
fragments.mapping = mapping
fragments.coordinates = coordinates
fragments.create_cellxgene_indptr()
pickle.dump(folds, open(fragments.path / 'folds.pkl', 'wb'))

# %%