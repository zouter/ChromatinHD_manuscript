# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess

# %%
import polyptich as pp
pp.setup_ipython()

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

# train_dataset_name = "pbmc10k"; test_dataset_name = "pbmc3k"; organism = "hs"
# train_dataset_name = "pbmc10k"; test_dataset_name = "lymphoma"; organism = "hs"
train_dataset_name = "pbmc10k"; test_dataset_name = "pbmc10k_gran"; organism = "hs"
train_dataset_name = "pbmc10k"; test_dataset_name = "pbmc10kx"; organism = "hs"

dataset_name = test_dataset_name + "-" + train_dataset_name

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
train_folder_data_preproc = folder_data / train_dataset_name
test_folder_data_preproc = folder_data / test_dataset_name

train_dataset_folder = chd.get_output() / "datasets" / train_dataset_name
test_dataset_folder = chd.get_output() / "datasets" / test_dataset_name

# %%
# !ln -s 

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
train_transcriptome = chd.data.Transcriptome(train_dataset_folder / "transcriptome")

# %% [markdown]
# ### Create transcriptome

# %%
adata = pickle.load((test_folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
adata2 = adata.copy()
missing_genes = ~train_transcriptome.var.index.isin(adata.var.index)

# %%
var2 = pd.concat([adata2.var, train_transcriptome.var.loc[missing_genes]])

# %%
layers = {}
for layer in adata.layers:
    X = adata.layers[layer]
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    layers[layer] = np.concatenate([X, np.zeros((adata.shape[0], missing_genes.sum()))], axis=1)

# %%
adata2 = sc.AnnData(
    X = layers["normalized"],
    obs = adata.obs,
    var = var2,
    layers = layers
)

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata2[:, train_transcriptome.var.index], path=dataset_folder / "transcriptome")

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((train_folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
fragments_file = test_folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")

fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    # overwrite = True
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load((train_folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
fragments_file = test_folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite = True,
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Peaks

# %%
peaks_folder = chd.get_output() / "peaks" / dataset_name
if peaks_folder.exists():
    if not peaks_folder.is_symlink():
        import shutil
        shutil.rmtree(peaks_folder)

    train_peaks_folder = chd.get_output() / "peaks" / train_dataset_name
    peaks_folder = chd.get_output() / "peaks" / dataset_name
    # symlink peaks folder to point to train_peaks_folder
    peaks_folder.symlink_to(train_peaks_folder)


# %%

# %%

# %% [markdown]
# ## Softlink fragments

# %%
# !ln -s {test_folder_data_preproc}/atac_fragments.tsv.gz {folder_data_preproc}
# !ln -s {test_folder_data_preproc}/atac_fragments.tsv.gz.tbi {folder_data_preproc}

# %%
# !ls {test_folder_data_preproc}
