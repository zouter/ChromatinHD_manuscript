# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests
import xarray as xr

# %%
fragment_dataset_name = "pbmc10k_leiden_0.1"
dataset_name = fragment_dataset_name + "_gwas"

# %%
genotype_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %%
dataset_folder = chd.get_output() / "datasets" / "genotyped" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Predictive model

# %% [markdown]
# ### Load direct model

# %%
variantxgene_effect = chd.load(pathlib.Path("variantxgene_effect.pkl").open("rb"))
model_direct = chd.load(pathlib.Path("model_direct.pkl").open("rb"))

# %% [markdown]
# ### Data

# %%
fragments = chd.data.fragments.ChunkedFragments(dataset_folder / "fragments")
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(dataset_folder / "transcriptome")
genotype = chd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
gene_variants_mapping = pickle.load((dataset_folder / "gene_variants_mapping.pkl").open("rb"))

# %%
import chromatinhd.models.eqtl.prediction.v2 as prediction_model

# %%
loader = prediction_model.Loader(transcriptome, genotype, fragments, gene_variants_mapping)
loaders = chd.loaders.LoaderPool(
    prediction_model.Loader,
    loader = loader,
    n_workers=10,
    shuffle_on_iter=True,
)
loaders_validation = chd.loaders.LoaderPool(
    prediction_model.Loader,
    loader = loader,
    n_workers=5,
    shuffle_on_iter=False,
)

# %%
all_genes = transcriptome.var.query("chr != 'chr6'")
train_genes = train_genes = all_genes.query("chr != 'chr1'")
validation_genes = all_genes.query("chr == 'chr1'")

# %%
minibatches_train = prediction_model.loader.create_bins_ordered(train_genes["ix"].values, n_genes_step = 300)
len(minibatches_train)

minibatches_validation = prediction_model.loader.create_bins_ordered(validation_genes["ix"].values, n_genes_step = 300)
len(minibatches_train), len(minibatches_validation)

# %% [markdown]
# #### Test loader

# %%
import torch

# %%
# fragments.chunkcoords = fragments.chunkcoords.to(torch.int64)

# %%
loader = prediction_model.Loader(transcriptome, genotype, fragments, gene_variants_mapping)

# %%
import copy

# %%
genes_oi = np.array(np.arange(100).tolist() + transcriptome.var.loc[transcriptome.gene_id(["CTLA4", "BACH2", "BLK"]), "ix"].tolist())
# genes_oi = np.array([transcriptome.var.loc[transcriptome.gene_id("CTLA4"), "ix"]])

minibatch = prediction_model.Minibatch(genes_oi)

data = loader.load(minibatch)

# %% [markdown]
# ### Model

# %%
model_pred = prediction_model.Model.create(
    transcriptome,
    genotype,
    fragments,
    gene_variants_mapping,
    variantxgene_effect = variantxgene_effect,
    reference_expression_predictor = model_direct.expression_predictor
)

# %%
model_pred.parameters_dense()

# %%
model_pred = model_pred.to("cpu")
model_pred.forward(data)

# %% [markdown]
# ### Train

# %%
loaders.initialize(minibatches_train)
loaders_validation.initialize(minibatches_validation)

# %%
optim = chd.optim.SparseDenseAdam(model_pred.parameters_sparse(), model_pred.parameters_dense(), lr = 1e-2)
trainer = prediction_model.Trainer(model_pred, loaders, loaders_validation, optim, checkpoint_every_epoch=5, n_epochs = 20)
trainer.train()

# %%
# chd.save(model, pathlib.Path("model_direct.pkl").open("wb"))
# chd.save(model_dummy, pathlib.Path("model_dummy.pkl").open("wb"))

# %%
bias = model_pred.fc_log_predictor.nn[0].bias[0].item()
weight = model_pred.fc_log_predictor.nn[0].weight[0, 0].item()

# %%
x = torch.linspace(-5., 5., 100)
y = torch.sigmoid(x * weight + bias)
fig, ax = plt.subplots()
ax.plot(x, y)
