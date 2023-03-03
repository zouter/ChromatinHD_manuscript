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

# %%
fragment_dataset_name = "pbmc10k_leiden_0.1"
dataset_name = fragment_dataset_name + "_gwas"

# %%
genotype_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %%
dataset_folder = chd.get_output() / "datasets" / "genotyped" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Direct eQTL model

# %% [markdown]
# ### Data

# %%
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(dataset_folder / "transcriptome")
genotype = chd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
gene_variants_mapping = pickle.load((dataset_folder / "gene_variants_mapping.pkl").open("rb"))

# %%
import chromatinhd.models.eqtl.mapping.v2 as eqtl_model

# %%
loaders = chd.loaders.LoaderPool(
    eqtl_model.Loader,
    {"transcriptome":transcriptome, "genotype":genotype, "gene_variants_mapping":gene_variants_mapping},
    n_workers=10,
    shuffle_on_iter=False,
)

# %%
minibatches = eqtl_model.loader.create_bins_ordered(transcriptome.var["ix"].values)

# %% [markdown]
# ### Model

# %%
model = eqtl_model.Model.create(transcriptome, genotype, gene_variants_mapping)
model_dummy = eqtl_model.Model.create(transcriptome, genotype, gene_variants_mapping, dummy = True)

# %% [markdown]
# ### Test loader

# %%
genes_oi = np.array(np.arange(1000).tolist() + transcriptome.var.loc[transcriptome.gene_id(["CTLA4", "BACH2", "BLK"]), "ix"].tolist())
# genes_oi = np.array([transcriptome.var.loc[transcriptome.gene_id("CTLA4"), "ix"]])

minibatch = eqtl_model.Minibatch(genes_oi)

data = loader.load(minibatch)

# %% [markdown]
# ### Train

# %%
loaders.initialize(minibatches)

# %%
optim = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-2)
trainer = eqtl_model.Trainer(model, loaders, optim, checkpoint_every_epoch=50, n_epochs = 300)
trainer.train()

# %%
optim = chd.optim.SparseDenseAdam(model_dummy.parameters_sparse(), model_dummy.parameters_dense(), lr = 1e-2)
trainer = eqtl_model.Trainer(model_dummy, loaders, optim, checkpoint_every_epoch=50, n_epochs = 300)
trainer.train()

# %%
chd.save(model, pathlib.Path("model_direct.pkl").open("wb"))
chd.save(model_dummy, pathlib.Path("model_dummy.pkl").open("wb"))

# %% [markdown]
# ### Inference & interpretion

# %%
model = chd.load(pathlib.Path("model_direct.pkl").open("rb"))
model_dummy = chd.load(pathlib.Path("model_dummy.pkl").open("rb"))

# %%
minibatches_inference = minibatches

# %%
loaders_inference = chd.loaders.LoaderPool(
    eqtl_model.Loader,
    {"transcriptome":transcriptome, "genotype":genotype, "gene_variants_mapping":gene_variants_mapping},
    n_workers=5,
    shuffle_on_iter=False,
)

# %%
variantxgene_index = []
for gene, gene_ix in zip(transcriptome.var.index, transcriptome.var["ix"]):
    variantxgene_index.extend([[gene, genotype.variants_info.index[variant_ix]] for variant_ix in gene_variants_mapping[gene_ix]])
variantxgene_index = pd.MultiIndex.from_frame(pd.DataFrame(variantxgene_index, columns = ["gene", "variant"]))

# %%
device = "cpu"

# %%
loaders_inference.initialize(minibatches_inference)
elbo = np.zeros((len(transcriptome.clusters_info), len(variantxgene_index)))
elbo_dummy = np.zeros((len(transcriptome.clusters_info), len(variantxgene_index)))

model = model.to(device)
model_dummy = model_dummy.to(device)
for data in tqdm.tqdm(loaders_inference):
    data = data.to(device)
    model.forward(data)
    elbo_mb = model.get_full_elbo().sum(0).detach().cpu().numpy()
    elbo[:, data.variantxgene_ixs.cpu().numpy()] += elbo_mb
    
    model_dummy.forward(data)
    elbo_mb = model_dummy.get_full_elbo().sum(0).detach().cpu().numpy()
    elbo_dummy[:, data.variantxgene_ixs.cpu().numpy()] += elbo_mb
    
    loaders_inference.submit_next()
    
bf = xr.DataArray(elbo_dummy - elbo, coords = [transcriptome.clusters_info.index, variantxgene_index])

# %%
import xarray as xr

# %%
fc_log_mu = xr.DataArray(model.fc_log_predictor.variantxgene_cluster_effect.weight.detach().cpu().numpy().T, coords = [transcriptome.clusters_info.index, variantxgene_index])

# %%
scores = fc_log_mu.to_pandas().T.stack().to_frame("fc_log")

# %%
scores["bf"] = bf.to_pandas().T.stack()

# %%
scores.sort_values("bf")

# %%
scores.query("cluster == 'pDCs'").sort_values("bf").join(transcriptome.var[["symbol"]])

# %%
scores["significant"] = scores["bf"] > np.log(10)

# %%
scores.groupby("gene")["significant"].sum().sort_values(ascending = False).to_frame().join(transcriptome.var[["symbol", "chr"]]).head(50)

# %%
scores.groupby("cluster")["bf"].sum().to_frame("bf").style.bar()

# %%
variant_id = genotype.variants_info.query("rsid == 'rs3087243'").index[0]

# %%
scores.join(genotype.variants_info[["rsid"]]).xs(variant_id, level = "variant").sort_values("fc_log")

# %% [markdown]
# ### Get reference variantxgene effect

# %%
scores["abs_fc_log"] = np.abs(scores["fc_log"])

# %%
variantxgene_effect_sorted = scores.query("significant").sort_values("abs_fc_log", ascending = False).groupby(["gene", "variant"]).first()["fc_log"]
variantxgene_effect_max = scores.sort_values("bf", ascending = False).groupby(["gene", "variant"])["fc_log"].first()

# %%
variantxgene_effect = pd.Series(np.nan, variantxgene_index)
variantxgene_effect[variantxgene_effect_sorted.index] = variantxgene_effect_sorted

missing_reference_fc = variantxgene_effect.index[np.isnan(variantxgene_effect)]
variantxgene_effect[missing_reference_fc] = variantxgene_effect_max[missing_reference_fc]

# %% [markdown]
# ## Predictive model

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
minibatches_train = eqtl_model.loader.create_bins_ordered(train_genes["ix"].values, n_genes_step = 300)
len(minibatches_train)

minibatches_validation = eqtl_model.loader.create_bins_ordered(validation_genes["ix"].values, n_genes_step = 300)
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

minibatch = eqtl_model.Minibatch(genes_oi)

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
    reference_expression_predictor = model.expression_predictor
)

# %%
model_pred.parameters_dense()

# %%
model_pred.forward(data)

# %% [markdown]
# ### Train

# %%
loaders.initialize(minibatches_train)
loaders_validation.initialize(minibatches_validation)

# %%
optim = chd.optim.SparseDenseAdam(model_pred.parameters_sparse(), model_pred.parameters_dense(), lr = 1e-2)
trainer = prediction_model.Trainer(model_pred, loaders, loaders_validation, optim, checkpoint_every_epoch=5, n_epochs = 300)
trainer.train()

# %%
optim = chd.optim.SparseDenseAdam(model_dummy.parameters_sparse(), model_dummy.parameters_dense(), lr = 1e-2)
trainer = eqtl_model.Trainer(model_dummy, loaders, optim, checkpoint_every_epoch=50, n_epochs = 300)
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
