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

# %%
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

#export LD_LIBRARY_PATH=/data/peak_free_atac/software/peak_free_atac/lib
import torch
import torch_sparse

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import tempfile

# %% [markdown]
# ## Simulation

# %%
from simulate import Simulation

# %%
simulation = Simulation(n_cells = 1000, n_genes = 30)

# %%
window = simulation.window

# %%
coordinates, mapping, cell_latent_space= torch.from_numpy(simulation.coordinates).to(torch.float32), torch.from_numpy(simulation.mapping), torch.from_numpy(simulation.cell_latent_space).to(torch.float32)

# %%
cellmapping = mapping[:, 0]
genemapping = mapping[:, 1] 

# %%
## Plot prior distribution
gene_ix = 0
fig, (ax) = plt.subplots(1, 1, figsize=(40, 2))
for dim in range(cell_latent_space.shape[1]):
    fragments_oi = (cell_latent_space[:, dim] > 0)[cellmapping] & (genemapping == gene_ix)
    ax.hist(coordinates.cpu().numpy()[fragments_oi, 0], bins=1000, range = window, histtype = "step")

# %%
# library sizes
sns.histplot(torch.bincount(cellmapping, minlength = simulation.n_cells).numpy())

# %%
# gene means
sns.histplot(torch.bincount(genemapping, minlength = simulation.n_genes).numpy())

# %% [markdown]
# Create fragments

# %%
import peakfreeatac as pfa
import pathlib
import tempfile

# %%
fragments = pfa.data.Fragments(path = pathlib.Path(tempfile.TemporaryDirectory().name))

# %%
# need to make sure the order of fragments is cellxgene
order = np.argsort((simulation.mapping[:, 0] * simulation.n_genes) + simulation.mapping[:, 1])

fragments.coordinates = torch.from_numpy(simulation.coordinates)[order]
fragments.mapping = torch.from_numpy(simulation.mapping)[order]
fragments.var = pd.DataFrame({"gene":np.arange(simulation.n_genes)}).set_index("gene")
fragments.obs = pd.DataFrame({"cell":np.arange(simulation.n_cells)}).set_index("cell")
fragments.create_cellxgene_indptr()

# %% [markdown]
# ## Single gene real

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments_original = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
# one gene
gene_id = "PAX5"
gene_ix = fragments_original.var.loc[transcriptome.gene_id(gene_id)]["ix"]

fragments = pfa.data.Fragments(path = pathlib.Path(tempfile.TemporaryDirectory().name))

fragments_oi = fragments_original.genemapping == gene_ix

fragments.coordinates = fragments_original.coordinates[fragments_oi]
fragments.mapping = fragments_original.mapping[fragments_oi]
fragments.mapping[:, 1] = 0
fragments.var = pd.DataFrame({"gene":[1]}).set_index("gene")
fragments.obs = pd.DataFrame({"cell":np.arange(fragments_original.n_cells)}).set_index("cell")
fragments.create_cellxgene_indptr()

# %% [markdown]
# ## Full real

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %% [markdown]
# ## Create loaders

# %%
cells_train = np.arange(0, int(fragments.n_cells * 9 / 10))
cells_validation = np.arange(int(fragments.n_cells * 9 / 10), fragments.n_cells)

# %%
import peakfreeatac.loaders.fragments
n_cells_step = 1000
n_genes_step = 1000

# n_cells_step = 2000
# n_genes_step = 500

loaders_train = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window},
    n_workers = 20,
    shuffle_on_iter = True
)
minibatches_train = pfa.loaders.minibatching.create_bins_random(cells_train, np.arange(fragments.n_genes), fragments.n_genes, n_genes_step = n_genes_step, n_cells_step=n_cells_step, use_all = True)
loaders_train.initialize(minibatches_train)

loaders_validation = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window},
    n_workers = 2
)
minibatches_validation = pfa.loaders.minibatching.create_bins_random(cells_validation, np.arange(fragments.n_genes), fragments.n_genes, n_genes_step = n_genes_step, n_cells_step=n_cells_step, use_all = True)
loaders_validation.initialize(minibatches_validation)

# %% [markdown]
# ## Given latent space

# %%
# latent = torch.from_numpy(simulation.cell_latent_space).to(torch.float32)

# latent = torch.ones((fragments.obs.shape[0], 1)).to(torch.float32)
# latent.data.uniform_(1-1e-1, 1+1e-1)

sc.tl.leiden(transcriptome.adata)
latent = torch.from_numpy(pd.get_dummies(transcriptome.adata.obs["leiden"]).values).to(torch.float)

n_latent_dimensions = latent.shape[-1]

# %%
fragments.window = window

# %%
# import peakfreeatac.models.vae.v3 as vae_model
# model = vae_model.Decoding(fragments, latent, n_bins = 50)

import peakfreeatac.models.vae.v2 as vae_model
model = vae_model.Decoding(fragments, latent, n_components = 64)

# %% [markdown]
# Test

# %%
device = "cuda"
model = model.to(device)


# %%
def evaluate_pseudo(pseudocoordinates, pseudocell_latent_space, gene_ix):
    # for each cell, gene and locus we have a change
    genes_oi = torch.tensor([gene_ix])
    logit_change, *_ = decoder(pseudocell_latent_space, genes_oi)
    
    logit_change_cellxgene = logit_change.view(np.prod(logit_change.shape[:2]), logit_change.shape[-1])
    
    local_gene_ix = torch.zeros(len(pseudocoordinates), dtype = torch.long)
    local_cell_ix = torch.zeros(len(pseudocoordinates), dtype = torch.long)
    local_cellxgene_ix = local_cell_ix * len(genes_oi) + local_gene_ix
    prob = torch.exp(mixture.log_prob(pseudocoordinates, logit_change_cellxgene[local_cellxgene_ix], genes_oi, local_gene_ix).detach())
    return prob



# %% jp-MarkdownHeadingCollapsed=true
gene_ix = 0
ndim = latent.shape[1] + 1

## Plot prior distribution
fig, axes = plt.subplots(ndim, 1, figsize=(40, 2 * ndim))
fragments_oi = (fragments.genemapping == gene_ix)
axes[0].hist(fragments.coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False, color = "#AAA")
for ax, dim in zip(axes[1:], range(ndim-1)):
    fragments_oi = (latent[:, dim] > 0.8)[fragments.cellmapping] & (fragments.genemapping == gene_ix)
    ax.hist(fragments.coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False)

    pseudocoordinates = torch.arange(*window).to(device)
    pseudolatent = torch.zeros((pseudocoordinates.shape[0], latent.shape[1])).to(device)
    pseudolatent[:, dim] = 1.
    
    prob = model.evaluate_pseudo(pseudocoordinates, pseudolatent, gene_ix).detach().cpu().numpy()
    ax.plot(pseudocoordinates.detach().cpu().numpy(), np.exp(prob) / (window[1] - window[0]), lw = 2)
# ax1.set_ylim(0, 10.)

# %% [markdown]
# ### Train

# %%
optimizer = pfa.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-3)

# %%
loaders_train.restart()
loaders_validation.restart()
import gc
gc.collect()

# %%
model = model.to(device).train()
trainer = pfa.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 500, checkpoint_every_epoch=20, optimize_every_step = 1)
trainer.train()

# %%
device = "cuda"

# %%
model = model.to(device).eval()

# %% jp-MarkdownHeadingCollapsed=true
gene_ix = 5
ndim = latent.shape[1] + 1

## Plot prior distribution
fig, axes = plt.subplots(ndim, 1, figsize=(40, 2 * ndim))
fragments_oi = (fragments.genemapping == gene_ix)
axes[0].hist(fragments.coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False, color = "#AAA")
for ax, dim in zip(axes[1:], range(ndim-1)):
    fragments_oi = (latent[:, dim] > 0.8)[fragments.cellmapping] & (fragments.genemapping == gene_ix)
    ax.hist(fragments.coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False)

    pseudocoordinates = torch.arange(*window).to(device)
    pseudolatent = torch.zeros((pseudocoordinates.shape[0], latent.shape[1])).to(device)
    pseudolatent[:, dim] = 1.
    
    prob = model.evaluate_pseudo(pseudocoordinates, pseudolatent, gene_ix).detach().cpu().numpy()
    ax.plot(pseudocoordinates.detach().cpu().numpy(), np.exp(prob) / (window[1] - window[0]), lw = 2)
# ax1.set_ylim(0, 10.)

# %% [markdown]
# ## VAE

# %%
device = "cuda"

# %% [markdown]
# Train

# %%
fragments.window = window

# %%
# import peakfreeatac.models.vae.v3 as vae_model
# model = vae_model.VAE(fragments, n_bins = 50)

# import peakfreeatac.models.vae.v2 as vae_model
# model = vae_model.VAE(fragments, n_components=64, n_frequencies = 40, n_latent_dimensions = 10)

model = pickle.load(open("model.pkl", "rb"))

# %%
optimizer = pfa.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-3, weight_decay = 1e-5)

# %%
model

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
loaders_train.shuffle_on_iter = True

# %%
loaders_train.restart()
loaders_validation.restart()

model = model.train()
trainer = pfa.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 200, checkpoint_every_epoch=20, optimize_every_step = 1)
trainer.train()

# %%
trainer.trace.plot()

# %%
model = model.to("cpu")
pickle.dump(model, open("model.pkl", "wb"))

# %%
import itertools

# %%
model = model.to(device).eval()

# %%
embedding = np.zeros((fragments.n_cells, model.n_latent_dimensions))

loaders_train.restart()
loaders_validation.restart()
for data in loaders_train:
    data = data.to(device)
    with torch.no_grad():   
        embedding[data.cells_oi] = model.evaluate_latent(data).detach().cpu().numpy()
    
    loaders_train.submit_next()
for data in loaders_validation:
    data = data.to(device)
    with torch.no_grad():
        embedding[data.cells_oi] = model.evaluate_latent(data).detach().cpu().numpy()
    
    loaders_validation.submit_next()

# %%
[(fragments.n_cells * fragments.n_genes) / (len(mb.cells_oi) * len(mb.genes_oi)) for mb in minibatches_train]

# %%
len(minibatches_train)

# %%
# cell_order = np.argsort(simulation.cell_latent_space[:, 5])
cell_order = np.arange(fragments.obs.shape[0])
sns.heatmap(embedding[cell_order], vmin = -1, vmax = 1)

# %%
adata = sc.AnnData(X = transcriptome.adata.X, var = transcriptome.adata.var, obs = fragments.obs)

# %%
adata.obsm["latent"] = embedding

# %%
sc.pp.neighbors(adata, use_rep = "latent")
sc.tl.umap(adata)

# %%
adata.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
# cell_latent_space_pd = pd.DataFrame(simulation.cell_latent_space, columns = "latent_gs_" + pd.Series(np.arange(simulation.n_cell_components)).astype("str"), index = adata.obs.index)
# adata.obs[cell_latent_space_pd.columns] = cell_latent_space_pd

# %%
# obs_mb = pd.concat([pd.DataFrame({"ix":mb.cells_oi.tolist(), "mb":i})  for i, mb in enumerate(minibatches_train)])
# obs_mb.index = adata.obs.index[obs_mb["ix"]]
# adata.obs["minibatch"] = obs_mb["mb"]

# %%
sc.pl.umap(adata, color = ["n_fragments"])

# %%
# sc.pl.umap(adata, color = [*cell_latent_space_pd.columns])

# %%
adata_transcriptome = transcriptome.adata

# %%
sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome)

# %%
sc.pl.umap(adata_transcriptome, color = ["leiden"])

# %%
adata.obs["leiden_transcriptome"] = adata_transcriptome.obs["leiden"]

# %%
sc.pl.umap(adata, color = ["leiden_transcriptome"])

# %%
adata_atac = adata

# %%
# sc.pp.neighbors(adata_transcriptome, n_neighbors=50, key_added = "100")
# sc.pp.neighbors(adata_atac, n_neighbors=50, key_added = "100", use_rep="latent")

sc.pp.neighbors(adata_transcriptome, n_neighbors=100, key_added = "100")
sc.pp.neighbors(adata_atac, n_neighbors=100, key_added = "100", use_rep="latent")

# %%
assert (adata_transcriptome.obs.index == adata_atac.obs.index).all()

# %%
A = np.array(adata_transcriptome.obsp["100_connectivities"].todense() != 0)
B = np.array(adata_atac.obsp["100_connectivities"].todense() != 0)

# %%
intersect = A * B
union = (A+B) != 0

# %%
ab = intersect.sum() / union.sum()
ab

# %%
C = A[np.random.choice(A.shape[0], A.shape[0], replace = False)]
# C = B[np.random.choice(B.shape[0], B.shape[0], replace = False)]

# %%
intersect = C * B
union = (C+B) != 0

# %%
intersect.sum()

# %%
union.sum()

# %%
ac = intersect.sum() / union.sum()
ac

# %%
ab/ac

# %%
gene_ix = 3
gene_ix = fragments.var.loc[transcriptome.gene_id("PAX5")]["ix"]
ndim = model.n_latent_dimensions

## Plot prior distribution
fig, axes = plt.subplots(ndim, 1, figsize=(40, 2 * ndim))
fragments_oi = (fragments.genemapping == gene_ix)
axes[0].hist(fragments.coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False, color = "#AAA")
for ax, dim in zip(axes[1:], range(ndim-1)):
    # fragments_oi = (latent[:, dim] > 0.8)[fragments.cellmapping] & (fragments.genemapping == gene_ix)
    # ax.hist(fragments.coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False)

    pseudocoordinates = torch.arange(*window).to(device)
    pseudolatent = torch.zeros((pseudocoordinates.shape[0], model.n_latent_dimensions)).to(device)
    pseudolatent[:, dim] = 1.
    
    prob = model.evaluate_pseudo(pseudocoordinates, pseudolatent, gene_ix).detach().cpu().numpy()
    ax.plot(pseudocoordinates.detach().cpu().numpy(), np.exp(prob) / (window[1] - window[0]), lw = 2)
# ax1.set_ylim(0, 10.)

# %%
sc.pl.umap(adata_transcriptome, color = transcriptome.gene_id("PAX5"))
sc.pl.umap(adata_atac, color = transcriptome.gene_id("PAX5"))

# %%
import sklearn

# %%
import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(adata.obsm["latent"][cells_train], adata.obs["leiden_transcriptome"].astype(int)[cells_train])
prediction = classifier.predict(adata.obsm["latent"][cells_validation])
sklearn.metrics.balanced_accuracy_score(adata_atac.obs["leiden_transcriptome"].astype(int)[cells_validation], prediction)

# %%
