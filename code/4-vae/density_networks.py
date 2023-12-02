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
from simulate import Simulation

# %%
simulation = Simulation(n_cells = 500)

# %%
window = simulation.window

# %%
coordinates, mapping, cell_latent_space= torch.from_numpy(simulation.coordinates).to(torch.float32), torch.from_numpy(simulation.mapping), torch.from_numpy(simulation.cell_latent_space).to(torch.float32)

# %%
cellmapping = mapping[:, 0]
genemapping = mapping[:, 1] 

# %%
## Plot prior distribution
gene_ix = 5
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
# Create loaders

# %%
cells_train = np.arange(0, int(fragments.n_cells * 4 / 5))
cells_validation = np.arange(int(fragments.n_cells * 4 / 5), fragments.n_cells)

# %%
import peakfreeatac.loaders.fragments
n_cells_step = 1000
n_genes_step = 1000

loaders_train = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window},
    n_workers = 10
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
latent = torch.from_numpy(simulation.cell_latent_space).to(torch.float32)
n_latent_dimensions = latent.shape[-1]

# %% [markdown]
# Initialize

# %%
import peakfreeatac.models.vae.v2 as vae_model

# %%
n_layers = 2
n_components = 32

# %%
locs, logits = vae_model.initialize_mixture(window, coordinates, genemapping, fragments.n_genes, n_components = n_components)
mixture = vae_model.Mixture(n_components, fragments.n_genes, window, loc_init = locs, logit_init = logits, debug = True)
decoder = vae_model.Decoder(n_latent_dimensions, simulation.n_genes, mixture.n_components, n_layers = n_layers)

# %% [markdown]
# Test

# %%
gene_ix = 1
genes_oi = torch.tensor([gene_ix])

# %%
coordinates = fragments.coordinates
genemapping = fragments.genemapping
cellmapping = fragments.mapping[:, 0]

# %%
# for each cell, gene and locus we have a change
logit_change, rho_change = decoder(cell_latent_space, genes_oi)
logit_change.shape # cells, genes, loci

# %%
logit_change_cellxgene = logit_change.view(np.prod(logit_change.shape[:2]), logit_change.shape[-1])

# %%
pseudocoordinates = torch.arange(*window)
local_gene_ix = torch.zeros(len(pseudocoordinates), dtype = torch.long)
local_cell_ix = torch.zeros(len(pseudocoordinates), dtype = torch.long)
local_cellxgene_ix = local_cell_ix * len(genes_oi) + local_gene_ix
prob = torch.exp(mixture.log_prob(pseudocoordinates, torch.index_select(logit_change_cellxgene, 0, local_cellxgene_ix), genes_oi, local_gene_ix).detach())


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


# %%
gene_ix = 4

# %% jp-MarkdownHeadingCollapsed=true
## Plot prior distribution
fig, (ax, ax1) = plt.subplots(2, 1, figsize=(40, 4))
fragments_oi = (fragments.genemapping == gene_ix)
ax.hist(coordinates.cpu().numpy()[fragments_oi, 0], bins=1000, range = window, histtype = "stepfilled", density = True, log = False, color = "#AAA")
for dim in range(latent.shape[1]):
    fragments_oi = (cell_latent_space[:, dim] > 0)[fragments.cellmapping] & (fragments.genemapping == gene_ix)
    ax.hist(coordinates.cpu().numpy()[fragments_oi, 0], bins=1000, range = window, histtype = "step", density = True, log = False)

## Plot posterior distribution
for dim in range(latent.shape[1]):
    pseudocell_latent_space = torch.zeros((pseudocoordinates.shape[0], cell_latent_space.shape[1]))
    pseudocell_latent_space[:, dim] = 0.
    pseudocoordinates = torch.arange(*window)
    prob = evaluate_pseudo(pseudocoordinates, pseudocell_latent_space, gene_ix)
    ax1.plot(pseudocoordinates.cpu().numpy(), prob)
# ax1.set_ylim(0, 0.01)
# ax.set_ylim(0, 0.01)

# %% [markdown]
# ### Subsample

# %%
import peakfreeatac.loaders.fragments

# %%
cells_oi = np.arange(0, 1000)
genes_oi = np.arange(0, 1000)

n_cells = len(cells_oi)
n_genes = len(genes_oi)
loader = pfa.loaders.fragments.Fragments(fragments, n_cells * n_genes, window)
cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

# %%
minibatch = pfa.loaders.minibatching.Minibatch(cellxgene_oi = cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)
data = loader.load(minibatch)
data.local_gene_ix = (data.local_cellxgene_ix % data.n_genes)
data.cell_latent_space = torch.from_numpy(simulation.cell_latent_space[cells_oi]).to(torch.float)

# %% [markdown]
# ## Train

# %%
import peakfreeatac.models.vae.v3 as vae_model

# %%
fragments.window = window

# %%
model = vae_model.Decoding(fragments, cell_latent_space)

# %%
# Train model
max_iter = 2000
show_iter = 50

loss_hist = np.array([])

decoder = decoder.to(device).train()
mixture = mixture.to(device).train()
coordinates = coordinates.to(device)
latent = latent.to(device)

for it in tqdm.tqdm(range(max_iter)):    
    optimizer.zero_grad()
    
    # Compute loss
    genes_oi = torch.from_numpy(data.genes_oi).to(device)
    logit_change, rho_change = decoder(data.cell_latent_space, genes_oi)
    logit_change_cellxgene = logit_change.view(np.prod(logit_change.shape[:2]), logit_change.shape[-1])
    logit_change_fragments = logit_change_cellxgene[data.local_cellxgene_ix]
    
    likelihood_left = mixture.log_prob(data.coordinates[:, 0], logit_change_fragments, genes_oi, data.local_gene_ix)
    likelihood_right = mixture.log_prob(data.coordinates[:, 1], logit_change_fragments, genes_oi, data.local_gene_ix)
    
    likelihood_loci = likelihood_left.sum() + likelihood_right.sum()
    
    fragmentexpression = (rho_bias[data.genes_oi] * torch.exp(rho_change)) * libsize[data.cells_oi].unsqueeze(1)
    fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
    fragmentcounts = torch.reshape(torch.bincount(data.local_cellxgene_ix, minlength = data.n_cells * data.n_genes), (data.n_cells, data.n_genes))
    likelihood_fragmentcounts = fragmentcounts_p.log_prob(fragmentcounts)
    
    loss = -likelihood_loci.sum() - likelihood_fragmentcounts.sum()
    
    # Do backprop and optimizer step
    loss.backward()
    optimizer.step()
    
    if (it % show_iter) == 0:
        print(loss.detach().cpu().item())
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

# Plot loss
plt.figure(figsize=(3, 3))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %%
decoder = decoder.to("cpu").eval()
mixture = mixture.to("cpu").eval()
coordinates = coordinates.to("cpu")
latent = latent.to("cpu")


# %% [markdown]
# Test

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


# %%
gene_ix = 1

# %% jp-MarkdownHeadingCollapsed=true
## Plot prior distribution
fig, (ax, ax1) = plt.subplots(2, 1, figsize=(40, 4))
fragments_oi = (fragments.genemapping == gene_ix)
ax.hist(coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "stepfilled", density = True, log = False, color = "#AAA")
for dim in range(latent.shape[1]):
    fragments_oi = (cell_latent_space[:, dim] > 0)[fragments.cellmapping] & (fragments.genemapping == gene_ix)
    ax.hist(coordinates.cpu().numpy()[fragments_oi, :].flatten(), bins=1000, range = window, histtype = "step", density = True, log = False)

## Plot posterior distribution
# value = torch.arange(*window)
# ax1.plot(pseudocoordinates.cpu().numpy(), prob)
for dim in range(latent.shape[1]):
    pseudocell_latent_space = torch.zeros((pseudocoordinates.shape[0], cell_latent_space.shape[1]))
    pseudocell_latent_space[:, dim] = 1.
    pseudocoordinates = torch.arange(*window)
    prob = evaluate_pseudo(pseudocoordinates, pseudocell_latent_space, gene_ix)
    ax1.plot(pseudocoordinates.cpu().numpy(), prob)
ax1.set_ylim(0, 0.01)
ax.set_ylim(0, 0.01)

# %% [markdown]
# ## VAE

# %%
device = "cuda"

# %%
import peakfreeatac.models.vae.v2 as vae_model
import peakfreeatac.models.positional.v16 as positional_model

# %% [markdown]
# Train

# %%
fragments.window = window

# %%
model = vae_model.VAE(fragments)

# %%
optimizer = pfa.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-3, weight_decay = 1e-5)

# %%
loaders_train.restart()
loaders_validation.restart()

# %%
trainer = pfa.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 400, checkpoint_every_epoch=10, optimize_every_step = 1)
trainer.train()

# %%
import itertools

# %%
model = model.to(device)

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
cell_order = np.argsort(simulation.cell_latent_space[:, 0])
sns.heatmap(embedding[cell_order])

# %%
adata = sc.AnnData(obs = fragments.obs)

# %%
adata.obsm["latent"] = embedding

# %%
sc.pp.neighbors(adata, use_rep = "latent")
sc.tl.umap(adata)

# %%
adata.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
cell_latent_space_pd = pd.DataFrame(simulation.cell_latent_space, columns = "latent_gs_" + pd.Series(np.arange(simulation.n_cell_components)).astype("str"), index = adata.obs.index)
adata.obs[cell_latent_space_pd.columns] = cell_latent_space_pd

# %%
sc.pl.umap(adata, color = ["n_fragments", *cell_latent_space_pd.columns])

# %%
