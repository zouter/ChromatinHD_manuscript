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
# Import required packages
import torch
import numpy as np
import normflows as nf

from sklearn.datasets import make_moons

from matplotlib import pyplot as plt

from tqdm import tqdm

# %%
import scipy.stats

# %%
from peakfreeatac.simulation import Simulation

# %%
import peakfreeatac as pfa
import pathlib
import tempfile

# %% [markdown]
# ### Simulation

# %%
simulation = Simulation(n_genes = 1)

# %%
window = simulation.window

# %% [markdown]
# Create fragments

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
# ### Real

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_clustered"
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
# ### Model

# %%
sns.histplot(fragments.coordinates.flatten(), bins = 500, binrange = window)

# %%
device = "cuda"

# %%
positions_torch = fragments.coordinates.flatten().to(device, torch.float)[:, None]
positions_torch = ((positions_torch - window[0]) / (window[1] - window[0]) - 0.5) * 2
r = np.array([-1, 1])

# %%
optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-5, weight_decay=1e-5)

# %%
x = positions_torch

# %%
from normflows.utils import splines

# %%
import math
class UniformSpline(torch.nn.Module):
    def __init__(self, n_channels, n_bins):
        super().__init__()
        default_init = np.log(np.exp(1 - splines.DEFAULT_MIN_DERIVATIVE) - 1)
        self.unnormalized_widths = torch.nn.Parameter(torch.ones((n_channels, n_bins)) * default_init)
        self.unnormalized_heights = torch.nn.Parameter(torch.ones((n_channels, n_bins)) * default_init)
        self.unnormalized_derivatives = torch.nn.Parameter(torch.ones((n_channels, n_bins-1)) * default_init)
        # self.distribution = torch.distributions.Uniform(torch.tensor(-1), torch.tensor(1.))
        self.distribution = torch.distributions.Normal(0., 1.)
        
    def log_prob(self, x):
        y_, logdet = nf.utils.splines.unconstrained_rational_quadratic_spline(
            x,
            self.unnormalized_widths.unsqueeze(0).expand(x.shape[0], -1, -1) * 10,
            self.unnormalized_heights.unsqueeze(0).expand(x.shape[0], -1, -1) * 10,
            self.unnormalized_derivatives.unsqueeze(0).expand(x.shape[0], -1, -1),
            tail_bound = 1.,
            inverse = True
        )
        return math.log(0.5) + logdet
        # return self.distribution.log_prob(y_) + logdet


# %%
dist = UniformSpline(1, 100).to(device)

# %%
## Plot prior distribution
fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(positions_torch.cpu().numpy()[:, 0], bins=500, range = r, lw = 0)

# Plot initial posterior distribution
log_prob = dist.log_prob(x_).detach().to('cpu')
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

# plt.figure(figsize=(6, 3))
ax2 = ax.twinx()
ax2.plot(x_.cpu().numpy(), prob, color = "red")
ax2.set_ylim(0)
# plt.show()
plt.show()

# %%
optimizer = torch.optim.Adam(dist.parameters(), lr=1e-3, weight_decay=1e-5)

# %%
x = positions_torch.to(device)
dist = dist.to(device)

# %%
# Train model
max_iter = 300
show_iter = 500

loss_hist = np.array([])

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    # x_np, _ = make_moons(num_samples, noise=0.1)
    # x = torch.tensor(x_np).float().to(device)
    
    # Compute loss
    loss = -dist.log_prob(x).sum()
    
    loss.backward()
    optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    # Plot learned posterior
    if (it + 1) % show_iter == 0:
        nfm.eval()
        log_prob = nfm.log_prob(zz)
        nfm.train()
        prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.gca().set_aspect('equal', 'box')
        plt.show()

# Plot loss
plt.figure(figsize=(3, 3))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(positions_torch.cpu().numpy()[:, 0], bins=500, range = r, lw = 0)

# Plot initial posterior distribution
x_ = torch.linspace(*r, 1000)[:, None].to(device)
log_prob = dist.log_prob(x_).detach().to('cpu')
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

# plt.figure(figsize=(6, 3))
ax2 = ax.twinx()
ax2.plot(x_.cpu().numpy(), prob, color = "red")
ax2.set_ylim(0)
# plt.show()
plt.show()

# %% [markdown]
# ## Fit VAE

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_clustered"
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
