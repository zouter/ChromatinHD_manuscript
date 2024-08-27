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

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

import pickle

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_name = "pbmc10k/subsets/mono_t_ab"

# %%
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "10k10k")

# %%
import chromatinhd.models.pred.model.encoders

# %%
loader = chd.loaders.Fragments(fragments, 1000000)
minibatch = chd.loaders.minibatches.Minibatch(np.arange(200), np.arange(100))

# %%
data = loader.load(minibatch)

# %%
fragment_embedder = chromatinhd.models.pred.model.encoders.RadialBinaryEncoding((10, ), window = fragments.regions.window)

# %%
embedding = fragment_embedder(data.coordinates)

# %%
import torch
import torch.nested

# %%
ixs = torch.diff(torch.where(torch.diff(torch.nn.functional.pad(data.local_cellxregion_ix, (1, 1), value = -1)) != 0)[0])
coordinates = torch.nested.nested_tensor(list(torch.split(data.coordinates.float(), tuple(ixs.numpy()))))

# %%
coordinates_padded = coordinates.to_padded_tensor(0)
mask = (coordinates_padded == 0).all(-1)

# %%
fragment_embedding = fragment_embedder(coordinates_padded) * ~mask.unsqueeze(-1)

# %%
attn = torch.nn.MultiheadAttention(fragment_embedder.n_embedding_dimensions, 1, batch_first=True, bias = False)
attn.in_proj_weight.data[2*fragment_embedder.n_embedding_dimensions:3*fragment_embedder.n_embedding_dimensions] = torch.eye(fragment_embedder.n_embedding_dimensions)
attn.in_proj_weight.data[0*fragment_embedder.n_embedding_dimensions:1*fragment_embedder.n_embedding_dimensions] = torch.eye(fragment_embedder.n_embedding_dimensions)
attn.in_proj_weight.data[1*fragment_embedder.n_embedding_dimensions:2*fragment_embedder.n_embedding_dimensions] = torch.eye(fragment_embedder.n_embedding_dimensions)
attn.out_proj.weight.data[:] = torch.eye(fragment_embedder.n_embedding_dimensions)

# %%
fig, ax  = plt.subplots(figsize = (2, 2))
sns.heatmap(fragment_embedding[~mask])

fig, ax  = plt.subplots(figsize = (2, 2))
output, weight = attn(fragment_embedding, fragment_embedding, fragment_embedding, key_padding_mask = mask, need_weights = True)
sns.heatmap(output[~mask].detach().numpy())

fig, ax  = plt.subplots(figsize = (2, 2))
sns.heatmap(output[~mask].detach().numpy() - fragment_embedding[~mask].detach().numpy())

# %%
sns.heatmap(output[~mask].detach().numpy())

# %% [markdown]
# ## Fixed context length

# %%
attn = torch.nn.MultiheadAttention(fragment_embedder.n_embedding_dimensions, 2, batch_first=True, bias = False)
attn.in_proj_weight.data[2*fragment_embedder.n_embedding_dimensions:3*fragment_embedder.n_embedding_dimensions] = torch.eye(fragment_embedder.n_embedding_dimensions)
attn.in_proj_weight.data[0*fragment_embedder.n_embedding_dimensions:1*fragment_embedder.n_embedding_dimensions] = torch.eye(fragment_embedder.n_embedding_dimensions)
attn.in_proj_weight.data[1*fragment_embedder.n_embedding_dimensions:2*fragment_embedder.n_embedding_dimensions] = torch.eye(fragment_embedder.n_embedding_dimensions)
attn.out_proj.weight.data[:] = torch.eye(fragment_embedder.n_embedding_dimensions)

# %%
fragment_embedding = fragment_embedder(data.coordinates)

# %%
fragment_embedding_doublet = fragment_embedding[data.doublet_idx]
fragment_embedding_doublet_reshaped = fragment_embedding_doublet.reshape(len(data.doublet_idx)//2, 2, -1)

# %%
output = attn(fragment_embedding_doublet_reshaped, fragment_embedding_doublet_reshaped, fragment_embedding_doublet_reshaped, need_weights = False)[0].reshape(-1, fragment_embedder.n_embedding_dimensions)

# %%
fig, ax  = plt.subplots(figsize = (2, 2))
sns.heatmap(fragment_embedding_doublet.detach().numpy())

fig, ax  = plt.subplots(figsize = (2, 2))
sns.heatmap(output.detach().numpy())

fig, ax  = plt.subplots(figsize = (2, 2))
sns.heatmap(output.detach().numpy() - fragment_embedding_doublet.detach().numpy())

# %%
sns.heatmap(fragment_embedding_doublet.detach().numpy()[6:8])

# %%
sns.heatmap(output[6:8].detach().numpy())

# %%
fragment_embedding[data.doublet_idx] = fragment_embedding[data.doublet_idx] + output

# %%
sns.heatmap(fragment_embedding.detach().numpy())

# %%
