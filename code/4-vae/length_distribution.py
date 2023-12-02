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
class FragmentDistribution():
    def __init__(self, rate = 0.01):
        self.rate = rate
        
    def pdf(self, x):
        return self.rate * torch.exp(-self.rate * x)
dist = FragmentDistribution()

# %%
norm_same = lambda x: 1/(1 + torch.exp((-x + 20.0) * 0.1))

# %%
plt.plot(
    x,
    dist.pdf(x)
)
plt.plot(
    x,
    norm_same(x)
)

# %%
import scipy.stats


# %%
class FragmentDistribution():
    def __init__(self, rate = 0.01):
        self.rate = rate
        
    def pdf(self, x):
        return self.rate * torch.exp(-self.rate * x)
dist = FragmentDistribution()

# %%
mean = 180
std = 100**2

scale = std**2 / mean
concentration = mean / scale
rate = 1/scale


# %%
def reparameterize_gamma(mean, std):
    scale = std**2 / mean
    concentration = mean / scale
    rate = 1/scale
    return concentration, rate


# %%
means = torch.tensor([150, 180, 180*2])
stds = torch.tensor([120, 30, 30])

# %%
components = torch.distributions.Gamma(*reparameterize_gamma(means, stds))
mixture = torch.distributions.Categorical(probs = torch.tensor([1, 0.2, 0.2]))

dist = torch.distributions.MixtureSameFamily(mixture, components)

# %%
x = torch.linspace(0, 1000, 100)
prob = torch.exp(dist.log_prob(x))

# %%
fig, ax = plt.subplots()
ax.plot(x, prob)

# %%
samples = dist.sample((1000, ))
print(samples.mean())
print(samples.var())

# %%
scipy.stats.gamma(0, 1).pdf

# %%
