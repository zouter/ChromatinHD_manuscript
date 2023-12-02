# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %%
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

# %%
n_genes = 20
n_latent = 2

# %%
design = chd.utils.crossing(
    # pd.DataFrame({"gene_ix":[2]}),
    pd.DataFrame({"gene_ix": np.arange(n_genes)}),
    pd.DataFrame({"coord": np.linspace(0, 1, 100)}),
    pd.DataFrame({"reflatent": [0, 1]}),
)

# %%
nbins = (8, 16, 32, 64, 128, 256)
# nbins = (8, )
# nbins = (2, 2)

# %%
dtype = torch.float32
dtype = torch.float64

# %%
cut_coordinates = torch.from_numpy(design["coord"].values).to(dtype)
cut_local_gene_ix = torch.from_numpy(design["gene_ix"].values)
design["position"] = (cut_coordinates + cut_local_gene_ix) / n_genes
cut_positions = torch.from_numpy(design["position"].values)
cut_local_reflatentxgene_ix = torch.from_numpy(
    design["reflatent"].values * n_genes + design["gene_ix"].values
)
cut_local_reflatent_ix = torch.from_numpy(design["reflatent"].values)
mixture_delta_reflatentxgene = torch.zeros((n_latent, n_genes, sum(nbins))).to(dtype)

# %%
import chromatinhd.models.likelihood.v8 as likelihood_model

# %%
transform = likelihood_model.spline.DifferentialQuadraticSplineStack(
    nbins=nbins, n_genes=n_genes
)

# %%
transform.unnormalized_widths.data = transform.unnormalized_widths.data.to(dtype)
transform.unnormalized_heights.data = transform.unnormalized_heights.data.to(dtype)

transform.unnormalized_widths.data[:, :] = 0.0
transform.unnormalized_heights.data[:, :] = 0.0
torch.manual_seed(5)
transform.unnormalized_heights.data.normal_(std=0.5)
transform.unnormalized_heights.data[0, : nbins[0]] = 1.0
# transform.unnormalized_heights.data[:, :] = 1.
# transform.unnormalized_heights.data[:1, :] = 1.
# transform.unnormalized_heights.data[2, :] = -10
# transform.unnormalized_heights.data[:1, :98] = 1.
# transform.unnormalized_heights.data[:1, :] = 1.
# transform.unnormalized_heights.data[0, :3] = torch.linspace(0, 1, 3)
# transform.unnormalized_heights.data[1, :98] = torch.linspace(20, 0, 98)

mixture_delta_reflatentxgene[:] = 0.0
# mixture_delta_reflatentxgene[0, 0, :1] = 2.
# mixture_delta_reflatentxgene[1, 1, 50] = -2
mixture_delta_reflatentxgene[1, 0, : nbins[0]] = -1

# transform.unnormalized_heights.data = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.float64)
# transform.unnormalized_heights.data = torch.tensor([[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float64)
# transform.unnormalized_heights.data = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.float64)

# %%
transform.unnormalized_heights.shape

# %%
transform.unnormalized_heights.data.shape

# %%
sns.heatmap(
    (
        transform.unnormalized_heights.data.cpu().numpy()
        + mixture_delta_reflatentxgene[0].data.cpu().numpy()
    )
)

# %%
logabsdets, outputs = transform.transform_progressive(
    cut_positions,
    cut_local_reflatentxgene_ix,
    cut_local_gene_ix,
    cut_local_reflatent_ix,
    mixture_delta_reflatentxgene,
)

# %%
np.trapz(np.exp(logabsdets[-1].detach().cpu().numpy()), design["position"])

# %%
fig, axes = plt.subplots(
    1, len(outputs), figsize=(len(logabsdets) * 2, 2), sharey=True, sharex=True
)
for (i, output), ax in zip(enumerate(outputs), axes):
    for reflatent, design_reflatent in design.groupby("reflatent"):
        ax.set_aspect(1)
        design_reflatent["position"] = (
            design_reflatent["coord"] + design_reflatent["gene_ix"]
        ) / n_genes
        output_reflatent = output.detach().cpu().numpy()[design_reflatent.index]
        ax.plot(design_reflatent["position"], output_reflatent, label=i)
        ax.scatter(
            design_reflatent.loc[design_reflatent["coord"] == 0, "position"],
            output.detach()
            .cpu()
            .numpy()[design_reflatent.index][design_reflatent["coord"] == 0],
        )

# %%
fig, axes = plt.subplots(len(logabsdets), figsize=(20, len(logabsdets)))
for (i, logabsdet), ax in zip(enumerate(logabsdets), axes):
    for reflatent, design_reflatent in design.groupby("reflatent"):
        logabsdet_reflatent = logabsdet[design_reflatent.index].detach().cpu().numpy()
        ax.plot(design_reflatent["position"], np.exp(logabsdet_reflatent), label=i)
        print(np.trapz(np.exp(logabsdet_reflatent), design_reflatent["position"]))

    for i in range(n_genes + 1):
        ax.axvline(i / n_genes, color="grey")
    ax.set_ylim(0)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")

# %%
inv_logabsdets, samples = transform.transform_progressive(
    cut_positions,
    cut_local_reflatentxgene_ix,
    cut_local_gene_ix,
    cut_local_reflatent_ix,
    mixture_delta_reflatentxgene,
    inverse=True,
)

# %%
fig, axes = plt.subplots(n_latent, figsize=(20, 1 * n_latent))
for latent_ix, ax in zip(range(n_latent), axes):
    sns.histplot(
        samples[-1][design["reflatent"] == latent_ix].cpu().detach().numpy(),
        bins=1000,
        ax=ax,
    )

    for i in range(n_genes + 1):
        ax.axvline(i / n_genes, color="grey")

    ax.set_xlim(0, 1)

    ax.set_yticks([])
    ax.set_ylabel("")

# %% [markdown]
# ## Attempt forward+reverse transformation

# %%
logabsdets, outputs = transform.transform_progressive(
    cut_positions,
    cut_local_reflatentxgene_ix,
    cut_local_gene_ix,
    cut_local_reflatent_ix,
    mixture_delta_reflatentxgene,
)

# %%
output = outputs[-1]

# %%
logabsdets, outputs2 = transform.transform_progressive(
    output,
    cut_local_reflatentxgene_ix,
    cut_local_gene_ix,
    cut_local_reflatent_ix,
    mixture_delta_reflatentxgene,
    inverse=True,
)

# %%
output2 = outputs2[-1]

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(
    cut_positions.cpu().detach().numpy(), output.cpu().detach().numpy(), label="forward"
)
ax.scatter(
    cut_positions.cpu().detach().numpy(),
    output2.cpu().detach().numpy(),
    label="forward+inverse",
)
ax.legend()

# %%
