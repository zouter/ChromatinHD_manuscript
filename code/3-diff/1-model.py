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

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %% [markdown]
# ## Get the dataset

# %%
dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "e18brain"
regions_name = "100k100k"
# regions_name = "10k10k"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
# fragments2 = fragments.filter_regions(fragments.regions.filter([transcriptome.gene_id("CD74")]))
# fragments2.create_regionxcell_indptr()
fragments2 = fragments

# %%
import chromatinhd.models.diff.model.binary
model = chd.models.diff.model.binary.Model.create(
    fragments2,
    clustering,
    fold = fold,
    encoder = "shared",
    # encoder = "split",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale = 1.5,
        bias_regularization=True,
        # binwidths = (5000, 1000)
        binwidths = (5000, 1000, 500, 100, 50)
    )
)
self = model

# %%
model.encoder.w_0.weight.data[0, 0] = 1
model.encoder.w_1.weight.data[0, 2] = 1
model.encoder.w_1.weight.data[0, 10] = 1
model.encoder.w_1.weight.data[0, 100] = 1
w, kl = model.encoder._calculate_w(torch.tensor([0]))

final_w = w.detach()#.repeat_interleave(model.encoder.total_width // model.encoder.n_final_bins, -1).detach()

# %%
final_w[0, 0].max()

# %%
(torch.exp(final_w)[0, 0].repeat_interleave(model.encoder.total_width // model.encoder.n_final_bins, -1)).sum()

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 2))
# plt.plot(torch.exp(final_w)[0, 0])
plt.plot((final_w)[0, 0])

# %%
assert set(model.parameters_sparse()).__contains__(model.encoder.w_delta_0.weight)
assert set(model.parameters_sparse()).__contains__(model.overall_delta.weight)

# %% [markdown]
# ## Test

# %%
loader = chd.loaders.clustering_fragments.ClusteringCuts(fragments, clustering, 50000)

symbol_oi = "IL1B"
minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.n_cells), np.array([transcriptome.gene_ix("IL1B")]))
data = loader.load(minibatch)

coords = torch.clamp(data.cuts.coordinates, self.window[0], self.window[1] - 1) - self.window[0]
bin_ix = coords // self.encoder.binwidths[-1]

# %%
model.forward(data)

# %% [markdown]
# ## Train

# %%
model.train_model()
model.trace.plot()
""

# %%
loader = chd.loaders.clustering_fragments.ClusteringCuts(fragments2, clustering, 500000)

genes_oi = fragments2.var.index[:100]
gene_ixs = fragments2.var.index.get_indexer(genes_oi)
minibatch = chd.loaders.minibatches.Minibatch(fold["cells_validation"], gene_ixs)
data = loader.load(minibatch)

# %%
multiplier = torch.tensor(1.0, requires_grad=True)
elbo = model.forward(data, w_delta_multiplier = multiplier)
elbo.backward()
multiplier.grad

# %%
scores = []
multipliers = np.linspace(0.8, 1.2, 10)
for multiplier in multipliers:
    elbo = model.forward(data, w_delta_multiplier = multiplier)
    scores.append(elbo.item())

plt.plot(multipliers, scores, marker = ".")

# %% [markdown]
# ## Evaluate

# %%
gene_ix = fragments2.var.index.get_loc(transcriptome.gene_id("CD74"))

w_delta = []
for i in range(len(model.encoder.binwidths)):
    w_delta.append(getattr(model.encoder, f"w_delta_{i}").get_full_weight()[gene_ix].reshape(clustering.n_clusters, -1).detach().numpy())
for i, w_delta_level in enumerate(w_delta):
    print(w_delta_level.std())

# %%
for phase in ["train", "validation", "test"]:
    cell_ixs = fold["cells_" + phase]
    # cell_ixs = cell_ixs[(clustering.labels[cell_ixs] == "B")]
    prediction = model.get_prediction(cell_ixs = cell_ixs, regions = transcriptome.gene_id(["CCL4"]))
    print(len(cell_ixs), prediction["likelihood"].sum().item() / len(cell_ixs))

# %%
# symbol = "CD74"
symbol = "CCL4"
# symbol = "ITGA6"
# symbol = "CD79A"
# symbol = "JCHAIN"
# symbol = "SPI1"
# symbol = "TCF4"
# symbol = "PKIA"
# symbol = "PDE7B"
# symbol = "IL1B"
# symbol = "GZMH"

gene_id = transcriptome.gene_id(symbol)
gene_ix = transcriptome.gene_ix(symbol)

# %%
genepositional = chd.models.diff.interpret.GenePositional()
genepositional.score(fragments2, clustering, [model], genes = [transcriptome.gene_id(symbol)], force = True)

# %%
import math

# %%
fragments.counts[:, gene_ix].mean()

# %%
plotdata = genepositional.probs[gene_id].sel(cluster = "NK").to_dataframe(name = "prob")
np.trapz(np.exp(plotdata.prob), plotdata.index)

# %%
clustering.cluster_info["n_cells"] = transcriptome.obs["celltype"].value_counts()

# %%
motifscan_name = "hocomoco_0001"
motifscan = chd.data.motifscan.Motifscan(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))
width = 10

window = fragments.regions.window
# window = [-50000, 50000]
# window = [-10000, 10000]
# window = [-25000, -15000]
# window = [-25000-10000, -15000+10000]
# window = [-100000, -90000]
# window = [-60000, -50000]
# window = [-50000, -40000]
# window = [-20000, 0]
# window = [0, 100000]

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window)
fig.main.add_under(panel_genes)

plotdata, plotdata_mean = genepositional.get_plotdata(transcriptome.gene_id(symbol))

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=transcriptome.gene_id(symbol), panel_height=0.4, order = True
)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=clustering.cluster_info, panel_height=0.4, width=width, window = window, order = panel_expression.order
)

fig.main.add_under(panel_differential)
fig.main.add_right(panel_expression, row=panel_differential)

# motifs_oi = motifscan.motifs.loc[motifscan.motifs.index.str.contains("SPI1") | motifscan.motifs.index.str.contains("PO2") | motifscan.motifs.index.str.contains("TCF")]
# panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi, width = width, window = window)

# fig.main.add_under(panel_motifs)

import chromatinhd_manuscript as chdm
panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = window, width = width)
fig.main.add_under(panel_peaks)

fig.plot()

# %%

# %%
cluster_id = ["memory B", "naive B"]
cluster_ixs = clustering.cluster_info.index.get_indexer(cluster_id)

# %%
level = 0
location = -10000
w_delta = getattr(model.encoder, f"w_delta_{level}").weight.data.reshape(fragments.n_regions, clustering.n_clusters, -1)

bin_ix = (location - self.window[0]) // self.encoder.binwidths[level]

for w in [-0.5, -0.1, 0., 0.1, 0.5, 1., -200.0]:
    w_delta[gene_ix, cluster_ixs, bin_ix] = w

    getattr(model.encoder, f"w_delta_{level}").weight.data[:] = w_delta.reshape(getattr(model.encoder, f"w_delta_{level}").weight.data.shape)

    for phase in ["validation"]:
        cell_ixs = fold["cells_" + phase]
        prediction = model.get_prediction(cell_ixs = cell_ixs, regions = [gene_id])
        print(len(cell_ixs), prediction["likelihood"].sum().item() / len(cell_ixs))

# %%
minibatch = chd.loaders.minibatches.Minibatch(fold["cells_test"], np.array([gene_ix]))
data = loader.load(minibatch)
model.forward(data)

# %% [markdown]
# ## Multiple models

# %%
models = chd.models.diff.model.binary.Models.create(
    fragments=fragments,
    clustering=clustering,
    model_params=dict(binwidths=(5000, 1000, 500, 200, 100, 50)),
    folds=folds[:1],
    reset=True,
)
# models = chd.models.diff.model.binary.Models.create(
#     fragments=fragments,
#     clustering=clustering,
#     model_params=dict(binwidths=(50, )),
#     folds=folds[:1],
#     reset=True,
# )
# models = chd.models.diff.model.binary.Models.create(
#     fragments=fragments,
#     clustering=clustering,
#     model_params=dict(binwidths=(50, ), w_delta_regularization=False),
#     folds=folds,
#     reset=True,
# )

# %%
models.train_models()

# %%
import chromatinhd.models.diff.interpret.performance
performance = chd.models.diff.interpret.Performance.create(
    folds = folds[:1],
    fragments = fragments,
    path = chd.get_output() / "test",
    overwrite = True
)
performance.score(models)

# %%
performance.scores["likelihood"].sel_xr().sel(phase = "test").mean("fold").mean("gene")

# %%
performance.scores["likelihood"].sel_xr().sel(phase = "test").mean("fold").mean("gene")

# %%
gene_ix = transcriptome.gene_ix("CCL4")

# %%
model = models["0"]

# %%
w_delta = []
for i in range(len(model.binwidths)):
    w_delta.append(getattr(model, f"w_delta_{i}").get_full_weight()[gene_ix].reshape(clustering.n_clusters, -1).detach().numpy())
for i, w_delta_level in enumerate(w_delta):
    print(w_delta_level.std())
