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

# %% [markdown]
# # Model promoters positionally

# %%
# %load_ext autoreload
# %autoreload 2

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

# export LD_LIBRARY_PATH=/data/peak_free_atac/software/peak_free_atac/lib
import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_clustered"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
fold = folds[0]

# %%
from design import get_folds_inference

folds = get_folds_inference(fragments, folds)

# %% [markdown]
# What we want is
# - Know which motifs are close to a cut sites. Either precalculate or extract on-the-fly as fast as possible
# - Extract all information of a motif in relation to the fragment
#   - Which motif
#   - Whether it is relative to the left or right cut site
#   - Its distance to the cut site
#   - Its score
# - We want to be able to index extremely fast on the fragments, ideally microseconds

# %% [markdown]
# ## Load fragments

# %%
import pyximport
import sys

pyximport.install(
    reload_support=True,
    language_level=3,
    setup_args=dict(include_dirs=[np.get_include()]),
)
if "chromatinhd.loaders.extraction.fragments" in sys.modules:
    del sys.modules["chromatinhd.loaders.extraction.fragments"]
import chromatinhd.loaders.extraction.fragments

# %% [markdown]
# ### Fragments

# %%
n_cells = 1000
n_genes = 100
cutwindow = np.array([-150, 150])
loader = chromatinhd.loaders.fragments.Fragments(fragments, n_cells * n_genes)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = chromatinhd.loaders.minibatching.Minibatch(cells_oi=cells_oi, genes_oi=genes_oi)
data = loader.load(minibatch)

# %%
# %%timeit -n 1
data = loader.load(minibatch)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

# %% [markdown]
# ### Fragments n (2)

# %%
fragments_oi = fragments.coordinates[:, 0] > 0

# %%
n_cells = 1000
n_genes = 100
cutwindow = np.array([-150, 150])
loader = chromatinhd.loaders.fragments.FragmentsCounting(
    fragments, n_cells * n_genes
)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = chromatinhd.loaders.minibatching.Minibatch(cells_oi=cells_oi, genes_oi=genes_oi)
data = loader.load(minibatch)

# %%
# %%timeit -n 1
data = loader.load(minibatch)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

# %%
data.local_cellxgene_ix[data.n[0]]

# %% [markdown]
# ## Loading using multithreading

# %%
import pyximport
import sys

pyximport.install(
    reload_support=True,
    language_level=3,
    setup_args=dict(include_dirs=[np.get_include()]),
)
if "chromatinhd.loaders.extraction.fragments" in sys.modules:
    del sys.modules["chromatinhd.loaders.extraction.fragments"]
import chromatinhd.loaders.extraction.fragments

# %%
# n_cells = 2000
n_cells = 3
n_genes = 100

cutwindow = np.array([-150, 150])

# %%
import chromatinhd.loaders.fragments

# %%
loaders = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {
        "fragments": fragments,
        "cellxgene_batch_size": n_cells * n_genes
    },
    n_workers=2,
)

# %%
import gc

gc.collect()

# %%
data = []
for i in range(2):
    cells_oi = np.sort(np.random.choice(fragments.n_cells, n_cells, replace=False))
    genes_oi = np.sort(np.random.choice(fragments.n_genes, n_genes, replace=False))

    cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

    data.append(
        chd.loaders.minibatching.Minibatch(
            cells_oi=cells_oi, genes_oi=genes_oi
        )
    )
loaders.initialize(data)

# %%
for i, data in enumerate(tqdm.tqdm(loaders)):
    print(i)
    data
    loaders.submit_next()

# %% [markdown]
# ## Positional encoding

# %%
import chromatinhd.models.positional.v20

# %%
n_frequencies = 20
encoder = chromatinhd.models.positional.v20.SineEncoding(n_frequencies)

# %%
x = torch.arange(-10000, 10000)
coordinates = torch.stack([x, x + 500], 1)
encoding = encoder(coordinates)

# %%
1 / (10000 ** (2 * 0 / 50))

# %%
fig, ax = plt.subplots(figsize=(1, 3), facecolor="w")
sns.heatmap(coordinates.numpy(), cbar_kws={"label": "position"})
ax.collections[0].colorbar.set_label("position", rotation=0)
ax.set_ylabel("fragment", rotation=0, ha="right")
ax.set_yticks([])
ax.set_xticklabels(["left", "right"])
fig.savefig("hi.png", bbox_inches="tight", transparent=True, dpi=300)

# %%
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(encoding.numpy())
ax.collections[0].colorbar.set_label(
    "embedding\nvalue", rotation=0, ha="left", va="center"
)
ax.set_ylabel("fragment", rotation=0, ha="right")
ax.set_yticks([])
ax.set_xlabel("components")
ax.set_xticks([])
fig.savefig("hi.png", bbox_inches="tight", transparent=True, dpi=300)

# %%
i = 0
plt.plot(coordinates[:100, 0], encoding[:100, i])
i = 10
plt.plot(coordinates[:100, 0], encoding[:100, i])
i = 20
plt.plot(coordinates[:100, 0], encoding[:100, i])
i = 50
plt.plot(coordinates[:100, 0], encoding[:100, i])

# %% [markdown]
# ## Fragment counts

# %%
counts = pd.Series(torch.diff(fragments.cellxgene_indptr).numpy())
pd.Series(counts).plot(kind="hist", range=(0, 10), bins=10)

# %% [markdown]
# ## Fragment embedder

# %%
import chromatinhd.models.positional.v20

# %%
embedder = chromatinhd.models.positional.v20.FragmentEmbedder(fragments.n_genes)

# %%
embedder.forward(data.coordinates, data.genemapping)

# %%
1 / (10000 ** (2 * 0 / 50))

# %%
sns.heatmap(encoding.numpy())

# %%
i = 0
plt.plot(coordinates[:100, 0], encoding[:100, i])
i = 10
plt.plot(coordinates[:100, 0], encoding[:100, i])
i = 20
plt.plot(coordinates[:100, 0], encoding[:100, i])
i = 50
plt.plot(coordinates[:100, 0], encoding[:100, i])

# %% [markdown]
# ## Model

# %%
import chromatinhd.models.positional.v20

# %%
mean_gene_expression = transcriptome.X.dense().mean(0)

# %%
model = chromatinhd.models.positional.v20.Model(
    fragments.n_genes,
    mean_gene_expression,
    n_frequencies=50,
    nonlinear="sigmoid",
    reduce="sum",
)
# model = pickle.load(
#     (
#         chd.get_output()
#         / ".."
#         / "output/prediction_positional/pbmc10k/10k10k/v14_50freq_sum_sigmoid_initdefault/model_0.pkl"
#     ).open("rb")
# )

# %%
effect = model.forward(data)
effect = effect - mean_gene_expression[data.genes_oi]

# %% [markdown]
# ## Single example

# %%
transcriptome.create_X()
transcriptome.X
mean_gene_expression = transcriptome.X.dense().mean(0)

# %%
from chromatinhd.models.positional.v20 import Model

# %%
model = Model(fragments.n_genes, mean_gene_expression)

# %%
model.forward(data)

# %% [markdown]
# ## Infer

# %% [markdown]
# ### Loaders

# %%
import chromatinhd.loaders
import chromatinhd.loaders.fragments
import chromatinhd.loaders.fragmentmotif

# %%
from design import get_design, get_folds_training

# %%
design = get_design(transcriptome, fragments)

# %%
prediction_name = "v20"
prediction_name = "counter"
design_row = design[prediction_name]

# %%
# loaders
print("collecting...")
if "loaders" in globals():
    loaders.terminate()
    del loaders
    import gc

    gc.collect()
if "loaders_validation" in globals():
    loaders_validation.terminate()
    del loaders_validation
    import gc

    gc.collect()
print("collected")
loaders = chd.loaders.LoaderPool(
    design_row["loader_cls"], design_row["loader_parameters"], n_workers=20
)
print("haha!")
loaders_validation = chd.loaders.LoaderPool(
    design_row["loader_cls"], design_row["loader_parameters"], n_workers=5
)
loaders_validation.shuffle_on_iter = False

# %%
# folds & minibatching
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
folds = get_folds_training(fragments, folds)


# %%
# loss
def paircor(x, y, dim=0, eps=0.1):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor

loss = lambda x, y: -paircor(x, y).mean() * 100


# %%
class Prediction(chd.flow.Flow):
    pass

print(prediction_name)
prediction = Prediction(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / prediction_name
)

# %%
fold_ix = 0
fold = folds[0]

# %%
# new_minibatch_sets = []
# for minibatch_set in fold["minibatches_train_sets"]:
#     tasks = [minibatch.filter_genes(improved) for minibatch in minibatch_set["tasks"]]
#     new_minibatch_sets.append({"tasks":tasks})

# %%
n_epochs = 20
checkpoint_every_epoch = 1

# %%
# model
model = design_row["model_cls"](**design_row["model_parameters"])

# %%
from chromatinhd.models.positional.trainer import Trainer

# %% tags=[]
# optimization
optimize_every_step = 1
lr = 1e-3  # / optimize_every_step
optim = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(autoextend = False), lr = lr, weight_decay = lr / 2)

# train
import chromatinhd.train

outcome = transcriptome.X.dense()
trainer = Trainer(
    model,
    loaders,
    loaders_validation,
    outcome,
    loss,
    optim,
    checkpoint_every_epoch=checkpoint_every_epoch,
    optimize_every_step=optimize_every_step,
    n_epochs=n_epochs,
    device="cuda",
)
trainer.train(fold["minibatches_train_sets"], fold["minibatches_validation_trace"])

# %%
pd.DataFrame(trainer.trace.validation_steps).groupby("checkpoint").mean()["loss"].plot(
    label="validation"
)
pd.DataFrame(trainer.trace.train_steps).groupby("checkpoint").mean()["loss"].plot(
    label="train"
)

# %%
# model = model.to("cpu")
# pickle.dump(model, open("../../" + dataset_name + "_" + "baseline_model.pkl", "wb"))

# %%
model = model.to("cpu")
pickle.dump(model, open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"))
