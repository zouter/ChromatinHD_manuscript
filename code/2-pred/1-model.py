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

# %% [markdown]
# # Model promoters positionally

# %%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)

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

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
chd.set_default_device("cuda:1")

# %%
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k/subsets/top5"
# dataset_name = "pbmc10k/subsets/top1"
dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "e18brain"
regions_name = "100k100k"
# regions_name = "500k500k"
# regions_name = "10k10k"

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
fragments2 = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
torch.manual_seed(1)

import chromatinhd.models.pred.model.better
model_params = dict(
    n_embedding_dimensions=100,
    n_layers_fragment_embedder=5,
    n_layers_embedding2expression=5,
    residual_embedding2expression=True,
    residual_fragment_embedder=True,
    layernorm_embedding2expression=True,
    encoder="spline_binary",
    nonlinear = "silu",
    library_size_encoder="linear",

)
train_params = dict(
    weight_decay=1e-1,
    optimizer = "adam",
    lr = 1e-4,
)

# gene_oi = fragments.var.index[20]
gene_oi = transcriptome.gene_id("JCHAIN")
gene_ix = transcriptome.var.index.get_loc(gene_oi)

model = chd.models.pred.model.better.Model(
    fragments = fragments,
    transcriptome = transcriptome,
    fold = fold,
    layer = "magic",
    region_oi = gene_oi,
    **model_params
)

# %%
minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.obs.shape[0]), np.array([gene_ix]))
# loader = chd.loaders.Fragments(fragments, 50000)
loader = chd.loaders.FragmentsRegional(fragments, 50000, gene_oi)
data = loader.load(minibatch)

# %%
model.train_model(**train_params, n_epochs = 1000, device = 'cuda:0')

# %%
model.trace.plot()

# %%
prediction = model.get_prediction(cell_ixs = fold["cells_train"]).sel(gene = gene_oi)
np.corrcoef(prediction["predicted"], prediction["expected"])[0, 1]

# %%
models = chd.models.pred.model.better.Models()
models[gene_oi + "_0"] = model
models.folds = folds

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(folds, transcriptome, fragments, censorer, path = chd.get_output() / "test", overwrite = True)

# %%
regionmultiwindow.score(models, regions = [gene_oi])

# %%
regionmultiwindow.scores["deltacor"].sel_xr().sel(gene = gene_oi).sel(phase = "test").mean("fold").to_pandas().plot()

# %%
import cProfile

stats = cProfile.run("regionmultiwindow.score(models, regions = [gene_oi], device = 'cpu')", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats(50)
