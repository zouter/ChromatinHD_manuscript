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
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
# dataset = "pbmc10k"
dataset = "hspc"

layer = "normalized"
layer = "magic"

regions = "100k100k"

phase = "test"
# phase = "validation"

transcriptome = transcriptome_original = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset / "transcriptome")
fragments = fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset / "fragments" / "100k100k")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset / "folds" / "5x5")
# fold = folds[0]
# folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset / "folds" / "1")
# folds.folds = [fold]

# %%
symbols_oi = ["CCL4", "IL1B", "TCF4", "SOX5"]
symbols_oi = transcriptome.symbol(transcriptome.var.sort_values("dispersions_norm", ascending = False).index[:10])
symbols_oi = transcriptome.symbol(transcriptome.var.sort_values("dispersions_norm", ascending = False).index[:1])
# symbols_oi = transcriptome.symbol(transcriptome.var.sort_values("dispersions_norm", ascending = False).index[:5])
symbols_oi = transcriptome.symbol(transcriptome.var.query("n_cells > 500").sort_values("dispersions_norm", ascending = False).index[:10])

regions_oi = transcriptome.gene_id(symbols_oi)

# %%
# transcriptome = transcriptome_original.filter_genes(transcriptome_original.gene_id(symbols_oi))
# fragments = fragments_original.filter_regions(fragments_original.regions.filter(transcriptome_original.gene_id(symbols_oi)))
# fragments.create_regionxcell_indptr()

# %%
import chromatinhd.models.pred.model.better

# %%
model_params = dict(
    n_embedding_dimensions = 100,
    n_layers_fragment_embedder=5,
    residual_fragment_embedder=False,
    n_layers_embedding2expression=5,
    residual_embedding2expression=False,
    dropout_rate_fragment_embedder=0.,
    dropout_rate_embedding2expression=0.,
    layer = "magic",
)
train_params = dict(
    weight_decay=1e-1,
)

# %%
import chromatinhd.models.pred.model.better
models = chd.models.pred.model.better.Models.create(fragments = fragments, transcriptome = transcriptome, folds = folds)
models.train_models()

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)

# %%
models = chd.models.pred.model.better.Models.create(
    path = chd.get_output() / "test" / "models",
    reset = True,
    transcriptome = transcriptome,
    fragments = fragments,
    folds = folds,
    model_params = model_params,
    train_params = train_params,
    regions_oi = regions_oi,
)
models.train_models(regions_oi = regions_oi)

# %%
performance = chd.models.pred.interpret.performance.Performance(reset = True)
performance.score(fragments, transcriptome, models, folds)

# %%
performance.regionscores["r2"].sel(phase = "test").mean("fold").to_pandas().mean()

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow()
regionmultiwindow.score(models, censorer)

# %%
regionmultiwindow.interpolate()

# %%
symbol = "CCL4"

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05))
width = 10

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width)
fig.main.add_under(panel_genes)

panel_pileup = chd.models.pred.plot.Pileup.from_regionmultiwindow(
    regionmultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_pileup)

panel_predictivity = chd.models.pred.plot.Predictivity.from_regionmultiwindow(
    regionmultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_predictivity)

fig.plot()

# %% [markdown]
# ---------------------------

# %%
fragments = fragments_original
transcriptome = transcriptome_original

# %%
transcriptome = transcriptome_original.filter_genes(transcriptome_original.gene_id(symbols_oi))
fragments = fragments_original.filter_regions(fragments_original.regions.filter(transcriptome_original.gene_id(symbols_oi)))
fragments.create_regionxcell_indptr()

# %%
fragments.var["ix"] = np.arange(len(fragments.var))
region_ixs = fragments.var["ix"]
region_ixs = fragments.var.loc[transcriptome.gene_id(symbols_oi), "ix"]
minibatcher = chd.loaders.minibatches.Minibatcher(
    np.concatenate([fold["cells_validation"], fold["cells_test"]]),
    region_ixs,
    n_regions_step=10,
    n_cells_step=10000,
    permute_cells=False,
    permute_regions=False,
)

loaders = chd.loaders.LoaderPool(
    chd.loaders.TranscriptomeFragments,
    dict(
        transcriptome=transcriptome,
        fragments=fragments,
        cellxregion_batch_size=minibatcher.cellxregion_batch_size,
        layer=layer,
        region_oi = regions_oi[0],
    ),
)
loaders.initialize(minibatcher)
data = next(iter(loaders))

# %%
import torch

# %%
param = dict(
    n_embedding_dimensions = 100,
    n_layers_fragment_embedder=5,
    residual_fragment_embedder=False,
    n_layers_embedding2expression=5,
    residual_embedding2expression=False,
    dropout_rate_fragment_embedder=0.0,
    dropout_rate_embedding2expression=0.0,
    weight_decay=1e-1,
)
train_params = {}
if "n_cells_step" in param:
    train_params["n_cells_step"] = param.pop("n_cells_step")
if "lr" in param:
    train_params["lr"] = param.pop("lr")
if "weight_decay" in param:
    train_params["weight_decay"] = param.pop("weight_decay")
if "n_epochs" in param:
    train_params["n_epochs"] = param.pop("n_epochs")
if "label" in param:
    param.pop("label")

import chromatinhd.models.pred.model.additive2
model = chd.models.pred.model.additive2.Model(fragments = fragments, transcriptome = transcriptome, fold = fold, regions_oi = transcriptome.gene_id(symbols_oi), layer = "magic", **param)

# model = model.to("cuda")
# data = data.to("cuda")

# loss = model.forward_loss(data)

model.train_model(**train_params)

# %%
model = model.to("cuda")
data = data.to("cuda")

# %%
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    loss = model.forward_loss(data)
    loss.backward()

print(p.key_averages().table(
    sort_by="self_cpu_time_total", row_limit=-1))

# %%
performance = chd.models.pred.interpret.Performance(path = chd.get_output() / "test" / "performance", reset = True)
performance.score(fragments, transcriptome, [model], [fold], pbar = True)
performance.genescores.mean("model")["cor"].to_pandas().style.bar()

# %%
import chromatinhd.models.pred.model.better

# %%
param = dict(
    cls=chd.models.pred.model.better.Model,
    n_embedding_dimensions=100,
    n_layers_fragment_embedder=5,
    residual_fragment_embedder=False,
    n_layers_embedding2expression=5,
    residual_embedding2expression=False,
    dropout_rate_fragment_embedder=0.0,
    dropout_rate_embedding2expression=0.0,
    encoder="spline_binary",
    # distance_encoder="split",
    label="spline_binary_1000-31frequencies_splitdistance_wd1e-1",
    weight_decay=1e-1,
)
param = param.copy()
cls = param.pop("cls")

train_params = {}
if "n_cells_step" in param:
    train_params["n_cells_step"] = param.pop("n_cells_step")
if "lr" in param:
    train_params["lr"] = param.pop("lr")
if "weight_decay" in param:
    train_params["weight_decay"] = param.pop("weight_decay")
if "n_epochs" in param:
    train_params["n_epochs"] = param.pop("n_epochs")
if "label" in param:
    param.pop("label")

model2 = cls(
    fragments = fragments,
    transcriptome=transcriptome,
    fold = fold,
    layer = layer,
    **param,
)

model2.train_model(**train_params, pbar = True)
performance = chd.models.pred.interpret.Performance(path = chd.get_output() / "test" / "performance", reset = True)
performance.score(fragments, transcriptome, [model2], [fold], pbar = False)
performance.genescores.mean("model").mean("gene")

# %%
performance2 = chd.models.pred.interpret.Performance(path = chd.get_output() / "test" / "performance2", reset = True)
performance2.score(fragments, transcriptome, [model2], [fold], pbar = False)
performance2.genescores.mean("model")["cor"].to_pandas().style.bar()
