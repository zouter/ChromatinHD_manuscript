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
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd

# %%
dataset_name = "pbmc10k"
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "100k100k")
regions = fragments.regions

transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x5")
folds.sample_cells(fragments, 5, 5, overwrite = True)
# folds.sample_cells(fragments, 25, 1, overwrite = True)
folds = folds[:1]

# %%
models_folder = chd.get_output() / "models" / dataset_name

# %%
import chromatinhd.models.pred.interpret

# %%
chd.set_default_device("cuda:0")
# chd.set_default_device("cpu")

# %%
models = []
for fold in folds:
    model = chd.models.pred.model.additive.Model(fragments = fragments, transcriptome = transcriptome, layer = "normalized")
    model.train_model(fold = fold, pbar = True, n_epochs = 1)
    models.append(model)
performance = chd.models.pred.interpret.Performance()
performance.score(fragments, transcriptome, models, folds)

# %%
# %%timeit -r 1 -n 1
chd.set_default_device("cuda")
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes = (500, ))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset=True)

regionmultiwindow.score(fragments, transcriptome, models, folds, censorer)

# %%
# %%timeit -r 1 -n 5
chd.set_default_device("cpu")
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes = (100, 200, 500))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset=True)

gene_oi = transcriptome.gene_id("CCL4")
regionmultiwindow.score(fragments, transcriptome, models, folds, censorer, regions = [gene_oi])

# %%
# %%timeit -r 1 -n 1
performance = chd.models.pred.interpret.Performance(reset = True)
performance.score(fragments, transcriptome, models, folds)

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes = (100, 200, 500))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset=True)

gene_oi = transcriptome.gene_id("CCL4")
regionmultiwindow.score(fragments, transcriptome, models, folds, censorer, regions = [gene_oi])
scores_raw = regionmultiwindow.scores[gene_oi].copy()

# %%
interpolateds = []
n = len(scores_raw.coords["model"])
for i in tqdm.trange(100):
    scores = scores_raw.sel(model = np.random.choice(n, n, replace = True))
    regionmultiwindow.scores[gene_oi] = scores
    regionmultiwindow.interpolate(force = True)
    interpolateds.append(regionmultiwindow.interpolated[gene_oi].copy())

# %%
interpolateds_stacked = xr.concat(interpolateds, dim = "replicate")["deltacor"]

# %%
alpha = 0.95
q95 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q95.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")
alpha = 0.90
q90 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q90.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")
alpha = 0.80
q80 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q80.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")
alpha = 0.50
q50 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q50.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")

med = q50.mean("quantile")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
import chromatinhd_manuscript as chdm

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height = 0))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 20000]
window = [-10000, 10000] # TSS
# window = [-20000, -10000] # PFKFB3 enhancer

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width = 10, window = window))
ax.set_xlim(*window)

# panel, ax = fig.main.add_under(chd.models.pred.plot.Predictivity(regionmultiwindow.get_plotdata(gene_oi), window = window, width = 10, color_by_effect=False))

panel, ax = fig.main.add_under(chd.grid.Panel((10, 1.5)))
ax.fill_between(q95.coords["position"], q95.sel(quantile = "upper"), q95.sel(quantile = "lower"), fc = "#AAAAAA", lw = 0.5, ec = "#33333344", label = "95%")
ax.fill_between(q90.coords["position"], q90.sel(quantile = "upper"), q90.sel(quantile = "lower"), fc = "#888888", lw = 0, label = "90%")
ax.fill_between(q80.coords["position"], q80.sel(quantile = "upper"), q80.sel(quantile = "lower"), fc = "#444444", lw = 0, label = "80%")
ax.fill_between(q50.coords["position"], q50.sel(quantile = "upper"), q50.sel(quantile = "lower"), fc = "black", lw = 0, label = "50%")
plotdata = regionmultiwindow.get_plotdata(gene_oi)
ax.plot(med.coords["position"], med, color = "red", lw = 0.5, label = "median")
ax.set_ylabel("$\Delta$ cor")
ax.invert_yaxis()
ax.set_xlim(*window)
ax.set_ylim(0, -0.1)
ax.legend(title = "Confidence interval")
ax.set_xticks([])

panel, ax = fig.main.add_under(chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = fragments.regions.window, width = 10, peakcallers = ["cellranger", "macs2_improved", "macs2_leiden_0.1", "macs2_leiden_0.1_merged"]))
ax.set_xlim(*window)
ax.set_xticks([])

panel, ax = fig.main.add_under(chd.models.pred.plot.Pileup(regionmultiwindow.get_plotdata(gene_oi), window = window, width = 10))

fig.plot()

# %%
fig, ax = plt.subplots(figsize = (20, 2))
ax.fill_between(q95.coords["position"], q95.sel(quantile = "upper"), q95.sel(quantile = "lower"), fc = "#AAAAAA", lw = 0.5, ec = "#33333344")
ax.fill_between(q90.coords["position"], q90.sel(quantile = "upper"), q90.sel(quantile = "lower"), fc = "#888888", lw = 0)
ax.fill_between(q80.coords["position"], q80.sel(quantile = "upper"), q80.sel(quantile = "lower"), fc = "#444444", lw = 0)
ax.fill_between(q50.coords["position"], q50.sel(quantile = "upper"), q50.sel(quantile = "lower"), fc = "black", lw = 0)
# ax.set_xlim(-1000, 1000)
ax.invert_yaxis()
ax.axhline(0, lw = 0.5, c = "#333", dashes = (2, 2))

# %%
interpolateds = []
n = len(scores_raw.coords["model"])
for i in tqdm.trange(n):
    # scores = scores_raw.sel(model = np.random.choice(25, 5, replace = True))
    scores = scores_raw.sel(model = [i])
    regionmultiwindow.scores[gene_oi] = scores
    regionmultiwindow.interpolate(force = True)
    interpolateds.append(regionmultiwindow.interpolated[gene_oi].copy())

# %%
interpolateds_stacked = xr.concat(interpolateds, dim = "replicate")["deltacor"]

# %%
alpha = 0.95
q95 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q95.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")
alpha = 0.90
q90 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q90.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")
alpha = 0.80
q80 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q80.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")
alpha = 0.50
q50 = interpolateds_stacked.quantile([(1-alpha)/2, 1- ((1-alpha)/2)], "replicate")
q50.coords["quantile"] = pd.Index(["lower", "upper"], name = "quantile")

# %%
fig, ax = plt.subplots(figsize = (20, 2))
ax.fill_between(q95.coords["position"], q95.sel(quantile = "upper"), q95.sel(quantile = "lower"), fc = "#AAAAAA", lw = 0.5, ec = "#33333344")
ax.fill_between(q90.coords["position"], q90.sel(quantile = "upper"), q90.sel(quantile = "lower"), fc = "#888888", lw = 0)
ax.fill_between(q80.coords["position"], q80.sel(quantile = "upper"), q80.sel(quantile = "lower"), fc = "#444444", lw = 0)
ax.fill_between(q50.coords["position"], q50.sel(quantile = "upper"), q50.sel(quantile = "lower"), fc = "black", lw = 0)
ax.set_xlim(-1000, 1000)
ax.invert_yaxis()
ax.axhline(0, lw = 0.5, c = "#333", dashes = (2, 2))

# %% [markdown]
# ## Delta cor delta

# %%
censorer = chd.models.pred.interpret.WindowCensorer(fragments.regions.window)
regionpairwindow = chd.models.pred.interpret.RegionPairWindow(chd.get_output() / "test2", reset=True)
regionpairwindow.score(fragments, transcriptome, models, folds, censorer=censorer, regions=transcriptome.gene_id(["CCL4"]))

# %%
symbol = "CCL4"

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05))
width = 10

# genes
region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width)
fig.main.add_under(panel_genes)

# pileup
panel_pileup = chd.models.pred.plot.Pileup.from_regionmultiwindow(
    regionmultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_pileup)

# predictivity
panel_predictivity = chd.models.pred.plot.Predictivity.from_regionmultiwindow(
    regionmultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_predictivity)

# copredictivity
panel_copredictivity = chd.models.pred.plot.Copredictivity.from_regionpairwindow(
    regionpairwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_copredictivity)

fig.plot()

# %%
ranking = regionpairwindow.get_plotdata(gene_oi).sort_values("cor")
ranking_oi = ranking.iloc[-1]
window1 = ranking_oi["window1"]
window2 = ranking_oi["window2"]

# %%
cors = regionpairwindow.interaction[gene_oi].stack({"window1_window2":["window1", "window2"]})

# %%
# cors = regionpairwindow.interaction[gene_oi].sel(window1 = window1, window2 = window2)

random = []
n = len(cors.coords["model"])
for i in range(1000):
    random.append(cors.sel(model = np.random.choice(n, n, replace = True)).mean("model"))
random = xr.concat(random, dim = "replicate")

# %%
plotdata_random = random.sel(window1_window2 = (window1, window2))
plotdata_cors = cors.sel(window1_window2 = (window1, window2))

fig, ax = plt.subplots(figsize = (2, 1.))
sns.boxplot(y = [2.] * len(plotdata_random) + [1.] * len(plotdata_cors), x = plotdata_random.values.tolist() + plotdata_cors.values.tolist(), ax = ax, orient = "h", showfliers = False, whis = [0.025, 0.975])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Prediction interval (n=25)", "Confidence interval (n=1000)"])
ax.set_xlabel("cor $\Delta$ cor")

# %%
pairs_oi = [(row.window1, row.window2) for _, row in ranking.iloc[-50:-1].iterrows()][::-1]
plotdata_random = random.sel(window1_window2 = pairs_oi)
plotdata_random = plotdata_random.to_pandas().unstack().to_frame("interaction").reset_index()
plotdata_random["x"] = np.repeat(np.arange(len(pairs_oi)), 1000)

# %%
fig, ax = plt.subplots(figsize = (10, 2))
sns.boxplot(x = plotdata_random["x"], y = plotdata_random["interaction"], ax = ax, showfliers = False, whis = [0.025, 0.975])
ax.set_ylabel("cor $\Delta$ cor")
ax.set_xlabel("Co-predictive window pair (ordered by cor $\Delta$ cor)")
ax.set_xticks([])
