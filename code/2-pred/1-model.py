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
import polyptich as pp
pp.setup_ipython()

import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import tqdm.auto as tqdm

import polyptich as pp

# %%
from elabjournal import elabjournal

# %%
api = elabjournal.api(key="vib.elabjournal.com;b4d3038d935dd415f4f1cc327be56876")

# %%
from elabjournal import elabjournal
api = elabjournal.api(key="API_KEY")
study = api.study(id = 227030)
experiment = api.experiment(id = 1305284)
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 2, 1])
ax.set_xlabel("Hours adding stuff to ELN")
ax.set_ylabel("Hours doing science")
experiment.add(fig)

# %%

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k/subsets/top5"
# dataset_name = "pbmc10k/subsets/top1"
# dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
regions_name = "100k100k"
# regions_name = "500k500k"
# regions_name = "10k10k"

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")
# folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
# gene_oi = fragments.var.index[20]
# gene_oi = transcriptome.gene_id("BCL2")
# gene_oi = transcriptome.gene_id("IRF1")
# gene_oi = transcriptome.gene_id("TNFAIP2")
# gene_oi = transcriptome.gene_id("KLF12")
gene_oi = transcriptome.gene_id("IRF1")
# gene_oi = transcriptome.gene_id("IGSF8")
# gene_oi = "ENSG00000165995"
gene_ix = transcriptome.var.index.get_loc(gene_oi)

# %% [markdown]
# ## Single model

# %%
from chromatinhd_manuscript.pred_params import params

# %%
torch.manual_seed(1)

import chromatinhd.models.pred.model.better
# model_params = dict(
#     n_frequencies=(1000, 500, 250, 125, 63, 31),
#     n_embedding_dimensions=100,
#     n_layers_fragment_embedder=1,
#     residual_fragment_embedder=True,
#     n_layers_embedding2expression=0,
#     residual_embedding2expression=True,
#     layernorm_embedding2expression=True,
#     dropout_rate_fragment_embedder=0.0,
#     dropout_rate_embedding2expression=0.0,
#     encoder="radial_binary",
#     nonlinear="silu",
#     # library_size_encoder="linear",
#     library_size_encoder=None,
# )

# model_params=dict(
#     n_frequencies=(1000, 500, 250, 125, 63, 31),
#     n_embedding_dimensions=100,
#     n_layers_fragment_embedder=1,
#     residual_fragment_embedder=True,
#     n_layers_embedding2expression=5,
#     residual_embedding2expression=True,
#     layernorm_embedding2expression=True,
#     dropout_rate_fragment_embedder=0.0,
#     dropout_rate_embedding2expression=0.0,
#     encoder="radial_binary",
#     nonlinear="silu",
#     library_size_encoder="linear",
#     library_size_encoder_kwargs = dict(scale = 0.5),
#     # library_size_encoder=None,
#     distance_encoder="direct",
# )

# train_params = dict(
#     weight_decay=1e-1,
#     lr=1e-4,
# )

model_params = params["v33"]["model_params"]
train_params = params["v33"]["train_params"]

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
model.train_model()

# %%
prediction = model.get_prediction(cell_ixs = fold["cells_train"]).sel(gene = gene_oi)
np.corrcoef(prediction["predicted"], prediction["expected"])[0, 1]

# %%
predicted = prediction["predicted"].values
predicted 

# %%
predicted = (((predicted - predicted.mean()) / predicted.std()) + (expected.mean())) * expected.std()

# %%
expected = prediction["expected"].values
expected = (expected - expected.mean()) / expected.std()

# %%
mse_m = (1/len(predicted))  * ((predicted - expected)**2).sum()

# %%
mse_bmk = (1/len(expected))  * ((expected - expected.mean())**2).sum()

# %%
1-mse_m/mse_bmk

# %%
(np.corrcoef(predicted, expected)[0, 1])**2

# %%
models = chd.models.pred.model.better.Models()
models[gene_oi + "_0"] = model
models.folds = folds

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes = (100, ))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(folds, transcriptome, fragments, censorer, path = chd.get_output() / "test", overwrite = True)

# %%
regionmultiwindow.score(models, regions = [gene_oi])

# %%
regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").mean("fold").to_pandas().plot()

# %%
censorer = chd.models.pred.interpret.SizeCensorer(20)
censorer.__class__ = chd.models.pred.interpret.censorers.SizeCensorer
import chromatinhd.models.pred.interpret.censorers
size = chd.models.pred.interpret.Size.create(folds, transcriptome, fragments, censorer, path = chd.get_output() / "test2", overwrite = True)
size.score(models, regions = [gene_oi])

# %%
size.scores["deltacor"].sel_xr(gene_oi).mean("fold").sel(phase = "test").plot()

# %%
windows_oi = regionmultiwindow.design.query("window_size == 100").index
regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).mean("phase").mean("fold").to_pandas().plot()

# %% [markdown]
# ## Models

# %%
gene_oi = transcriptome.gene_id("IRF1")
# gene_oi = transcriptome.gene_id("FOXN2")
# gene_oi = transcriptome.gene_id("ANXA2R")
# gene_oi = transcriptome.gene_id("CCL4")
# gene_oi = transcriptome.gene_id("TMTC2")
# gene_oi = transcriptome.gene_id("ADAM12")

# %%
models = chd.models.pred.model.better.Models.create(
    fragments = fragments,
    transcriptome = transcriptome,
    folds = folds,
    layer = "magic",
    regions_oi = [gene_oi],
    model_params = {**model_params, "library_size_encoder": None},
    train_params = train_params,
    reset = True
)
for model in models:
    model.layer = "magic"
models.train_models()

# %%
prediction = np.zeros(len(transcriptome.obs.index))
expected = np.zeros(len(transcriptome.obs.index))
n_fragments = np.zeros(len(transcriptome.obs.index))
for fold_ix in range(25):
    models[gene_oi + "_" + str(fold_ix)].layer = "magic"
    predicted = models[gene_oi + "_" + str(fold_ix)].get_prediction(cell_ixs = folds[fold_ix]["cells_test"]).sel(gene = gene_oi)
    predicted["predicted"] = (((predicted["predicted"] - predicted["predicted"].mean()) / predicted["predicted"].std()) + (predicted["expected"].mean())) * predicted["expected"].std()
    prediction[folds[fold_ix]["cells_test"]] = predicted["predicted"].values
    expected[folds[fold_ix]["cells_test"]] = predicted["expected"].values
    n_fragments[folds[fold_ix]["cells_test"]] = predicted["n_fragments"].values

# %%
import chromatinhd.models.pred.model.peakcounts
creprediction = chd.models.pred.model.peakcounts.Prediction(chd.get_output() / "test", reset=True)
peakcounts = chd.flow.Flow.from_path(
    chd.get_output() / "datasets" / dataset_name / "peakcounts" / "macs2_leiden_0.1_merged" / "100k100k"
    # chd.get_output() / "datasets" / dataset_name / "peakcounts" / "macs2_improved" / "100k100k"
)
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
creprediction.initialize(peakcounts, transcriptome, folds)

# %%
prediction2 = np.zeros(len(transcriptome.obs.index))
for fold_ix in range(5):
    cre_predicted, cre_expected = creprediction.get_prediction(gene_oi = gene_oi, layer = "magic", predictor = "lasso", fold_ix = fold_ix)
    cre_predicted = (((cre_predicted - cre_predicted.mean()) / cre_predicted.std()) + (cre_expected.mean())) * cre_expected.std()
    prediction2[folds[fold_ix]["cells_test"]] = cre_predicted

# %%
# prediction = (prediction  / prediction.std() * expected.std()) - prediction.mean() + expected.mean()
# prediction = prediction - prediction.mean() + expected.mean()
# prediction2 = (prediction2  / prediction2.std() * expected.std())
# prediction2 = prediction2 - prediction2.mean() + expected.mean()

# %%
gene_oi

# %%
zero = n_fragments == 0
print(zero.mean())

# %%
cors = []
for i in range(200):
    bootstrap = np.random.choice(len(prediction), len(prediction), replace = True)
    cors.append(np.corrcoef(prediction[bootstrap], expected[bootstrap])[0, 1])
cors2 = []
for i in range(200):
    bootstrap = np.random.choice(len(prediction2), len(prediction2), replace = True)
    cors2.append(np.corrcoef(prediction2[bootstrap], expected[bootstrap])[0, 1])

# %%
cor = np.corrcoef(prediction, expected)[0, 1]
cor2 = np.corrcoef(prediction2, expected)[0, 1]
cor, cor2

# %%
np.quantile(cors, (0.025, 0.975))

# %%
cor_q05 = np.quantile(cors, 0.025)
cor_q95 = np.quantile(cors, 0.975)

cor2_q05 = np.quantile(cors2, 0.025)
cor2_q95 = np.quantile(cors2, 0.975)

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (3.5, 1.75))
ax0.scatter(expected, prediction, s = 1, color = "#333")
ax0.scatter(expected[zero], prediction[zero], s = 1, color = "grey")
ax1.scatter(expected, prediction2, s = 1, color = "#333")
ax1.scatter(expected[zero], prediction2[zero], s = 1, color = "grey")
for fold in folds[:5]:
    sns.regplot(y = prediction[fold["cells_test"]], x = expected[fold["cells_test"]], ax = ax0, scatter_kws = dict(s = 0., color = "grey"), ci = None)
ax0.set_title("ChromatinHD-pred", fontsize = 9)
text = ax0.annotate(f"r = {cor:.2f}\n95%: {cor_q05:.2f},{cor_q95:.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")
text.set_path_effects([mpl.patheffects.withStroke(linewidth = 3, foreground = "white")])
# sns.regplot(y = prediction2, x = expected, ax = ax1, scatter_kws = dict(s = 0., color = "grey"))
ax1.set_title("MACS2 per celltype merged\n + Lasso", fontsize = 9)
for fold in folds[:5]:
    sns.regplot(y = prediction2[fold["cells_test"]], x = expected[fold["cells_test"]], ax = ax1, scatter_kws = dict(s = 0., color = "grey"), ci = None)
text = ax1.annotate(f"r = {cor2:.2f}\n95%: {cor2_q05:.2f},{cor2_q95:.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")
text.set_path_effects([mpl.patheffects.withStroke(linewidth = 3, foreground = "white")])
ax0.set_xlabel("Observed\n(MAGIC normalized expression)")
ax0.set_ylabel("Predicted")
ax1.set_ylim(ax0.get_ylim())
ax1.set_yticks([])

manuscript.save_figure(fig, "2", f"prediction_scatterplot_{transcriptome.symbol(gene_oi)}")

# %%
censorer.design

# %%
# # %%timeit -r 1 -n 1
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes=(25, 50, 100, 200, 500))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
    folds, transcriptome, fragments, censorer, path=chd.get_output() / "test", overwrite=True
)

regionmultiwindow.score(models, regions=[gene_oi], device = "cuda:0")

# %%
# import cProfile

# stats = cProfile.run('regionmultiwindow.score(models, regions=[gene_oi], device = "cuda:0")', "restats")
# import pstats

# p = pstats.Stats("restats")
# p.sort_stats("cumulative").print_stats()

# %%
regionmultiwindow.interpolate([gene_oi], force = True)

# %%
plt.plot(regionmultiwindow.interpolation["deltacor"][gene_oi])

# %%
self = regionmultiwindow

# %%
region_id = gene_oi

# %%
lost_cutoff = 1.0

# %%
plotdata = self.get_plotdata(region_id)
selection = pd.DataFrame({"chosen": (plotdata["lost"] > lost_cutoff)})

# %%
max_merge_distance=100
min_length=50
padding=50
lost_cutoff=0.02
lost_cutoff=0.05

# %%
regions = regionmultiwindow.select_regions(region_id, padding = padding, lost_cutoff = lost_cutoff)
regions

# %%
symbol = transcriptome.var.loc[gene_oi, "symbol"]

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

score_panel_width = 0.8

breaking = polyptich.grid.broken.Breaking(regions, 0.05)

region = fragments.regions.coordinates.loc[region_id]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking,
    genome="GRCh38",
    label_positions=True,
    # use_cache = False,
)
fig.main.add_under(panel_genes)

panel_pileup = fig.main.add_under(
    chd.models.pred.plot.PileupBroken.from_regionmultiwindow(regionmultiwindow, region_id, breaking, height = 0.5)
)
panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow, region_id, breaking, height = 0.5, ymax = -0.1)
)

fig.plot()

# %%
regions = pd.DataFrame({
    "start":[32950, 32750, 30000],
    "end":[33200, 34000, 35000],
    "resolution":[250, 500, 1000]
})
regions["resolution"] = (regions["end"] - regions["start"])

# %%
symbol = transcriptome.var.loc[region_id, "symbol"]
breaking = polyptich.grid.broken.Breaking(regions, 0.05)

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

region = fragments.regions.coordinates.loc[region_id]
# panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
#     region,
#     breaking,
#     genome="GRCh38",
#     label_positions=True,
#     # use_cache = False,
# )
# fig.main.add_under(panel_genes)

panel_pileup = fig.main.add_under(
    chd.models.pred.plot.PileupBroken.from_regionmultiwindow(regionmultiwindow, region_id, breaking, height = 0.5)
)

panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow, region_id, breaking, height = 0.5, ymax = -0.1)
)
fig.plot()

manuscript.save_figure(fig, "1", "chromatinhd_pred")

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name, add_rolling = True)
peakcallers["label"] = chdm.peakcallers["label"]
panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
    fragments.regions.coordinates.loc[region_id], peakcallers, breaking, label_rows = False, label_methods_side = "left"
)
fig.main.add_under(panel_peaks)

fig.plot()

manuscript.save_figure(fig, "1", "peaks")

# %%
regions

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

panel_genes = chd.plot.genome.genes.Genes.from_region(
    fragments.regions.coordinates.loc[region_id],
    window = [-500, 40000],
    genome="GRCh38",
    width = 3.5,
    annotate_tss = False,
)
panel_genes.ax.axhline(-0.5, color = "grey", lw = 1)

panel_genes.dim = (panel_genes.dim[0], 0.5)
ax = panel_genes.ax
ax.set_ylim(3.5, -0.5)
for region_ix, region in regions.iterrows():
    rect = mpl.patches.Rectangle((region["start"], region_ix+0.5), region["end"] - region["start"], 1, fc = "#333")
    ax.add_patch(rect)
    
fig.main.add_under(panel_genes)
fig.plot()

manuscript.save_figure(fig, "1", "gene")

# %% [markdown]
# ## Pairs

# %%
from scipy.ndimage import convolve
def spread_true(arr, width=5):
    kernel = np.ones(width, dtype=bool)
    result = convolve(arr, kernel, mode="constant", cval=False)
    result = result != 0
    return result



# %%
regionpairwindow = chd.models.pred.interpret.RegionPairWindow(models.path / "scoring" / "regionpairwindow")
for gene_oi in [gene_oi]:
    windows_oi = regionmultiwindow.design.query("window_size == 200").index
    # windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]

    windows_selected = (regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-3)
    windows_selected = spread_true(windows_selected, width = 5)

    windows_oi = windows_oi[regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-4]
    # windows_oi = windows_oi
    design = regionmultiwindow.censorer.design.loc[["control"] + windows_oi.tolist()]

    censorer = chd.models.pred.interpret.censorers.WindowCensorer(fragments.regions.window)
    censorer.design = design
    design.shape

    regionpairwindow.score(models, regions = [gene_oi], censorer = censorer)

# %%
plotdata = regionpairwindow.interaction[gene_oi].mean("fold")

# %%
import chromatinhd.data.associations
associations = chd.data.associations.Associations(
    chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / "gwas_immune_main"
    # chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / "gwas_immune"
)

# %%
symbol = transcriptome.var.loc[gene_oi, "symbol"]

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.0, padding_width=0.05))

score_panel_width = 0.8

breaking = polyptich.grid.broken.Breaking(regions, 0.02, resolution = 7500)

region = fragments.regions.coordinates.loc[region_id]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking,
    genome="GRCh38",
    label_positions=True,
    label_positions_minlength = 500,
    label_positions_rotation = 90,
    # use_cache = False,
)
fig.main.add_under(panel_genes)

panel_associations = fig.main.add_under(
    chd.data.associations.plot.AssociationsBroken(associations, region_id, breaking, show_ld = True),
    padding = 0.1
)

# panel_pileup = fig.main.add_under(
#     chd.models.pred.plot.PileupBroken.from_regionmultiwindow(regionmultiwindow, region_id, breaking, height = 0.5)
# )
panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow, region_id, breaking, height = 0.5, ymax = -0.1)
)

fig.plot()

# %%
plotdata_windows = regionpairwindow.scores[gene_oi].mean("fold").to_dataframe()
plotdata_interaction = regionpairwindow.interaction[gene_oi].mean("fold").to_pandas().unstack().to_frame("cor")

# %%
windows = design.loc[design.index != "control"]

# %%
plotdata = plotdata_interaction.copy()
import itertools

# make plotdata, making sure we have all window combinations, otherwise nan
plotdata = (
    pd.DataFrame(
        itertools.combinations(windows.index, 2), columns=["window1", "window2"]
    )
    .set_index(["window1", "window2"])
    .join(plotdata_interaction)
)
plotdata.loc[np.isnan(plotdata["cor"]), "cor"] = 0.0
plotdata["dist"] = windows.loc[plotdata.index.get_level_values(
    "window2"
), "window_mid"].values - windows.loc[plotdata.index.get_level_values(
    "window1"
), "window_mid"].values

transform = polyptich.grid.broken.TransformBroken(breaking)
plotdata["window1_broken"] = transform(
    windows.loc[plotdata.index.get_level_values(
    "window1"
), "window_mid"].values
)
plotdata["window2_broken"] = transform(
    windows.loc[plotdata.index.get_level_values(
    "window2"
), "window_mid"].values
)

plotdata = plotdata.loc[
    ~pd.isnull(plotdata["window1_broken"]) & ~pd.isnull(plotdata["window2_broken"])
]

plotdata.loc[plotdata["dist"] < 1000, "cor"] = 0.

# %%
import chromatinhd.data.associations

# %%
panel_interaction = fig.main.add_under(
    polyptich.grid.Panel((breaking.width, breaking.width / 2)), padding=0.0
)
ax = panel_interaction.ax

norm = mpl.colors.CenteredNorm(0, np.abs(plotdata["cor"]).max())
cmap = mpl.cm.RdBu_r

chd.plot.matshow45(
    ax,
    plotdata.query("dist > 0").set_index(["window1_broken", "window2_broken"])["cor"],
    cmap=cmap,
    norm=norm,
    radius=50,
)
ax.invert_yaxis()

if symbol in ["BCL2"]:
    panel_interaction_legend = panel_interaction.add_inset(polyptich.grid.Panel((0.05, 0.8)), pos = (-0.1, 0.5))
    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=panel_interaction_legend.ax,
        orientation="vertical",
        ticks = [-0.1, 0., 0.1]
    )
    panel_interaction_legend.ax.set_ylabel(
        "Co-predictivity\n(cor)",
        rotation=0,
        ha="left",
        va="center",
    )

ax.set_xlim([transform(regions["start"].min()), transform(regions["end"].max())])
fig.plot()
fig

manuscript.save_figure(fig, "5", "interaction_examples", symbol)

# %%
fig

# %%
fig_colorbar = plt.figure(figsize=(3.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=norm, cmap = cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal")
colorbar.set_label("Co-predictivity")
colorbar.set_ticks([colorbar.vmin, 0., colorbar.vmax])
colorbar.set_ticklabels(["min", "0", "max"])
manuscript.save_figure(fig_colorbar, "5", "colorbar_copredictivity")

# %% [markdown]
# ### Pairs consistency

# %%
transcriptome2 = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k_gran" / "transcriptome")
fragments2 = chd.data.Fragments(chd.get_output() / "datasets" / "pbmc10k_gran" / "fragments" / "100k100k")
folds2 = chd.data.folds.Folds(chd.get_output() / "datasets" / "pbmc10k_gran" / "folds" / "5x5")

regionpairwindow2 = chd.models.pred.interpret.RegionPairWindow(models.path / "scoring" / "regionpairwindow_test", reset = True)
for gene_oi in [gene_oi]:
    windows_oi = regionmultiwindow.design.query("window_size == 200").index
    # windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]

    windows_selected = (regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-3)
    windows_selected = spread_true(windows_selected, width = 5)

    windows_oi = windows_oi[regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-4]
    design = regionmultiwindow.censorer.design.loc[["control"] + windows_oi.tolist()]

    censorer = chd.models.pred.interpret.censorers.WindowCensorer(fragments.regions.window)
    censorer.design = design
    design.shape

    regionpairwindow2.score(models, regions = [gene_oi], censorer = censorer, transcriptome = transcriptome2, fragments = fragments2, folds = folds2)

# %%
plotdata_windows = regionpairwindow.scores[gene_oi].mean("fold").to_dataframe()
plotdata_interaction = regionpairwindow.interaction[gene_oi].mean("fold").to_pandas().unstack().to_frame("cor")
windows = design.loc[design.index != "control"]
plotdata = plotdata_interaction.copy()
import itertools

# make plotdata, making sure we have all window combinations, otherwise nan
plotdata = (
    pd.DataFrame(
        itertools.combinations(windows.index, 2), columns=["window1", "window2"]
    )
    .set_index(["window1", "window2"])
    .join(plotdata_interaction)
)
plotdata.loc[np.isnan(plotdata["cor"]), "cor"] = 0.0
plotdata["dist"] = windows.loc[plotdata.index.get_level_values(
    "window2"
), "window_mid"].values - windows.loc[plotdata.index.get_level_values(
    "window1"
), "window_mid"].values

transform = polyptich.grid.broken.TransformBroken(breaking)
plotdata["window1_broken"] = transform(
    windows.loc[plotdata.index.get_level_values(
    "window1"
), "window_mid"].values
)
plotdata["window2_broken"] = transform(
    windows.loc[plotdata.index.get_level_values(
    "window2"
), "window_mid"].values
)

plotdata = plotdata.loc[
    ~pd.isnull(plotdata["window1_broken"]) & ~pd.isnull(plotdata["window2_broken"])
]

plotdata.loc[plotdata["dist"] < 1000, "cor"] = 0.

# %%
plotdata_windows = regionpairwindow.scores[gene_oi].mean("fold").to_dataframe()
plotdata_interaction2 = regionpairwindow2.interaction[gene_oi].mean("fold").to_pandas().unstack().to_frame("cor")
plotdata["cor2"] = plotdata_interaction2["cor"]

plotdata = plotdata.loc[~np.isclose(plotdata["cor"], 0, atol = 1e-7) & ~np.isclose(plotdata["cor2"], 0., atol = 1e-7)].copy()

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["cor"], plotdata["cor2"], s = 1, alpha = 0.1)
ax.set_xlabel("co-predictivity pbmc10k")
ax.set_ylabel("co-predictivity pbmc10k_gran")

# %%
plotdata_unstacked = plotdata["cor"].unstack()
plotdata2_unstacked = plotdata["cor2"].unstack().T

# %%
fig, ax = plt.subplots(figsize = (8, 8))

norm = mpl.colors.CenteredNorm(0, np.abs(plotdata["cor"]).max())
cmap = mpl.cm.RdBu_r

ax.matshow(plotdata_unstacked, norm = norm, cmap = cmap)
ax.matshow(plotdata2_unstacked, norm = norm, cmap = cmap)
ax.axline((0, 0), slope = 1, color = "white")
ax.set_xlim(0, 550)
ax.set_ylim(550, 0)

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

panel_interaction = fig.main.add_under(
    polyptich.grid.Panel((breaking.width, breaking.width / 2)), padding=0.0
)
ax = panel_interaction.ax

norm = mpl.colors.CenteredNorm(0, np.abs(plotdata["cor"]).max())
cmap = mpl.cm.RdBu_r

chd.plot.matshow45(
    ax,
    plotdata.set_index(["window1_broken", "window2_broken"])["cor"],
    cmap=cmap,
    norm=norm,
    radius=50,
)

panel_interaction = fig.main.add_under(
    polyptich.grid.Panel((breaking.width, breaking.width / 2)), padding=0.0
)
ax = panel_interaction.ax

norm = mpl.colors.CenteredNorm(0, np.abs(plotdata["cor"]).max())
cmap = mpl.cm.RdBu_r

chd.plot.matshow45(
    ax,
    plotdata.set_index(["window1_broken", "window2_broken"])["cor2"],
    cmap=cmap,
    norm=norm,
    radius=50,
)
ax.invert_yaxis()

ax.set_xlim([transform(regions["start"].min()), transform(regions["end"].max())])
fig.plot()

# %% [markdown]
# ## Window size

# %%
import chromatinhd.models.pred.model.better

# %%
models = chd.models.pred.model.better.Models(
    chd.get_output() / "pred/pbmc10k/100k100k/5x5/magic/v33"
)

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes=(100,))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
    folds,
    transcriptome,
    fragments,
    censorer,
    path=models.path / "scoring" / "regionmultiwindow_100",
    # overwrite=True,
)
print(regionmultiwindow.scores["scored"].sel_xr().all("fold").sum())

for gene_oi in tqdm.tqdm(transcriptome.var.index[::-1]):
    if models.trained(gene_oi):
        if not regionmultiwindow.scores["scored"].sel_xr(gene_oi).all():
            regionmultiwindow.score(models, regions=[gene_oi])

# %%
gene_oi = transcriptome.gene_id("CCL4")

# %%
extrema_interleaved = [10, 110, 170, 270, 390, 470, 590, 690, 770]
cuts = [0, *(extrema_interleaved[:-1] + np.diff(extrema_interleaved) / 2), 99999]
sizes = pd.DataFrame(
    {
        "start": cuts[:-1],
        "end": cuts[1:],
        "length": np.diff(cuts),
        "mid": [*(cuts[:-2] + np.diff(cuts)[:-1] / 2), cuts[-2] + 10],
    }
)

sizes = pd.DataFrame({
    "start":[0, 60],
    "end":[60, 140],
    "length":[60, 80],
    "mid":[30, 100]
})

# %%
regionsizewindow = chd.models.pred.interpret.RegionSizeWindow(models.path / "scoring" / "regionsizewindow")

# %%
regionmultiwindow.score(models, regions=[gene_oi], force = True)

# %%
# force = True
force = False
for gene_oi in tqdm.tqdm(transcriptome.var.index[::-1]):
# for gene_oi in [transcriptome.gene_id("CCL4")]:
    if regionmultiwindow.scores["scored"].sel_xr(gene_oi).all():
        if force or (gene_oi not in regionsizewindow.scores):
            windows_oi = regionmultiwindow.design.query("window_size == 100").index
            # windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]
            windows_oi = windows_oi[
                (regionmultiwindow.scores["deltacor"]
                .sel_xr(gene_oi)
                .sel(phase = "test")
                .sel(window=windows_oi.tolist())
                .mean("fold")
                < -0.0001
                )
            ]
            print(len(windows_oi))
            if len(windows_oi) > 0:
                design_windows = regionmultiwindow.censorer.design.loc[windows_oi.tolist()]
                design_windows

                censorer = chd.models.pred.interpret.censorers.WindowSizeCensorer(design_windows, sizes)
                regionsizewindow.score(models, regions = [gene_oi], censorer = censorer, device = "cpu", force = force)

# %%
gene_oi = transcriptome.gene_id("CCL4")

# %%
scores = regionsizewindow.scores[gene_oi]
windows = scores.coords["window"].to_pandas().str.split("_").str[0][::2]
deltacor = pd.DataFrame(scores["deltacor"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])
lost = pd.DataFrame(scores["lost"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])

score = pd.concat([deltacor.unstack(), lost.unstack()], axis=1, keys=["deltacor", "lost"])
score.index.names = ["size", "window"]
score["deltacor_full"] = (regionmultiwindow.scores["deltacor"]
                .sel_xr(gene_oi)
                .sel(phase = "test")
                .sel(window= score.index.get_level_values("window")).mean("fold"))

# %%
score

# %%
(deltacor[100] == 0).mean()

# %%
sns.heatmap(score["deltacor"].unstack())

# %%
x = (regionmultiwindow.scores["deltacor"]
                .sel_xr(gene_oi)
                .sel(phase = "test")
                .sel(window=windows_oi.tolist())
                .mean("fold")
)
y = regionsizewindow.scores[gene_oi]["deltacor"].values.mean(0).reshape((-1, 2))
y_ = y[:, 0]

# %%
plt.scatter(x, y_)

# %%
sns.heatmap(y)

# %%
sns.heatmap(regionsizewindow.scores[gene_oi]["deltacor"].values)

# %%
regionsizewindow = chd.models.pred.interpret.RegionSizeWindow(models.path / "scoring" / "regionsizewindow")

# for gene_oi in [gene_oi]:
#     windows_oi = regionmultiwindow.design.query("window_size == 200").index
#     # windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]

#     windows_selected = (regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-3)
#     windows_selected = spread_true(windows_selected, width = 5)

#     windows_oi = windows_oi[regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-4]
#     # windows_oi = windows_oi
#     design = regionmultiwindow.censorer.design.loc[["control"] + windows_oi.tolist()]

#     censorer = chd.models.pred.interpret.censorers.WindowCensorer(fragments.regions.window)
#     censorer.design = design
#     design.shape

#     regionpairwindow.score(models, regions = [gene_oi], censorer = censorer)

# %%
censorer = chd.models.pred.interpret.censorers.WindowSizeCensorer(design_windows, sizes)
regionsizewindow.score(models, regions = [gene_oi], censorer = censorer, force = True, device = "cpu")

# %%

# %%
windows_oi = regionmultiwindow.design.query("window_size == 100").index
# windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]
windows_oi = windows_oi[
    regionmultiwindow.scores["deltacor"]
    .sel_xr(gene_oi)
    .mean("phase")
    .sel(window=windows_oi.tolist())
    .mean("fold")
    < -0.0001
]
design_windows = regionmultiwindow.censorer.design.loc[windows_oi.tolist()]
design_windows

# %%
sns.heatmap(regionsizewindow.scores[gene_oi].mean("fold")["deltacor"].values.reshape(-1, 2))

# %%
clustering = chd.data.clustering.Clustering.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / "leiden_0.1")

# %% [markdown]
# ## Positional encodings

# %%
import chromatinhd.models.pred.model.better

# %%
window = [-100000, 100000]

# %%
# sine_encoding = chromatinhd.models.pred.model.encoders.TophatEncoding(1000, window = window)
# sine_encoding = chromatinhd.models.pred.model.encoders.TophatBinaryEncoding(n_frequencies=(1000, 500, 250, 125, 63, 31), window = window)
# sine_encoding = chromatinhd.models.pred.model.encoders.SplineBinaryEncoding(window = window) # NOPE

# name = "radial_binary"
# sine_encoding = chromatinhd.models.pred.model.encoders.RadialBinaryEncoding(
#     n_frequencies=(1000, 500, 250, 125, 63, 31, 15), window=window, scale = 1.
# )

# sine_encoding = chromatinhd.models.pred.model.encoders.RadialEncoding(n_frequencies= 1000, window = window)

# name = "sine"
# sine_encoding = chromatinhd.models.pred.model.encoders.SineEncoding(n_frequencies= 20)

# sine_encoding = chromatinhd.models.pred.model.encoders.SineEncoding3(n_frequencies= 10)
# sine_encoding = chromatinhd.models.pred.model.encoders.DirectEncoding(window = window)

name = "spline_binary_full"
sine_encoding = chromatinhd.models.pred.model.encoders.SplineBinaryFullEncoding(
    window=window,
)

# %%
window=(-100000, 100000)


# %%
bw = 100

# %%
import math

# %%
import torch
coordinates = torch.tensor([
    np.arange(-100000, 100000, 100), 
    np.arange(-100000, 100000, 100)+200, 
]).T

# %%
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
cmap = mpl.cm.RdBu_r

# %%
encoding = sine_encoding(coordinates).detach().numpy()

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
main = fig.main

panel, ax = main.add_under(polyptich.grid.Panel((7, 2)))
ax.matshow(encoding, norm = norm, cmap = cmap, aspect = "auto")
ax.set_xlabel("Dimensions")
ax.xaxis.tick_bottom()
ax.set_ylabel("Fragments")
ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 1998])
ax.set_yticklabels([
    f"{coordinates[int(i), 0]} ; {coordinates[int(i), 1]}" for i in ax.get_yticks()
])
fig.plot()

manuscript.save_figure(fig, "2", "snote", "encoders", name + "_all")

# %%
import torch
coordinates = torch.tensor([
    [-20000, -19910],
    [5, 35],
    [85000, 85650]  
])

# %%
encoding = sine_encoding(coordinates).detach().numpy()

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
main = fig.main

panel, ax = main.add_under(polyptich.grid.Panel((8, 2)))
ax.matshow(encoding, norm = norm, cmap = cmap, aspect = "auto")
ax.set_xlabel("Dimensions")
ax.xaxis.tick_bottom()
ax.set_ylabel("Fragments")
fig.plot()

manuscript.save_figure(fig, "2", "snote", "encoders", name + "_subset")

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
main = fig.main

encoding = sine_encoding(coordinates).detach().numpy()

for i in range(encoding.shape[0]):
    panel, ax = main.add_under(polyptich.grid.Panel((8, 0.2)), padding = 0.35)
    encoding_oi = encoding[i]
    encoding_selection =  np.abs(encoding_oi) > 0.01
    encoding_oi = encoding_oi[encoding_selection]
    ax.matshow(encoding_oi.reshape(1, -1), norm = norm, cmap = cmap, aspect = "auto")
    ax.set_xticks(np.arange(len(encoding_oi)))
    ax.set_xticklabels(np.where(encoding_selection)[0], rotation = 90, fontsize = 7, zorder = 200)
    chosen = np.where(encoding_selection)[0]
    nonconseuctive = np.diff(chosen) > 1
    for j in np.where(nonconseuctive)[0]:
        line = mpl.lines.Line2D([j+0.5, j+0.5], [-0.55, 0.55], color = "white", linewidth = 3., clip_on = False, zorder = 100)
        ax.add_line(line)
        # ax.axvline(j+0.5, color = "white", linewidth = 1.5)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, pad = 0)
    ax.set_yticks([])
    ax.set_ylabel(f"{coordinates[i, 0]} ; {coordinates[i, 1]}", rotation = 0, ha = "right", va = "center")

ax.set_xlabel("Dimensions")

fig.plot()

manuscript.save_figure(fig, "2", "snote", "encoders", name + "_subset_nonzero")

# %%
