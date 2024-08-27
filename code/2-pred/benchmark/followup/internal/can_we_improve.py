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

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
from chromatinhd_manuscript.designs_pred import (
    dataset_folds_peakcaller_predictor_combinations as design_peaks,
)

# design_peaks = design_peaks.loc[design_peaks["predictor"] != "xgboost"].copy()

from chromatinhd_manuscript.designs_pred import (
    dataset_folds_method_combinations as design_methods,
)

from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_folds_method_combinations as design_methods_traintest,
)

design_methods_traintest["dataset"] = design_methods_traintest["testdataset"]
design_methods_traintest["folds"] = "all"

from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_folds_peakcaller_predictor_combinations as design_peaks_traintest,
)

design_peaks_traintest["dataset"] = design_peaks_traintest["testdataset"]
design_peaks_traintest["folds"] = "all"

# %%
design_peaks["method"] = design_peaks["peakcaller"] + "/" + design_peaks["predictor"]
design_peaks_traintest["method"] = (
    design_peaks_traintest["peakcaller"] + "/" + design_peaks_traintest["predictor"]
)
# design_peaks = design_peaks.loc[design_peaks["predictor"] == "lasso"]

# %%
from chromatinhd_manuscript.designs_pred import dataset_folds_baselinemethods_combinations as design_baseline
from chromatinhd_manuscript.designs_pred import dataset_folds_simulation_combinations as design_simulated


# %%
design = pd.concat(
    [design_peaks, design_methods, design_methods_traintest, design_peaks_traintest, design_baseline, design_simulated]
)
design.index = np.arange(len(design))
design.index.name = "design_ix"

# %%
# design = design.loc[((design["folds"].isin(["random_5fold", "all"])))]
design = design.loc[((design["folds"].isin(["5x5"])))]
# design = design.query("layer in ['magic']").copy()
# design = design.query("layer in ['normalized']").copy()
# design = design.query("regions in ['10k10k']").copy()
# design = design.query("regions in ['100k100k']").copy()
design = design.query("regions in ['10k10k', '100k100k']").copy()
design = design.loc[design["peakcaller"] != "stack"]

# %%
design["traindataset"] = [
    x["dataset"] if pd.isnull(x["traindataset"]) else x["traindataset"]
    for _, x in design.iterrows()
]

# %%
assert not design[[col for col in design.columns if not col in ["params"]]].duplicated(keep = False).any(), "Duplicate designs"

# %%
scores = {}
design["found"] = False
for design_ix, design_row in design.iterrows():
    prediction = chd.flow.Flow(
        chd.get_output()
        / "pred"
        / design_row["dataset"]
        / design_row["regions"]
        / design_row["folds"]
        / design_row["layer"]
        / design_row["method"]
    )
    if (prediction.path / "scoring" / "performance" / "genescores.pkl").exists():
        print(prediction.path)
        # print(prediction.path)
        genescores = pd.read_pickle(
            prediction.path / "scoring" / "performance" / "genescores.pkl"
        )

        if isinstance(genescores, xr.Dataset):
            genescores = genescores.mean("model").to_dataframe()

        genescores["design_ix"] = design_ix
        scores[design_ix] = genescores.reset_index()
        design.loc[design_ix, "found"] = True
scores = pd.concat(scores, ignore_index=True)
scores = pd.merge(design, scores, on="design_ix")

scores = scores.reset_index().set_index(
    ["method", "dataset", "regions", "layer", "phase", "gene"]
)
assert not scores.index.duplicated().any(), "scores index is not unique"

dummy_method = "baseline_v42"
scores["cor_diff"] = (
    scores["cor"] - scores.xs(dummy_method, level="method")["cor"]
).reorder_levels(scores.index.names)

design["found"].mean()

# %%
metric_ids = ["cor"]

group_ids = ["method", "dataset", "regions", "layer", "phase"]

meanscores = scores.groupby(group_ids)[[*metric_ids, "design_ix"]].mean()
diffscores = meanscores - meanscores.xs(dummy_method, level="method")
diffscores.columns = diffscores.columns + "_diff"
relscores = np.log(meanscores / meanscores.xs(dummy_method, level="method"))
relscores.columns = relscores.columns + "_rel"

scores_all = meanscores.join(diffscores).join(relscores)

# %%
dataset = "pbmc10k"
# dataset = "hspc"

layer = "normalized"
layer = "magic"

regions = design["regions"].iloc[0]
# regions = "10k10k"
regions = "100k100k"

phase = "test"
# phase = "validation"

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset / "fragments" / "100k100k")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset / "folds" / "5x5")
fold = folds[1]

cors = scores.xs(dataset, level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs(regions, level = "regions")["cor"].unstack().T
cmses = scores.xs(dataset, level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs(regions, level = "regions")["cmse"].unstack().T
ccors = cors * pd.Series(transcriptome.layers["normalized"][:].std(0), index = transcriptome.var.index)[cors.index].values[:, None]

# %%
ms = ["baseline_v42", "v20", "macs2_leiden_0.1_merged/linear"]
plotdata = pd.DataFrame({
    # m:scores.loc[m].loc["pbmc10k"].loc["10k10k"].loc["normalized"].loc["test"]["cor"] for m in ms
    m:scores.loc[m].loc["pbmc10k"].loc["100k100k"].loc["magic"].loc["test"]["cor"] for m in ms
    # m:scores.loc[m].loc["pbmc10k"].loc["100k100k"].loc["normalized"].loc["test"]["cor"] for m in ms
})
m1, m2 = "baseline_v42", "v20"
m1, m2 = "macs2_leiden_0.1_merged/linear", "v20"

fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(plotdata[m1], plotdata[m2], color = "#333", s = 1)
ax.scatter(plotdata[m1].mean(), plotdata[m2].mean(), color = "red", s = 10)
ax.plot([0, 1], [0, 1], color = "black", linestyle = "--")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(m1)
ax.set_ylabel(m2)

# %%
plotdata["diff"] = (plotdata["v20"] - plotdata["macs2_leiden_0.1_merged/linear"])
plotdata.sort_values("diff", ascending = True).head(20).style.bar()

# %%
# plt.scatter(np.log(transcriptome_original.var["nonzero"].loc[plotdata.index]), (plotdata["v20"] - plotdata["macs2_leiden_0.1_merged/linear"]))

# %%
transcriptome_original = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset / "transcriptome")
fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset / "fragments" / "100k100k")

# %%
# genes_oi = ["ENSG00000112655"]
# genes_oi = ["ENSG00000046653"]
# genes_oi = ["ENSG00000177917"] # a single peak upstream is important
# genes_oi = ["ENSG00000112902"]
# genes_oi = ["ENSG00000023839"]
# genes_oi = ["ENSG00000153157"]
genes_oi = [transcriptome_original.gene_id("CCL4")]
# genes_oi = [transcriptome_original.gene_id("IL1B")]
# genes_oi = [transcriptome_original.gene_id("QKI")]
# genes_oi = [transcriptome_original.gene_id("EBF1")]
# genes_oi = [transcriptome_original.gene_id("CD79A")]
# genes_oi = [transcriptome_original.gene_id("CD79B")]

# %%
transcriptome = transcriptome_original.filter_genes(genes_oi)
fragments = fragments_original.filter_regions(fragments_original.regions.filter(genes_oi))
fragments.create_regionxcell_indptr()

# %%
sns.ecdfplot(fragments.counts[:, 0])

# %%
plt.hist(transcriptome.layers["X"].flatten())

# %%
layer = "magic"
# layer = "normalized"

# %%
import chromatinhd.models.pred.model.better

# %%
encoding.locs

# %%
self = encoding

# %%
coordinates = x[0][:, None]

# %%
sns.heatmap(np.exp(-(((coordinates - self.locs)/self.scales) ** 2) / 2))

# %%
self.scales

# %%
sns.heatmap(np.abs(((coordinates - self.locs)/self.scales)))

# %%
sns.heatmap(x[0][:, None] - encoding.locs)

# %%
plt.plot(y[:, 0].detach().numpy())

# %%
# encoding = chd.models.pred.model.better.TophatEncoding(50)
# encoding = chd.models.pred.model.better.RadialEncoding(50)
# encoding = chd.models.pred.model.better.SineEncoding2(100)
# encoding = chd.models.pred.model.better.RadialEncoding(100)
encoding = chd.models.pred.model.better.RadialBinaryEncoding(100)
# encoding = model.fragment_embedder.encoder
import torch
x = torch.stack([
    # torch.arange(-1000, 1000),
    # torch.arange(-1000, 1000),
    torch.linspace(-100000, 100000, 200),
    torch.linspace(-100000, 100000, 200),
])

y = encoding(x.T)
sns.heatmap(y.detach().numpy())

# model.fragment_embedder.weight1.weight.data[:] = 0.
# model.embedding_to_expression.weight1.weight.data[:] = 0.

# %%
model = chd.models.pred.model.better.Model(
# model = chd.models.pred.model.additive.Model(
    fragments = fragments,
    transcriptome=transcriptome,
    fold = fold,
    layer = layer,
    n_frequencies = 100,
    # n_frequencies = "direct",
    n_embedding_dimensions=100,
    n_layers_fragment_embedder = 5,
    # residual_fragment_embedder = True,
    n_layers_embedding2expression = 5,
    # residual_embedding2expression = True,
    dropout_rate_fragment_embedder=0.,
    dropout_rate_embedding2expression=0.,
    encoder = "tophat",
)

# %%
import logging
chd.models.pred.trainer.trainer.logger.handlers = []
chd.models.pred.trainer.trainer.logger.propagate = False

# %%
model.train_model(
    lr = 1e-4,
    n_epochs = 1000,
    n_cells_step = 10000,
    checkpoint_every_epoch=1,
    # weight_decay = 1e-2,
)

# %%
model.trace.plot();

# %%
performance = chd.models.pred.interpret.Performance()
performance.score(fragments, transcriptome, [model], [fold])
performance.genescores.mean("model")["cor"].to_pandas()

# %%
gene_oi = genes_oi[0]
region = fragments.regions.coordinates.loc[genes_oi[0]]
dataset_name = "pbmc10k"

# %%
import chromatinhd.data.peakcounts
import sklearn.linear_model
peakcounts = chd.data.peakcounts.PeakCounts(
    path=chd.get_output() / "datasets" / dataset_name / "peakcounts" / "macs2_leiden_0.1_merged" / fragments_original.regions.path.name
    # path=chd.get_output() / "datasets" / dataset_name / "peakcounts" / "rolling_500" / fragments_original.regions.path.name
)
peak_ids = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi]["peak"]
# peak_ids = pd.Series([peak_ids[3]])
peak_ixs = peakcounts.var.loc[peak_ids, "ix"]
x = np.array(peakcounts.counts[:, peak_ixs].todense())
y = transcriptome.layers[layer][:, transcriptome.var.index == gene_oi][:, 0]

x_train = x[fold["cells_train"]]
x_validation = x[fold["cells_validation"]]
x_test = x[fold["cells_test"]]

y_train = y[fold["cells_train"]]
y_validation = y[fold["cells_validation"]]
y_test = y[fold["cells_test"]]

# %%
lm = sklearn.linear_model.LinearRegression()
# lm = sklearn.linear_model.LassoCV(n_alphas = 10)
# lm = sklearn.linear_model.RidgeCV(alphas = 10)
lm.fit(x_train, y_train)

y_predicted = lm.predict(x_train)
print(np.corrcoef(y_train, y_predicted)[0, 1])

y_predicted = lm.predict(x_validation)
print(np.corrcoef(y_validation, y_predicted)[0, 1])

y_predicted = lm.predict(x_test)
print(np.corrcoef(y_test, y_predicted)[0, 1])

# %%
peaks = peakcounts.peaks.loc[peak_ids + "_" + gene_oi]
peaks["coef"] = lm.coef_
peaks.style.bar(subset = "coef")

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(window = fragments.regions.window)
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
regionmultiwindow.score(fragments, transcriptome, [model], [fold], censorer)
regionmultiwindow.interpolate()

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height = 0))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 20000]
# window = [-10000, 10000] # TSS
# window = [-20000, -10000] # PFKFB3 enhancer

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width = 10, window = window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(chd.models.pred.plot.Predictivity(regionmultiwindow.get_plotdata(gene_oi), window = window, width = 10))

panel, ax = fig.main.add_under(chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = fragments.regions.window, width = 10, peakcallers = ["macs2_leiden_0.1_merged"]))
ax.set_xlim(*window)
ax.set_xticks([])

panel, ax = fig.main.add_under(chd.models.pred.plot.Pileup(regionmultiwindow.get_plotdata(gene_oi), window = window, width = 10))
panel, ax = fig.main.add_under(chd.models.pred.plot.Effect(regionmultiwindow.get_plotdata(gene_oi), window = window, width = 10))

fig.plot()

# %%
plotdata = regionmultiwindow.get_plotdata(gene_oi)
plt.scatter(plotdata["lost"], plotdata["effect"])

# %%
gene_oi = genes_oi[0]
