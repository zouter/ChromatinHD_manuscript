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

# %% [markdown]
# ## Load

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_peakcaller_predictor_combinations as design_cre,
)
design_cre["method"] = design_cre["peakcaller"] + "/" + design_cre["predictor"]
design_cre["group"] = "cre"

from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_method_combinations as design_methods,
)
design_methods["group"] = "chd"



from chromatinhd_manuscript.designs_pred import (
    dataset_baseline_combinations as design_baseline,
)
design_baseline["group"] = "baseline"

design = pd.concat([design_cre, design_methods, design_baseline], axis=0, ignore_index=True)

# %%
scores = {}

for design_ix, design_row in design.query("group == 'chd'").iterrows():
    prediction = chd.flow.Flow(
        chd.get_output() / "pred" / design_row.dataset / design_row.regions / design_row.splitter / design_row.layer / design_row.method,
    )
    performance = chd.models.pred.interpret.Performance(prediction.path / "scoring" / "performance")

    try:
        scores[design_ix] = performance.scores.sel_xr()
        # raise ValueError("Scores already exist")
    except FileNotFoundError:
        continue

for design_ix, design_row in design.query("group == 'cre'").iterrows():
    prediction = chd.flow.Flow(
        chd.get_output() / "pred" / design_row.dataset / design_row.regions / design_row.splitter / design_row.layer / design_row.method,
    )
    performance = chd.models.pred.interpret.Performance(prediction.path)

    try:
        scores[design_ix] = performance.scores.sel_xr()
        # raise ValueError("Scores already exist")
    except FileNotFoundError:
        continue

for design_ix, design_row in design.query("group == 'baseline'").iterrows():
    prediction = chd.flow.Flow(
        chd.get_output() / "pred" / design_row.dataset / design_row.regions / design_row.splitter / design_row.layer / design_row.method,
    )
    performance = chd.models.pred.interpret.Performance(prediction.path / "scoring" / "performance")

    try:
        scores[design_ix] = performance.scores.sel_xr()
        # raise ValueError("Scores already exist")
    except FileNotFoundError:
        continue

scores = xr.concat(scores.values(), dim = pd.Index(scores.keys(), name = "design_ix"))

# %%
scores["r2"] = scores["cor"]**2

# %%
scores = scores.reindex({"design_ix":pd.Index(design.index, name = "design_ix")}, fill_value = dict(scored = 0.))
scores["scored"] = scores["scored"].fillna(0.)

# %% [markdown]
# ## Score overall

# %%
design_oi = design
# design_oi = design.query("regions == '10k10k'")
# design_oi = design.query("regions == '100k100k'")

# %%
design["label"] = design["dataset"] + "/" + design["regions"] + "/" + design["method"]

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
# plotdata = scores_oi.mean("fold").sel(phase = "test")["scored"].mean("gene").to_pandas()
# plotdata.index = design.iloc[plotdata.index]["label"].values
# plotdata.plot.barh()

# %%
main_features = ["dataset", "regions", "splitter", "layer"]
reference_method = "v33"

# %%
design["reference"] = design.query("method == @reference_method").reset_index().set_index(main_features).loc[pd.MultiIndex.from_frame(design[main_features])]["index"].values

# %%
r2 = (scores["r2"].sel(phase = "test").mean("fold") * scores["scored"].all("fold").sel(design_ix = design["reference"].values).values[:, :]).mean("gene")

# %%
r2s = scores["r2"].sel(phase = "test").mean("fold")
r2s.values[~scores["scored"].all("fold").sel(design_ix = design["reference"].values).values] = np.nan
nonan = (~np.isnan(r2s)).sum("gene")
r2 = r2s.mean("gene")

# %%
scored = (scores["scored"].all("fold").values * scores["scored"].all("fold").sel(design_ix = design["reference"].values)).sum("gene").to_pandas()
scored.index = design.index

# %%
from chromatinhd_manuscript.methods import prediction_methods
import chromatinhd.plot.quasirandom

# %%
plotdata = pd.DataFrame(
    {
        "r2": r2.to_pandas(),
        "scored": scored,
        "cor": np.sqrt(r2.to_pandas()),
        "nonan": nonan.to_pandas(),
    }
).dropna()
plotdata["design_ix"] = plotdata.index
plotdata["label"] = design.loc[plotdata.index, "label"].values
plotdata.index = pd.MultiIndex.from_frame(design.loc[plotdata.index, main_features])

plotdata_grouped = plotdata.groupby(main_features)

fig, ax = plt.subplots(figsize=(4, len(plotdata_grouped) * 0.4))

score = "r2"
# score = "scored"
# score = "cor"

x = 0.0
xlabels = []
for dataset, plotdata_dataset in plotdata_grouped:
    x += 1.0
    xlabels += ["/".join(dataset)]

    y = np.array(chd.plot.quasirandom.offsetr(np.array(plotdata_dataset[score].values.tolist()), adjust=0.1)) * 0.8 + x
    plotdata_dataset["y"] = y

    plotdata_dataset["color"] = prediction_methods.loc[
        design.loc[plotdata_dataset["design_ix"], "method"], "color"
    ].values
    ax.axhspan(x - 0.5, x + 0.5, color="#33333308", ec="white")
    ax.scatter(plotdata_dataset[score], plotdata_dataset["y"], s=5, color=plotdata_dataset["color"], lw=0)

    plotdata_dataset["type"] = prediction_methods.loc[
        design.loc[plotdata_dataset["design_ix"], "method"], "type"
    ].values
    plotdata_top = plotdata_dataset.sort_values(score, ascending=False).groupby("type").first()
    for type, plotdata_top in plotdata_top.groupby("type"):
        ax.plot(
            [plotdata_top[score]] * 2,
            [x - 0.5, x + 0.5],
            color=plotdata_top["color"].values[0],
            lw=1,
            zorder=0,
            alpha=0.5,
        )
ax.set_yticks(np.arange(1, x + 1))
ax.set_yticklabels(xlabels)
ax.set_xlabel("R2")
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

# %%
plotdata.loc["pbmc10k"].loc["10k10k"].loc["5x1"].loc["magic"].sort_values("r2")

# %%
fig, ax = plt.subplots()
plt.scatter(r2s.sel(design_ix = 358).to_pandas().dropna(), r2s.sel(design_ix = 393).to_pandas().dropna())
# plt.scatter(r2s.sel(design_ix = 112).to_pandas().dropna(), r2s.sel(design_ix = 428).to_pandas().dropna())
# plt.scatter(r2s.sel(design_ix = 192).to_pandas().dropna(), r2s.sel(design_ix = 434).to_pandas().dropna())
ax.plot([0, 1], [0, 1])

# %%
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "hspc"
# dataset_name = "lymphoma"
regions_name = "100k100k"
# regions_name = "10k10k"
transcriptome = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
design_oi = design.query("dataset == @dataset_name").query("regions == @regions_name").query("splitter == '5x1'").query("layer == 'magic'")

# %%
plotdata = r2s.sel(design_ix = design_oi.index).to_pandas().T
plotdata.columns = design_oi["method"]
plotdata = plotdata.loc[~pd.isnull(plotdata["v33"])]

# %%
a = "macs2_leiden_0.1_merged/lasso"
# a = "macs2_leiden_0.1_merged/xgboost"
# a = "rolling_500/lasso"
# a = "rolling_500/xgboost"
# a = "rolling_100/lasso"
b = "v33"
 
plotdata = plotdata.loc[(plotdata[a] > 0.) & (plotdata[b] > 0.)]
plotdata.shape[0]

# %%
import statsmodels.api as sm
import scipy.stats

fig = chd.grid.Figure(chd.grid.Wrap(padding_width = 0.7, ncol = 4))

plotdata = plotdata.sort_values(a)
plotdata["diff"] = plotdata[b] - plotdata[a]
plotdata["tick"] = (np.arange(len(plotdata)) % int(len(plotdata)/10)) == 0
plotdata["i"] = np.arange(len(plotdata))
plotdata["dispersions"] = transcriptome.var["dispersions"].loc[plotdata.index.get_level_values("gene")].values
plotdata["means"] = transcriptome.var["means"].loc[plotdata.index.get_level_values("gene")].values
plotdata["dispersions_norm"] = transcriptome.var["dispersions_norm"].loc[plotdata.index.get_level_values("gene")].values
plotdata["log10means"] = np.log10(plotdata["means"])
plotdata["log10dispersions_norm"] = np.log10(plotdata["dispersions_norm"])
n_fragments = pd.Series(fragments.counts.mean(0), index = fragments.var.index)
plotdata["log10n_fragments"] = np.log10(n_fragments.loc[plotdata.index.get_level_values("gene")].values)
n_fragments_std = pd.Series(np.log1p(fragments.counts).std(0), index = fragments.var.index)
plotdata["n_fragments_std"] = n_fragments_std.loc[plotdata.index.get_level_values("gene")].values
plotdata["oi"] = pd.Categorical([True] * len(plotdata), categories = [True, False])

transcriptome.var["kurtosis"] = (scipy.stats.kurtosis(transcriptome.layers["magic"]))
plotdata["kurtosis"] = transcriptome.var["kurtosis"].loc[plotdata.index.get_level_values("gene")].values
plotdata["kurtosis_rank"] = scipy.stats.rankdata(plotdata["kurtosis"]) / len(plotdata)

a_label = str(a)
b_label = str(b)

cmap = mpl.colormaps["Set1"]

# rank vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(np.arange(len(plotdata)), (plotdata["diff"]), c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a_label + " rank")
ax.set_ylabel("$\Delta$ r2")
ax.axhline(0, color = "black", linestyle = "--")
ax.set_xticks(plotdata["i"].loc[plotdata["tick"]])
ax.set_xticklabels(plotdata[a].loc[plotdata["tick"]].round(2).values, rotation = 90)

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], np.arange(len(plotdata)), frac = 0.5)
ax.plot(z[:, 0], z[:, 1], color = "green")

# a vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata[a], (plotdata["diff"]), c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a_label)
ax.set_ylabel("$\Delta$ r2")
ax.axhline(0, color = "black", linestyle = "--")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata[a], frac = 0.5)
ax.plot(z[:, 0], z[:, 1], color = "green")

lm = scipy.stats.linregress(plotdata[a], plotdata["diff"])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope

# vs
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata[a], plotdata[b], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a_label)
ax.set_ylabel(b_label)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

lowess = sm.nonparametric.lowess
z = lowess(plotdata[b], plotdata[a], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")
ax.axline((0, 0), slope = 1, color = "black", linestyle = "--")
ax.set_aspect("equal")

lm = scipy.stats.linregress(plotdata[a], plotdata[b])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope

cut = (1 - lm.intercept) / lm.slope
print(cut)
ax.annotate(f"cut {1-cut:.1%}", (1, 1), (1, 1.1), arrowprops = dict(arrowstyle = "->"), ha = "right")
ax.annotate(f"$r^2$={lm.rvalue**2:.1%}", (0.95, 0.95), ha = "right", va = "top")

# dispersions vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["dispersions"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("dispersion")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["dispersions"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10means"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10means")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10means"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions_norm vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10dispersions_norm"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10dispersions_norm")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10dispersions_norm"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10n_fragments"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10n_fragments")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10n_fragments"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments std
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["kurtosis_rank"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("kurtosis_rank")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["kurtosis_rank"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

fig.plot()

# %%
import rpy2.robjects as ro
import rpy2.robjects.packages
from rpy2.robjects import pandas2ri
pandas2ri.activate()

plotdata["oi"] = plotdata[a]

# do lm in R
stats = ro.packages.importr("stats")
base = ro.packages.importr('base')
broom = ro.packages.importr('broom')

lm = stats.lm("diff ~ kurtosis_rank + log10means + dispersions", data = plotdata)
ro.conversion.rpy2py(broom.tidy_lm(lm))

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata[b], plotdata["diff"])
lm = scipy.stats.linregress(plotdata[b], plotdata["diff"])
ax.plot(plotdata[b], lm.slope * plotdata[b] + lm.intercept, color = "black", linestyle = "--")
plotdata["diff_relative"] = np.clip(plotdata["diff"] / plotdata[b], 0, np.inf)

# %%
fig, ax = plt.subplots()
lines = [([row.kurtosis_rank, row[a]], [row.kurtosis_rank, row[b]]) for ix, row in plotdata.iterrows()]
colors = mpl.colormaps["Blues"](plotdata["diff_relative"]*2)
lc = mpl.collections.LineCollection(lines, color = colors)
ax.add_collection(lc)
ax.scatter(plotdata["kurtosis_rank"], plotdata[a], s = 1, marker = "_", c = colors)
ax.scatter(plotdata["kurtosis_rank"], plotdata[b], s = 1, c = colors)
ax.set_xlim(0, plotdata["kurtosis_rank"].max())

# %%
symbols_oi = transcriptome.symbol(plotdata["kurtosis_rank"].sort_values().index).tail(2)
genes_oi = transcriptome.gene_id(symbols_oi)

import scanpy as sc
sc.pl.umap(transcriptome.adata, color = genes_oi, layer = "magic", title = symbols_oi)

# %%
design = chd.utils.crossing(
    pd.DataFrame({
        "differentiation":np.linspace(0, 1, 10),
    }),
    pd.DataFrame({
        "cellcycle":np.linspace(0, 1, 10),
    })
)
design["proliferating"] = scipy.stats.norm.pdf(design["differentiation"], 0.4, 0.2)
design["proliferating"] = design["proliferating"] / design["proliferating"].max()

design["prob_a"] = 1 * (scipy.stats.norm.pdf(design["cellcycle"], 0., 0.1) * (1-design["proliferating"]) + scipy.stats.uniform.pdf(design["cellcycle"]) * design["proliferating"] * 2)
# design["prob_a"] = design["prob_a"] / design["prob_a"].max()

design["proliferating_b"] = design["proliferating"]
design["prob_b"] = 1 * (scipy.stats.norm.pdf(design["cellcycle"], 0., 0.1) * (design["differentiation"] < 0.5) * (1-design["proliferating_b"]) + scipy.stats.uniform.pdf(design["cellcycle"]) * design["proliferating_b"] * 2)
# design["prob_b"] = design["prob_b"] / design["prob_b"].max()

design["prob_diff"] = design["prob_b"] - design["prob_a"]

plotdata = design.set_index(["differentiation", "cellcycle"])["prob_a"].unstack().T
fig, ax = plt.subplots(figsize = (1, 1))
cmap = mpl.colormaps["Blues"]
ax.matshow(plotdata, cmap=cmap)
ax.axis("off")

plotdata = design.set_index(["differentiation", "cellcycle"])["prob_b"].unstack().T
fig, ax = plt.subplots(figsize = (1, 1))
cmap = mpl.colormaps["Blues"]
ax.matshow(plotdata, cmap=cmap)
ax.axis("off")
