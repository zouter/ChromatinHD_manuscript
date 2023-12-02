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
from params import nlayers_design, lr_design

# %%
from design import (
    dataset_splitter_peakcaller_predictor_combinations as design_cre,
)
design_cre["method"] = design_cre["peakcaller"] + "/" + design_cre["predictor"]
design_cre["group"] = "cre"

from design import (
    dataset_splitter_method_combinations as design_methods,
)
design_methods["group"] = "chd"

design = pd.concat([design_cre, design_methods], axis=0, ignore_index=True)
print(design.duplicated().any())
design = design.drop_duplicates()
design.index = range(len(design))

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

scores = xr.concat(scores.values(), dim = pd.Index(scores.keys(), name = "design_ix"))

# %%
scores["r2"] = scores["cor"]**2

# %% [markdown]
# ## Overall

# %%
design_oi = design
# design_oi = design.query("regions == '10k10k'")
design_oi = design.query("regions == '100k100k'")
# design_oi = design.query("regions == '500k500k'")
design_oi = design_oi.loc[~design_oi["method"].isin(nlayers_design.index) & ~design_oi["method"].isin(lr_design.index)]

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi.mean("fold").sel(phase = "test")["scored"].mean("gene").to_pandas()
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata.plot.barh()

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi["time"].to_pandas()
plotdata.index = design.iloc[plotdata.index]["method"].values
fig, ax = plt.subplots(figsize = (10, 6))
sns.heatmap(plotdata, vmax = 10)
plotdata.mean(1)

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index), fold = [0])
plotdata = scores_oi.mean("fold").sel(phase = "test")["cor"].mean("gene").to_pandas()
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata.plot.barh()

# %%
designs_oi = scores_oi["scored"].all(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = "test")["cor"].mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata["r2"] = plotdata["cor"]**2
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata["x"] = plotdata["time"]*5000/60/60
plotdata["y"] = plotdata["r2"]

# %%
plotdata.sort_values("r2", ascending = False).style.bar()

# %%
fig, ax = plt.subplots(figsize = (5, 5))
ax.scatter(plotdata["x"], plotdata["y"])
ax.set_ylim(plotdata["y"].max() * 0.5, plotdata["y"].max())
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_ylim(0)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("R2")
texts = []
for method, row in plotdata.iterrows():
    text = ax.text(row["x"], row["y"], method, ha = "center", bbox = dict(facecolor = "white", alpha = 0.8, boxstyle='square,pad=0'), fontsize = 5)
    texts.append(text)
import adjustText
adjustText.adjust_text(texts)

# %% [markdown]
# ## Pairwise

# %%
design_oi = design.query("regions == '100k100k'")
# design_oi = design.query("regions == '10k10k'")

# %%
# a = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance_wd1e-1_linearlib'").index[0]

# a = design_oi.query("method == 'macs2_leiden_0.1_merged/lasso'").index[0]
# a = design_oi.query("method == 'macs2_leiden_0.1_merged/xgboost'").index[0]
# a = design_oi.query("method == 'rolling_50/lasso'").index[0]
# a = design_oi.query("method == 'rolling_100/lasso'").index[0]
# a = design_oi.query("method == 'rolling_100/xgboost'").index[0]
# a = design_oi.query("method == 'rolling_500/xgboost'").index[0]
# a = design_oi.query("method == 'radial_binary_1000-31frequencies'").index[0]
a = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e'").index[0]
# a = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e_lr0.0001'").index[0]

# b = design_oi.query("method == 'radial_binary_1000-31frequencies_adamw'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance_residualfull'").index[0]
# b = design_oi.query("method == 'spline_binary_residualfull_lne2e_1layerfe'").index[0]
b = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e_linearlib'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e_1layerfe'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lnfull'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies'").index[0]
# b = design_oi.query("method == 'sine_50frequencies_residualfull_lne2e'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e_lr1e-05'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance_lrs_1e-3'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance_lr1e-3'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance_lrs1e-3-2'").index[0]

plotdata = scores["r2"].sel(phase = "test").mean("fold").to_pandas().T
plotdata = plotdata.loc[(plotdata[b] != 0) & (plotdata[a] != 0)] # remove zero values

# %%
plotdata[b].mean()

# %%
(plotdata[b] - plotdata[a]).mean(), (plotdata[b] > plotdata[a]).mean()

# %%
dataset_name = "pbmc10k/subsets/top250"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
regions_name = "100k100k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

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
plotdata["kurtosis"] = 0.

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
ax.axline((0, lm.intercept), slope = lm.slope, color = "cyan", linestyle = "--")

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
ax.scatter(plotdata["kurtosis"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("kurtosis")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["kurtosis"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

fig.plot()

# %%
# regress both b and kurtosis_rank with diff
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

lm = linear_model.LinearRegression().fit(plotdata[["kurtosis_rank", b, "means"]], plotdata["diff"])
lm.coef_

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

lm = stats.lm("diff ~ kurtosis_rank + oi + means + dispersions_norm", data = plotdata)
ro.conversion.rpy2py(broom.tidy_lm(lm))

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata[b], plotdata["diff"])
lm = scipy.stats.linregress(plotdata[b], plotdata["diff"])
ax.plot(plotdata[b], lm.slope * plotdata[b] + lm.intercept, color = "black", linestyle = "--")
plotdata["diff_relative"] = plotdata["diff"] / plotdata[b]

# %%
fig, ax = plt.subplots()
lines = [([row.kurtosis_rank, row[a]], [row.kurtosis_rank, row[b]]) for ix, row in plotdata.iterrows()]
colors = mpl.colormaps["Blues"](plotdata["diff_relative"])
lc = mpl.collections.LineCollection(lines, color = colors, alpha = 0.5)
ax.add_collection(lc)
ax.set_xlim(0, plotdata["kurtosis_rank"].max())

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["kurtosis_rank"], plotdata["diff_relative"])
lm = scipy.stats.linregress(plotdata["kurtosis_rank"], plotdata["diff_relative"])
ax.plot(plotdata["kurtosis_rank"], lm.slope * plotdata["kurtosis_rank"] + lm.intercept, color = "black", linestyle = "--")

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["kurtosis_rank"], plotdata["diff"])
lm = scipy.stats.linregress(plotdata["kurtosis_rank"], plotdata["diff"])
ax.plot(plotdata["kurtosis_rank"], lm.slope * plotdata["kurtosis_rank"] + lm.intercept, color = "black", linestyle = "--")

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["kurtosis_rank"], plotdata[b])
lm = scipy.stats.linregress(plotdata["kurtosis_rank"], plotdata[b])
ax.plot(plotdata["kurtosis_rank"], lm.slope * plotdata["kurtosis_rank"] + lm.intercept, color = "black", linestyle = "--")

# %%
transcriptome.var

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata[a], plotdata[b])
import adjustText
plotdata_oi = plotdata.sort_values("diff").tail(5)
texts = []
for i, row in plotdata_oi.iterrows():
    texts.append(ax.text(row[a], row[b], transcriptome.symbol(i), ha="center"))
adjustText.adjust_text(texts, arrowprops = dict(arrowstyle = "->", color = "black"), ha = "center", expand_text=(1.5, 1.5))
ax.plot([0, 1], [0, 1])
ax.scatter(plotdata[a].mean(), plotdata[b].mean(), color = "red")
ax.text(plotdata[a].mean(), plotdata[b].mean(), (plotdata[a].mean() - plotdata[b].mean()).round(2), ha = "center", va = "bottom", color = "red", bbox = dict(facecolor = "#FFFFFF99", edgecolor = "none"))

# %%
transcriptome.adata.obs["ncounts"] = np.log1p(transcriptome.adata.obs["n_counts"])
sc.get.obs_df(transcriptome.adata, [gene_oi, "nfragments", "ncounts"]).corr()

# %% [markdown]
# ## Comparisons

# %% [markdown]
# ### Comparison with # of fragments and diff

# %%
cors = pd.Series(chd.utils.paircor(np.log1p(fragments.counts.sum(1)[:, None]), transcriptome.layers["magic"][:, transcriptome.var.index.get_indexer(fragments.var.index)]), index = fragments.var.index)
plt.hist(cors)
plotdata["cor"] = cors

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["diff"], plotdata["cor"])

# %% [markdown]
# ### Number of NN layers

# %%
design_oi = design.query("regions == '100k100k'")
design_oi = design_oi.loc[design_oi["method"].isin(nlayers_design.index)]

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].all(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = "test")["cor"].mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = "test")["r2"]).mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata = plotdata.join(nlayers_design)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

panel, ax = fig.main.add_right(chd.grid.Panel((3, 3)))

sns.heatmap(plotdata.set_index(["layerfe", "layere2e"])["r2"].unstack(), annot = True, fmt = ".2%", cmap = "Blues", ax = ax, cbar = False)

panel, ax = fig.main.add_right(chd.grid.Panel((3, 3)))
sns.heatmap(np.log(plotdata.set_index(["layerfe", "layere2e"])["time"].unstack()), ax = ax, cbar = False, annot = True)

fig.plot()


# %%
plotdata["x"] = plotdata["time"]*5000/60/60
plotdata["y"] = plotdata["r2"]
fig, ax = plt.subplots(figsize = (5, 5))
ax.scatter(plotdata["x"], plotdata["y"])
# ax.set_ylim(plotdata["y"].max() * 0.7, plotdata["y"].max())
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlabel("Time (hours)")
ax.set_ylabel("R2")

# %% [markdown]
# ### Learning rate

# %%
design_oi = design.query("regions == '100k100k'")
design_oi = design_oi.loc[design_oi["method"].isin(lr_design.index)]

scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].all(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = "test")["cor"].mean("gene").to_pandas(),
    "r2":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = "test")["r2"].mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata = plotdata.join(lr_design)

# %%
plotdata["x"] = plotdata["time"]*5000/60/60
plotdata["y"] = plotdata["r2"]
fig, ax = plt.subplots(figsize = (5, 5))
norm = mpl.colors.LogNorm(vmin = plotdata["lr"].min(), vmax = plotdata["lr"].max())
cmap = mpl.colormaps.get_cmap("viridis")
ax.scatter(plotdata["x"], plotdata["y"], c = cmap(norm(plotdata["lr"])))
texts = []
for _, row in plotdata.iterrows():
    text = ax.text(row["x"], row["y"], "{:.0e}".format(row["lr"]), ha = "center", bbox = dict(facecolor = "white", alpha = 0.8, boxstyle='square,pad=0'), fontsize = 10)
    texts.append(text)

# ax.set_ylim(plotdata["y"].max() * 0.7, plotdata["y"].max())
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlabel("Time (hours)")
ax.set_ylabel("R2")
fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, label = "Learning rate")

# %% [markdown]
# ### Library size normalization

# %%
dataset_name = "pbmc10k/subsets/top250"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
regions_name = "100k100k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
design_oi = design.query("regions == '100k100k'")
design_oi = design_oi.loc[design_oi["method"].isin(
    [
        "radial_binary_1000-31frequencies_residualfull_lne2e_linearlib",
        "radial_binary_1000-31frequencies_residualfull_lne2e"
    ]
)]

scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].all(["gene", "fold"])
r2 = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = "test")["r2"].to_pandas()

plotdata = pd.DataFrame({
    "diff":(r2.iloc[0] - r2.iloc[1]).sort_values()
})
plotdata["symbol"] = transcriptome.symbol(plotdata.index)
plotdata.sort_values("diff", ascending = False).head(10)

# %%
fig, ax = plt.subplots()
for _, row in plotdata.T.iterrows():
    ax.plot(row.index, row.values)

# %% [markdown]
# ## Across dataset sizes

# %%
method_oi = "spline_binary_residualfull_lne2e_1layerfe"

# %%
design_oi = design.query("method == @method_oi")
design_oi = design_oi.loc[design_oi.index.isin(scores.coords["design_ix"][scores["scored"].all("fold").any("gene")].values)]

# %%
from design import region_progression
regions_design = pd.DataFrame({
    "regions":region_progression,
    "size":[1000*2, 2000*2, 5000*2, 10000*2, 20000*2, 50000*2, 100000*2, 200000*2, 500000*2, 1000000*2]
}).set_index("regions")

# %%
plotdata = scores.sel(design_ix = design_oi.index)["time"].to_pandas()
plotdata.index = design.iloc[plotdata.index]["regions"].values
plotdata = plotdata.loc[:, (~pd.isnull(plotdata)).all(0)]
plotdata = plotdata.reindex(region_progression)
plotdata = plotdata.dropna()

fig, ax = plt.subplots(figsize = (2, 2))
# plotdata = plotdata.loc[["10k10k", "100k100k", "500k500k"]]
plotdata_mean = plotdata.mean(1)
ax.plot(regions_design.loc[plotdata_mean.index, "size"], plotdata_mean, marker = "o", color = "#333333", zorder = 10)
ax.set_xscale("log")
ax.set_yscale("log")
lines = []
for i, row in plotdata.T.iterrows():
    line = ax.plot(regions_design.loc[row.index, "size"], row, color = "#0074D9", alpha = 0.1)
    lines.append(line[0])

# %%
metric = "r2"
metric = "cor"

# %%
plotdata = scores.sel(design_ix = design_oi.index)[metric].sel(phase = "test").mean("fold").to_pandas()
plotdata = plotdata.loc[:, (plotdata != 0).all(0)]
plotdata.index = design.iloc[plotdata.index]["regions"].values

plotdata_train = scores.sel(design_ix = design_oi.index)[metric].sel(phase = "train").mean("fold").to_pandas()
plotdata_train = plotdata_train.loc[:, (plotdata_train != 0).all(0)]
plotdata_train.index = design.iloc[plotdata_train.index]["regions"].values

# %%
import scipy.optimize
import scipy.stats

# %%
fig, ax = plt.subplots()
plotdata = plotdata.reindex(region_progression)
plotdata = plotdata.dropna()
plotdata_mean = plotdata.mean(1)
ax.plot(regions_design.loc[plotdata_mean.index, "size"], plotdata_mean, marker = "o", color = "#0074D9")
ax.set_xscale("log")
lines = []
for i, row in plotdata.T.iterrows():
    row = row.dropna()
    line = ax.plot(regions_design.loc[row.index, "size"], row, color = "#0074D9", alpha = 0.1)
    lines.append(line[0])

plotdata_train = plotdata_train.reindex(region_progression)
plotdata_train_mean = plotdata_train.mean(1)
plotdata_train_mean = plotdata_train_mean.dropna()
ax.plot(regions_design.loc[plotdata_train_mean.index, "size"], plotdata_train_mean, marker = "o", color = "orange")

x = np.log10(regions_design.loc[plotdata_mean.index, "size"])
y = plotdata_mean
lm = scipy.stats.linregress(x, y)
ax.plot(10**x, lm.intercept + x * lm.slope, color = "#0074D9", linestyle = "--")


# %%
def f(x, a, b, c):
    # return a * x + b
    return a * np.log(x) + b
    # return a * x**(1/b) + c

x = regions_design.loc[plotdata_mean.index, "size"]
# y = np.sqrt(plotdata_mean.values)
y = (plotdata_mean.values)

popt, pcov = scipy.optimize.curve_fit(f, x, y)

# %%
fig, ax = plt.subplots()
ax.plot(x, y, marker = "o", color = "#0074D9")
ax.plot(x, f(x, *popt), color = "black", linestyle = "--")

# %%
