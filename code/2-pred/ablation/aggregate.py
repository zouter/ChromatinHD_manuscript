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
from params import nlayers_design, lr_design, nfrequencies_design

# %% [markdown]
# ## Load scores

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
fig, ax = plt.subplots(figsize = (10, 8))
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
a = design_oi.query("method == 'v33_relu'").index[0]
# a = design_oi.query("method == 'v33_radial_binary'").index[0]
# a = design_oi.query("method == 'counter'").index[0]
# a = design_oi.query("method == 'v33_tophat_[6250]frequencies'").index[0]
# a = design_oi.query("method == 'v33_[1000, 500, 250, 125, 63, 31, 15]frequencies'").index[0]
# a = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e'").index[0]
# a = design_oi.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e_lr0.0001'").index[0]

# b = design_oi.query("method == 'radial_binary_1000-31frequencies_adamw'").index[0]
# b = design_oi.query("method == 'radial_binary_1000-31frequencies_splitdistance_residualfull'").index[0]
# b = design_oi.query("method == 'spline_binary_residualfull_lne2e_1layerfe'").index[0]
# b = design_oi.query("method == 'v33_tophat_binary'").index[0]
b = design_oi.query("method == 'v33_silu'").index[0]
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
plotdata[b].mean(), plotdata[a].mean()

# %%
plotdata_time = scores["time"].to_pandas().T
plotdata_time = plotdata_time.loc[(plotdata_time[b] != 0) & (plotdata_time[a] != 0)] # remove zero values
print(plotdata_time[b].mean(), plotdata_time[a].mean())

plotdata_time = scores["time_interpretation"].to_pandas().T
plotdata_time = plotdata_time.loc[(plotdata_time[b] != 0) & (plotdata_time[a] != 0)] # remove zero values
print(plotdata_time[b].mean(), plotdata_time[a].mean())

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
# transcriptome.adata.obs["ncounts"] = np.log1p(transcriptome.adata.obs["n_counts"])
# sc.get.obs_df(transcriptome.adata, [gene_oi, "nfragments", "ncounts"]).corr()

# %% [markdown]
# ## Comparisons

# %% [markdown]
# ### Time

# %%
design_oi = design.loc[design["regions"] == "100k100k"]
design_oi = design_oi.iloc[
pd.Index(design_oi["method"]).get_indexer([
            # "v33",
            "v33_windows50",
            "v33_windows100",
            "v33_windows500",
            # "v33_tophat_[6250]frequencies",
            "rolling_50/xgboost_gpu",
            "rolling_100/xgboost_gpu",
            "rolling_500/xgboost_gpu",
            "v33_cpu",
            "rolling_50/lasso",
            "rolling_50/xgboost",
            "rolling_100/lasso",
            "rolling_100/xgboost",
            "rolling_500/lasso",
            "rolling_500/xgboost",
            "encode_screen/linear",
            "encode_screen/lasso",
            "encode_screen/xgboost",
            # "encode_screen/xgboost_gpu",
            "macs2_leiden_0.1_merged/linear",
            "macs2_leiden_0.1_merged/lasso",
            "macs2_leiden_0.1_merged/xgboost",
            # "macs2_leiden_0.1_merged/xgboost_gpu",
        ])
]
# design_oi = design_oi.loc[design_oi["method"].isin(["v33"])]
design_oi = design_oi.copy()

# %%
design_oi

# %%
design_oi.loc[~design_oi.index.isin(scores.coords["design_ix"].values)]

# %%
scores_oi = scores.sel(design_ix = design_oi.index).mean("fold").sel(phase = ["validation", "test"]).mean("phase").to_dataframe()
scores_oi = scores_oi.loc[scores_oi["scored"] == 1.]

# %%
scores_oi["time_interpretation"].fillna(0., inplace = True)

# %%
design_oi.loc[design_oi["peakcaller"] == 'macs2_leiden_0.1_merged', "time_peakcalling"] = 32/60 # peak calling + counting took 32 minutes
design_oi.loc[design_oi["peakcaller"] == 'encode_screen', "time_peakcalling"] = 7/60 # counting took 7 minutes
design_oi["time_peakcalling"].fillna(0., inplace = True)

design_oi["time"] = ((scores_oi["time"] + scores_oi["time_interpretation"])).groupby("design_ix").median() * 5000 / 60 / 60
design_oi["time"] = design_oi["time"] + design_oi["time_peakcalling"]
design_oi["time_train"] = ((scores_oi["time"])).groupby("design_ix").median() * 5000 / 60 / 60
design_oi["time_interpretation"] = ((scores_oi["time_interpretation"])).groupby("design_ix").median() * 5000 / 60 / 60
design_oi["r2"] = ((scores_oi["r2"])).groupby("design_ix").median() * 5000 / 60 / 60
design_oi["cor"] = ((scores_oi["cor"])).groupby("design_ix").median() * 5000 / 60 / 60

# %%
design_oi.loc[design_oi["method"] == 'v33_cpu', "r2"] = design_oi.loc[design_oi["method"] == 'v33_windows50', "r2"].values
# design_oi.loc[design_oi["method"] == 'v33_cpu', "r2"] = design_oi.loc[design_oi["method"] == 'v33', "r2"].values

# %%
from chromatinhd_manuscript.methods import prediction_methods
prediction_methods.loc["v33", "label_full"] = "ChromatinHD default"
prediction_methods.loc["v33_cpu"] = prediction_methods.loc["v33"].copy()
prediction_methods.loc["v33_cpu", "label_full"] = "ChromatinHD CPU"
prediction_methods.loc["v33_windows50"] = prediction_methods.loc["v33"].copy()
prediction_methods.loc["v33_windows50", "label_full"] = "ChromatinHD default"
prediction_methods.loc["v33_windows100"] = prediction_methods.loc["v33"].copy()
prediction_methods.loc["v33_windows100", "label_full"] = "ChromatinHD 100bp interpretation"
prediction_methods.loc["v33_windows500"] = prediction_methods.loc["v33"].copy()
prediction_methods.loc["v33_windows500", "label_full"] = "ChromatinHD 500bp interpretation"
prediction_methods.loc["v33_tophat_[6250]frequencies"] = prediction_methods.loc["v33"].copy()
prediction_methods.loc["v33_tophat_[6250]frequencies", "label_full"] = "ChromatinHD 32bp tophat kernel"

prediction_methods_xgboost_gpu = prediction_methods.query("predictor == 'xgboost'").assign(predictor = "xgboost_gpu")
prediction_methods_xgboost_gpu.index = prediction_methods_xgboost_gpu.index + "_gpu"
prediction_methods = pd.concat([prediction_methods, prediction_methods_xgboost_gpu])

design_oi["label"] = design_oi["method"].map(prediction_methods["label_full"])
# design_oi["label"] = [label if pd.isnull(peakcaller) else label + " " + diffexp for label, peakcaller, diffexp in zip(design_oi["method"], design_oi["peakcaller"], design_oi["diffexp"])]
design_oi["color"] = design_oi["method"].map(prediction_methods["color"])
design_oi["cpu"] = [False if method in ["v33", "v33_windows100", "v33_windows500", "v33_windows50", "v33_tophat_[6250]frequencies"] or "gpu" in method else True for method in design_oi["method"]]

# %%
fig, ax = plt.subplots(figsize = (5, 5))
plotdata = design_oi.query("cpu")
points1 = ax.scatter(plotdata["time"], plotdata["r2"], c = plotdata["color"], s = 50, alpha = 1.0, marker = "o")
plotdata = design_oi.query("~cpu")
points2 = ax.scatter(plotdata["time"], plotdata["r2"], c = plotdata["color"], s = 50, alpha = 1.0, marker = "x")

import textwrap
texts = []
for i, row in design_oi.iterrows():
    text = ax.text(row["time"], row["r2"], "\n".join(textwrap.wrap(row["label"], 11)), ha = "center", va = "bottom", fontsize = 7, color = row["color"])
    text.set_path_effects([mpl.patheffects.withStroke(linewidth = 2, foreground = "white")])
    texts.append(text)

ax.set_xscale("log")
ax.set_xlabel("Total time (hours)")
ax.set_xticks([1, 2, 5, 10, 20, 50])
ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])

ax.set_ylim(design_oi["r2"].min() * 0.95, design_oi["r2"].max() * 1.1)

import adjustText
adjustText.adjust_text(texts, arrowprops = dict(arrowstyle = "-", color = "#33333333"), time_lim = 4)

ax.set_ylabel("OOS-$R^2$")

# markers purely with symbol
markers = [
    mpl.lines.Line2D([0], [0], marker = "o", color = "#333", markerfacecolor = "black", markersize = 8, linestyle = 'None'),
    mpl.lines.Line2D([0], [0], marker = "x", color = "#333", markerfacecolor = "black", markersize = 8, linestyle = 'None'),
]
labels = ["CPU", "GPU"]
ax.legend(markers, labels, loc = "upper left")
sns.despine()

manuscript.save_figure(fig, "2", "snote", "time_vs_r2")

# %%
design_oi["ix"] = np.arange(len(design_oi))

# %%
fig, ax = plt.subplots(figsize = (3, 5))
ax.barh(design_oi["ix"], design_oi["time_peakcalling"], color = design_oi["color"], hatch = "\\\\\\", lw = 0.)
ax.barh(design_oi["ix"], design_oi["time_train"], left = design_oi["time_peakcalling"], color = design_oi["color"],lw = 0.)
plotdata = design_oi.query("time_interpretation > 0")
ax.barh(plotdata["ix"], plotdata["time_interpretation"], left = plotdata["time_peakcalling"] + plotdata["time_train"], color = plotdata["color"], lw = 0., hatch = "///")

def format_time(hours):
    if hours >= 1:
        # Extract the whole hours and the fractional part representing minutes
        whole_hours = int(hours)
        minutes = int((hours - whole_hours) * 60)
        return f"{whole_hours}h {minutes}m"
    else:
        # Convert hours to minutes directly if less than 1 hour
        minutes = int(hours * 60)
        return f"{minutes}m"
for i, row in design_oi.iterrows():
    ax.text(row["time"], row["ix"], f" {format_time(row['time'])}", ha = "left", va = "center", fontsize = 10)

ax.set_yticks(design_oi["ix"])
ax.set_yticklabels(design_oi["label"])

ax.set_xlabel("Total time (hours)")

handles = [
    ["Peak calling", mpl.patches.Rectangle((0, 0), 1, 1, fc = "#333", hatch = "\\\\\\")],
    ["Training", mpl.patches.Rectangle((0, 0), 1, 1, fc = "#333")],
    ["Interpretation", mpl.patches.Rectangle((0, 0), 1, 1, fc = "#333", hatch = "///")],
]
ax.legend([handle for label, handle in handles], [label for label, handle in handles], loc = "lower right")

sns.despine()
ax.set_ylim(len(design_oi)-0.5, -0.5)

manuscript.save_figure(fig, "2", "snote", "time")

# %%
design_oi

# %%
print("Windows 100 vs Windows 50")
plotdata = scores_oi.query("design_ix == 158").droplevel("design_ix").join(scores_oi.query("design_ix == 157").droplevel("design_ix"), rsuffix = "2").dropna()[["time_interpretation", "time_interpretation2"]]
print(plotdata.mean())

print("Windows 500 vs Windows 50")
plotdata = scores_oi.query("design_ix == 159").droplevel("design_ix").join(scores_oi.query("design_ix == 157").droplevel("design_ix"), rsuffix = "2").dropna()[["time", "time2"]]
print(plotdata.mean())

print("CPU lasso vs CPU ChromatinHD")
plotdata = scores_oi.query("design_ix == 29").droplevel("design_ix").join(scores_oi.query("design_ix == 156").droplevel("design_ix"), rsuffix = "2").dropna()[["time", "time2"]]
print(plotdata.mean())

# %%
plotdata = scores_oi.query("design_ix == 159").droplevel("design_ix").join(scores_oi.query("design_ix == 158").droplevel("design_ix"), rsuffix = "2").dropna()[["r2", "r22"]]
fig, ax = plt.subplots()
ax.scatter(plotdata["r2"], plotdata["r22"])
ax.axline((0, 0), slope = 1)
plotdata.T.style.bar()

# %%
scores_oi.query("design_ix == 159").droplevel("design_ix").join(scores_oi.query("design_ix == 158").droplevel("design_ix"), rsuffix = "2").dropna()[["r2", "r22"]].mean()

# %%
scores_oi.loc["ENSG00000275302"].join(design)

# %%
design_oi

# %%
((scores_oi["time"] + scores_oi["time_interpretation"])).mean() * 5000 / 60 / 60

# %%
scores_oi.sum()/60*(5000/250)/60

# %% [markdown]
# ### Distance

# %%
design_oi = design.query("regions == '100k100k'")
design_oi = design_oi.loc[design_oi["method"].isin(["v33", "v33_nodistance"])]

# %%
scores_oi = scores.sel(design_ix = design_oi.index).mean("fold").sel(phase = ["validation", "test"]).mean("phase")["r2"].to_pandas().T
scores_oi.columns = design_oi["method"].values
scores_oi["diff"] = scores_oi["v33"] - scores_oi["v33_nodistance"]

# %%
fig, ax = plt.subplots()
sns.ecdfplot(scores_oi["diff"])
ax.axvline(0, color = "black", linestyle = "--")
ax.axvline(scores_oi["diff"].mean(), color = "red", linestyle = "--")

# %% [markdown]
# ### Non-linear

# %%
from params import nonlinear_design
nonlinear_design = nonlinear_design.fillna(False)

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*nonlinear_design.index, "counter"])]
design_oi = design_oi.iloc[design_oi.set_index("method").index.get_indexer(["counter"] + list(nonlinear_design.index))]
# design_oi = design_oi.loc[(design_oi["method"].map(len)).sort_values().index]
nonlinear_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(nonlinear_design.index)]
design_oi = design_oi.join(nonlinear_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
features = pd.DataFrame([
    ["silu", "SiLU"],
    ["tanh", "tanh"],
    ["sigmoid", "Sigmoid"],
    ["gelu", "GELU"],
    ["relu", "ReLU"],
], columns = ["value", "label"])
features["ix"] = features.index
features.index = features.value
features

design_oi[features.index] = design_oi[features.index].fillna(False)

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
# plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.reindex(design_oi.index, fill_value = 0.)

b = design_oi.query("method == 'v33_relu'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
# plotdata_individual_relb = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[a] - plotdata_individual.loc[ref]), 0, 1)
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 0.5, 2.1)
# plotdata_individual_rela = np.clip((plotdata_individual) / (plotdata_individual.loc[b]), 0.5, 2.)

# %%
import chromatinhd.plot.quasirandom
from rpy2.robjects import pandas2ri

# %%
# genes_oi = [transcriptome.gene_id("IRF8"), transcriptome.gene_id("PKIA")]
genes_oi = []

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    for gene in genes_oi:
        plotdata_oi = plotdata_individual_rela.loc[design_ix]
        ax.scatter([x], [plotdata_oi.loc[gene]], color="red", linewidth=0.5, s=10.0)
        if design_ix == design_oi.index[-1]:
            ax.annotate(
                transcriptome.symbol(gene),
                (x, plotdata_oi.loc[gene]),
                (10, 0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", ec="red"),
                ha="left",
                va="center",
            )

    x += 1

# ax.set_yscale("log")
ax.set_ylim(0.5, 1.2)
ax.set_yticks([0.5, 1.0, 1.2])
ax.set_yticklabels(["-50%", "Baseline", "+20%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks([])

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(chd.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iloc[1:].iterrows():
    n_frequencies = design_row[features.index]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)

    features_oi = features.loc[n_frequencies]
    ax.scatter([x] * len(features_oi), features_oi["ix"], color="#333333", linewidth=0.5, s=s)
    ax.plot(
        [x] * len(features_oi),
        features_oi["ix"],
        color="black",
        linewidth = 1.5,
        zorder = -10
    )

    x += 1
ax.set_xticks([])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("Activation function", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "2/snote", "nonlinear")

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

panel, ax = fig.main.add_right(chd.grid.Panel((2.5, 2.5)))

sns.heatmap(plotdata.set_index(["layerfe", "layere2e"])["r2"].unstack(), annot = True, fmt = ".3f", cmap = "Blues", ax = ax, cbar = False)
ax.set_ylabel("# blocks\nfragment embedder")
ax.set_xlabel("# blocks embedding to expression")
ax.set_title("OOS-$R^2$")


panel, ax = fig.main.add_right(chd.grid.Panel((2.5, 2.5)))
sns.heatmap(plotdata.set_index(["layerfe", "layere2e"])["time"].unstack(), ax = ax, cbar = False, annot = True)
ax.set_ylabel("")
ax.set_yticklabels([])
ax.set_xlabel("# blocks embedding to expression")
ax.set_title("Time (hours) for 5000 genes")

fig.plot()

manuscript.save_figure(fig, "2", "snote", "layers_heatmap")


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
# ### Multiscale

# %%
dataset_name = "pbmc10k/subsets/top250"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
regions_name = "100k100k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
from params import nfrequencies_tophat_design

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*nfrequencies_tophat_design.index, "counter"])]
design_oi = design_oi.loc[(design_oi["method"].map(len)).sort_values().index]
# design_oi = design_oi.join(nfrequencies_tophat_design)

nfrequencies_tophat_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(nfrequencies_tophat_design.index)]
design_oi["n_frequencies"] = pd.Series(nfrequencies_tophat_design["n_frequencies"].values, nfrequencies_tophat_design["design_ix"]).reindex(design_oi.index)

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
features = pd.DataFrame({
    "value":sorted(list(set([x for fset in plotdata["n_frequencies"].dropna() for x in fset])))
})
features["label"] = [chd.plot.tickers.format_distance(2**round(np.log2(round(200000/x))), add_sign = False) for x in features["value"]]
features["ix"] = features.index
features

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.loc[design_oi.index]

b = design_oi.query("method == 'v33_tophat_[6250]frequencies'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
a = design_oi.query("method == 'v33_tophat_[6250, 3125, 1563, 781, 391, 195, 98, 49, 25, 13]frequencies'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
plotdata_individual_rel = plotdata_individual / plotdata_individual.loc[a]
plotdata_individual_relb = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[a] - plotdata_individual.loc[ref]), 0, 1)
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 0.5, 2.1)
# plotdata_individual_rela = np.clip((plotdata_individual) / (plotdata_individual.loc[b]), 0.5, 2.)

# %%
import chromatinhd.plot.quasirandom
from rpy2.robjects import pandas2ri

# %%
genes_oi = [transcriptome.gene_id("IRF8"), transcriptome.gene_id("PKIA")]

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    for gene in genes_oi:
        plotdata_oi = plotdata_individual_rela.loc[design_ix]
        ax.scatter([x], [plotdata_oi.loc[gene]], color="red", linewidth=0.5, s=10.0)
        if design_ix == design_oi.index[-1]:
            ax.annotate(
                transcriptome.symbol(gene),
                (x, plotdata_oi.loc[gene]),
                (10, 0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", ec="red"),
                ha="left",
                va="center",
            )

    x += 1
ax.plot(xs, ys, color = color)

# ax.set_yscale("log")
ax.set_ylim(0.8, 2.0)
ax.set_yticks([0.8, 1.0, 1.5, 2])
ax.set_yticklabels(["-20%", "Baseline", "+50%", "+100%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks([])

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(chd.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iloc[1:].iterrows():
    n_frequencies = design_row["n_frequencies"]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)
    if not isinstance(n_frequencies, list):
        x += 1
        continue

    features_oi = features.loc[features["value"].isin(n_frequencies)]
    ax.scatter([x] * len(features_oi), features_oi["ix"], color="#333333", linewidth=0.5, s=s)
    ax.plot(
        [x] * len(features_oi),
        features_oi["ix"],
        color="black",
        linewidth = 1.5,
        zorder = -10
    )

    x += 1
ax.set_xticks([])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("Included resolutions", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "2/snote", "multiscale")

# %%
# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))

norm = mpl.colors.CenteredNorm(halfrange=1)
cmap = mpl.cm.get_cmap("RdYlBu_r")
ax.scatter(
    plotdata_individual.loc[b],
    plotdata_individual.loc[a],
    # c=cmap(norm(np.log2(plotdata["ratio"]))),
    color = "#333",
    s=0.1,
)

symbols_oi = [
    "IRF8", "PKIA",
]
offsets = {
    "ITM2C": (-0., 0.2),
    "PKIA": (0.1, -0.1),
}
for symbol in symbols_oi:
    if symbol not in offsets:
        offsets[symbol] = (-0.1, 0.1)

genes_oi = transcriptome.gene_id(symbols_oi)
texts = []
for symbol_oi, gene_oi in zip(symbols_oi, genes_oi):
    x, y = (
        plotdata_individual.loc[b, gene_oi],
        plotdata_individual.loc[a, gene_oi],
    )
    print(symbol_oi)
    text = ax.annotate(
        symbol_oi,
        (x, y),
        xytext=(x + offsets[symbol_oi][0], y + offsets[symbol_oi][1]),
        ha="center",
        va="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color="black", shrinkA=0.0, shrinkB=0.0),
        # bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.5),
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFFAA"),
            mpl.patheffects.Normal(),
        ]
    )
    texts.append(text)
ax.axline((0, 0), slope = 1, color="#999", zorder = -10, lw = 1)

b_label = "Single-scale model"
a_label = "Multiscale model"
ax.set_xlabel(f"OOS-$R^2$\n{b_label}")
ax.set_ylabel(f"OOS-$R^2$\n{a_label}", rotation=0, ha="right", va="center")
ax.set_aspect(1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

manuscript.save_figure(fig, "2/snote", "multiscale_pairwise")

# %%
plotdata_individual_rela.loc[a].sort_values().to_frame().join(transcriptome.var).tail(30)

# %% [markdown]
# ### # Hidden

# %%
from params import nhidden_design

# %%
design_oi = design.query("regions == '100k100k'")
design_oi = design_oi.loc[design_oi["method"].isin(nhidden_design.index)]

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata = plotdata.join(nhidden_design)
plotdata

# %%
fig, ax = plt.subplots(figsize = (2., 2.))
ax.scatter(plotdata["nhidden"], plotdata["r2"], color = "black")
ax.plot(plotdata["nhidden"], plotdata["r2"], color = "black", linestyle = "-")
ax.set_xscale("log")
ax.set_xticks(plotdata["nhidden"])
ax.set_xlabel("size of hidden layers")
ax.set_ylabel("OOS-$R^2$")
ax.set_xticklabels(plotdata["nhidden"], rotation = 90)
ax.set_ylim(0)

manuscript.save_figure(fig, "2", "snote", "nhidden")

# %% [markdown]
# ### Layernorm

# %%
from params import layernorm_design
layernorm_design = layernorm_design.fillna(False)

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*layernorm_design.index, "counter"])]
design_oi = design_oi.iloc[design_oi.set_index("method").index.get_indexer(["counter"] + list(layernorm_design.index))]
# design_oi = design_oi.loc[(design_oi["method"].map(len)).sort_values().index]
layernorm_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(layernorm_design.index)]
design_oi = design_oi.join(layernorm_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
features = pd.DataFrame([
    ["layernorm_fragment_embedder", "LayerNorm fragment embedder"],
    ["layernorm_embedding2expression", "LayerNorm embedding to expression"],
    ["batchnorm_fragment_embedder", "BatchNorm fragment embedder"],
    ["batchnorm_embedding2expression", "BatchNorm embedding to expression"],
], columns = ["value", "label"])
features["ix"] = features.index
features.index = features.value
features

design_oi[features.index] = design_oi[features.index].fillna(False)

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
# plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.reindex(design_oi.index, fill_value = 0.)

b = design_oi.query("method == 'v33_no_layernorm'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
# plotdata_individual_relb = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[a] - plotdata_individual.loc[ref]), 0, 1)
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 0.5, 2.1)
# plotdata_individual_rela = np.clip((plotdata_individual) / (plotdata_individual.loc[b]), 0.5, 2.)

# %%
import chromatinhd.plot.quasirandom
from rpy2.robjects import pandas2ri

# %%
# genes_oi = [transcriptome.gene_id("IRF8"), transcriptome.gene_id("PKIA")]
genes_oi = []

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    for gene in genes_oi:
        plotdata_oi = plotdata_individual_rela.loc[design_ix]
        ax.scatter([x], [plotdata_oi.loc[gene]], color="red", linewidth=0.5, s=10.0)
        if design_ix == design_oi.index[-1]:
            ax.annotate(
                transcriptome.symbol(gene),
                (x, plotdata_oi.loc[gene]),
                (10, 0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", ec="red"),
                ha="left",
                va="center",
            )

    x += 1

# ax.set_yscale("log")
ax.set_ylim(0.8, 2.0)
ax.set_yticks([0.8, 1.0, 1.5, 2])
ax.set_yticklabels(["-20%", "Baseline", "+50%", "+100%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks([])

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(chd.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iloc[1:].iterrows():
    n_frequencies = design_row[features.index]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)

    features_oi = features.loc[n_frequencies]
    ax.scatter([x] * len(features_oi), features_oi["ix"], color="#333333", linewidth=0.5, s=s)
    ax.plot(
        [x] * len(features_oi),
        features_oi["ix"],
        color="black",
        linewidth = 1.5,
        zorder = -10
    )

    x += 1
ax.set_xticks([])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("Included\nnormalizations", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "2/snote", "layernorm")

# %% [markdown]
# ### Positional encoding

# %%
from params import encoder_design

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*encoder_design.index, "counter"])]#.iloc[:7]
encoder_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(encoder_design.index)]
design_oi = design_oi.join(encoder_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
# genes_oi = plotdata.loc[ref].index[plotdata.loc[ref]["cor"] != 0.]

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.loc[design_oi.index]

b = design_oi.query("method == 'v33_direct'").index[0]
a = design_oi.query("method == 'v33_radial_binary'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
# plotdata_individual_relb = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[a] - plotdata_individual.loc[ref]), 0, 1)
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 0.5, 2.1)
# plotdata_individual_rela = np.clip((plotdata_individual) / (plotdata_individual.loc[b]), 0.5, 2.)

# %%
import chromatinhd.plot.quasirandom
from rpy2.robjects import pandas2ri

# %%
# genes_oi = [transcriptome.gene_id("IRF8"), transcriptome.gene_id("PKIA")]
genes_oi = []

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    for gene in genes_oi:
        plotdata_oi = plotdata_individual_rela.loc[design_ix]
        ax.scatter([x], [plotdata_oi.loc[gene]], color="red", linewidth=0.5, s=10.0)
        if design_ix == design_oi.index[-1]:
            ax.annotate(
                transcriptome.symbol(gene),
                (x, plotdata_oi.loc[gene]),
                (10, 0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", ec="red"),
                ha="left",
                va="center",
            )

    x += 1

# ax.set_yscale("log")
ax.set_ylim(0.8, 2.0)
ax.set_yticks([0.8, 1.0, 1.5, 2])
ax.set_yticklabels(["-20%", "Baseline", "+50%", "+100%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks(list(range(len(design_oi)))[:-1])
ax.set_xticklabels(design_oi["encoder"].values[1:], rotation = 90)

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

fig.plot()

manuscript.save_figure(fig, "2/snote", "encoder")

# %%
# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))

norm = mpl.colors.CenteredNorm(halfrange=1)
cmap = mpl.cm.get_cmap("RdYlBu_r")
ax.scatter(
    plotdata_individual.loc[b],
    plotdata_individual.loc[a],
    # c=cmap(norm(np.log2(plotdata["ratio"]))),
    color = "#333",
    s=0.1,
)

ax.axline((0, 0), slope = 1, color="#999", zorder = -10, lw = 1)

a_label = "Model with radial\npositional encoder"
b_label = "Model without\npositional encoder"
ax.set_xlabel(f"OOS-$R^2$\n{b_label}")
ax.set_ylabel(f"OOS-$R^2$\n{a_label}", rotation=0, ha="right", va="center")
ax.set_aspect(1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

manuscript.save_figure(fig, "2/snote", "encoder_pairwise")

# %%
design_oi

# %% [markdown]
# ### N cells

# %%
from params import ncells_design

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*ncells_design.index])]#.iloc[:7]
ncells_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(ncells_design.index)]
design_oi = design_oi.join(ncells_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.loc[design_oi.index]

plotdata_individual_rel = plotdata_individual / plotdata_individual.max(0)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width=0.9))

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
for gene, plotdata_individual_gene in plotdata_individual.loc[:, plotdata_individual.max(0) < 0.7].T.iterrows():
    ax.plot(design_oi["n_cells_train"], plotdata_individual_gene, color = "#33333311")
ax.plot(design_oi["n_cells_train"], plotdata["r2"], lw = 2, marker = "o", color = "#0074D9")
ax.set_xscale("log")
ax.set_xlabel("Number of cells")
ax.set_ylabel("OOS-$R^2$")
ax.set_xticks(design_oi["n_cells_train"])
ax.set_xticklabels(design_oi["n_cells_train"], rotation = 90)
sns.despine(ax = ax)
ax.set_xlim(design_oi["n_cells_train"].min()*0.9, design_oi["n_cells_train"].max()*1.1)
ax.set_ylim(0, 0.7)

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
for gene, plotdata_individual_gene in plotdata_individual_rel.T.iterrows():
    ax.plot(design_oi["n_cells_train"], plotdata_individual_gene, color = "#33333311")
ax.plot(design_oi["n_cells_train"], plotdata_individual_rel.mean(1), lw = 2, marker = "o", color = "#0074D9")
ax.set_xscale("log")
ax.set_xlabel("Number of cells")
ax.set_ylabel("OOS-$R^2$ normalized")
ax.set_xticks(design_oi["n_cells_train"])
ax.set_xticklabels(design_oi["n_cells_train"], rotation = 90)
sns.despine(ax = ax)
ax.set_xlim(design_oi["n_cells_train"].min()*0.9, design_oi["n_cells_train"].max()*1.1)
ax.set_ylim(0)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

fig.plot()

manuscript.save_figure(fig, "2", "snote", "ncells")

# %% [markdown]
# ### Dropout

# %%
from params import dropout_design

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*dropout_design.index])]#.iloc[:7]
dropout_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(dropout_design.index)]
design_oi = design_oi.join(dropout_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.loc[design_oi.index]

plotdata_individual_rel = plotdata_individual / plotdata_individual.max(0)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width=0.9))

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
for gene, plotdata_individual_gene in plotdata_individual.loc[:, plotdata_individual.max(0) < 0.7].T.iterrows():
    ax.plot(design_oi["rate"], plotdata_individual_gene, color = "#33333311")
ax.plot(design_oi["rate"], plotdata["r2"], lw = 2, marker = "o", color = "#0074D9")
ax.set_xlabel("Dropout rate")
ax.set_ylabel("OOS-$R^2$")
ax.set_xticks(design_oi["rate"])
ax.set_xticklabels(design_oi["rate"], rotation = 90)
sns.despine(ax = ax)
ax.set_xlim(design_oi["rate"].min()*0.9, design_oi["rate"].max()*1.1)
ax.set_ylim(0, 0.7)

fig.plot()

manuscript.save_figure(fig, "2", "snote", "dropout")

# %% [markdown]
# ### Residual

# %%
from params import residual_design
residual_design = residual_design.fillna(False)

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*residual_design.index, "counter"])]
design_oi = design_oi.iloc[design_oi.set_index("method").index.get_indexer(["counter"] + list(residual_design.index))]
residual_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(residual_design.index)]
design_oi = design_oi.join(residual_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
features = pd.DataFrame([
    ["residual_fragment_embedder", "Residual connections fragment embedder"],
    ["residual_embedding2expression", "Residual connections embedding to expression"],
], columns = ["value", "label"])
features["ix"] = features.index
features.index = features.value
features

design_oi[features.index] = design_oi[features.index].fillna(False)

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
plotdata_individual = plotdata_individual.reindex(design_oi.index, fill_value = 0.)

b = design_oi.query("method == 'v33_no_residual'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
# plotdata_individual_relb = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[a] - plotdata_individual.loc[ref]), 0, 1)
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 0.5, 2.1)
# plotdata_individual_rela = np.clip((plotdata_individual) / (plotdata_individual.loc[b]), 0.5, 2.)

# %%
import chromatinhd.plot.quasirandom
from rpy2.robjects import pandas2ri

# %%
# genes_oi = [transcriptome.gene_id("IRF8"), transcriptome.gene_id("PKIA")]
genes_oi = []

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    for gene in genes_oi:
        plotdata_oi = plotdata_individual_rela.loc[design_ix]
        ax.scatter([x], [plotdata_oi.loc[gene]], color="red", linewidth=0.5, s=10.0)
        if design_ix == design_oi.index[-1]:
            ax.annotate(
                transcriptome.symbol(gene),
                (x, plotdata_oi.loc[gene]),
                (10, 0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", ec="red"),
                ha="left",
                va="center",
            )

    x += 1

# ax.set_yscale("log")
ax.set_ylim(0.8, 2.0)
ax.set_yticks([0.8, 1.0, 1.5, 2])
ax.set_yticklabels(["-20%", "Baseline", "+50%", "+100%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks([])

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(chd.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iloc[1:].iterrows():
    n_frequencies = design_row[features.index]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)

    features_oi = features.loc[n_frequencies]
    ax.scatter([x] * len(features_oi), features_oi["ix"], color="#333333", linewidth=0.5, s=s)
    ax.plot(
        [x] * len(features_oi),
        features_oi["ix"],
        color="black",
        linewidth = 1.5,
        zorder = -10
    )

    x += 1
ax.set_xticks([])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "2/snote", "residual")

# %% [markdown]
# ### Learning rate

# %%
from params import lr_design

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*lr_design.index])]
lr_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(lr_design.index)]
design_oi = design_oi.join(lr_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.loc[design_oi.index]

plotdata_individual_rel = plotdata_individual / plotdata_individual.max(0)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width=0.9))

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
for gene, plotdata_individual_gene in plotdata_individual.loc[:, plotdata_individual.max(0) < 0.7].T.iterrows():
    ax.plot(design_oi["lr"], plotdata_individual_gene, color = "#33333311")
ax.plot(design_oi["lr"], plotdata["r2"], lw = 2, marker = "o", color = "#0074D9")
ax.set_xscale("log")
ax.set_xlabel("Learning rate")
ax.set_ylabel("OOS-$R^2$")
ax.set_xticks(design_oi["lr"])
ax.set_xticklabels(design_oi["lr"], rotation = 90)
sns.despine(ax = ax)
ax.set_xlim(design_oi["lr"].min()*0.9, design_oi["lr"].max()*1.1)
ax.set_ylim(0, 0.7)

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
for gene, plotdata_individual_gene in plotdata_individual_rel.T.iterrows():
    ax.plot(design_oi["lr"], plotdata_individual_gene, color = "#33333311")
ax.plot(design_oi["lr"], plotdata_individual_rel.mean(1), lw = 2, marker = "o", color = "#0074D9")
ax.set_xscale("log")
ax.set_xlabel("Learning rate")
ax.set_ylabel("OOS-$R^2$ normalized")
ax.set_xticks(design_oi["lr"])
ax.set_xticklabels(design_oi["lr"], rotation = 90)
sns.despine(ax = ax)
ax.set_xlim(design_oi["lr"].min()*0.9, design_oi["lr"].max()*1.1)
ax.set_ylim(0)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

fig.plot()

manuscript.save_figure(fig, "2", "snote", "lr")

# %% [markdown]
# ### Library size normalization

# %%
libsize_design = pd.DataFrame({
    "libsize":[False, True],
    "method":["v33_nolib", "v33"]
}).set_index("method")

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*libsize_design.index, "counter"])]
design_oi = design_oi.iloc[design_oi.set_index("method").index.get_indexer(["counter"] + list(libsize_design.index))]
# design_oi = design_oi.loc[(design_oi["method"].map(len)).sort_values().index]
libsize_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(libsize_design.index)]
design_oi = design_oi.join(libsize_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
features = pd.DataFrame([
    ["libsize", "Library size as a covariate"],
], columns = ["value", "label"])
features["ix"] = features.index
features.index = features.value
features

design_oi[features.index] = design_oi[features.index].fillna(False)

# %%
import scanpy as sc
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")
# sc.pl.umap(transcriptome.adata, color = ["ENSG00000177694", "ENSG00000167483", "ENSG00000269404", "celltype"], layer = "magic")
# sc.pl.umap(transcriptome.adata, color = ["ENSG00000126970"])
genes2 = pd.DataFrame(transcriptome.X[:], index = transcriptome.adata.obs_names, columns = transcriptome.adata.var_names).groupby(transcriptome.obs.celltype).mean().idxmax()
genes2 = genes2[~genes2.isin(["CD4 naive T", "MAIT", "CD4 memory T"])]

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
# plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.reindex(design_oi.index, fill_value = 0.)
plotdata_individual = plotdata_individual.loc[:, plotdata_individual.columns[plotdata_individual.columns.isin(genes2.index)]]

a = design_oi.query("method == 'v33'").index[0]
b = design_oi.query("method == 'v33_nolib'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 1/2.1, 2.1)

# %%
import chromatinhd.plot.quasirandom
from rpy2.robjects import pandas2ri

# %%
# genes_oi = [transcriptome.gene_id("IRF8"), transcriptome.gene_id("PKIA")]
genes_oi = []

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    for gene in genes_oi:
        plotdata_oi = plotdata_individual_rela.loc[design_ix]
        ax.scatter([x], [plotdata_oi.loc[gene]], color="red", linewidth=0.5, s=10.0)
        if design_ix == design_oi.index[-1]:
            ax.annotate(
                transcriptome.symbol(gene),
                (x, plotdata_oi.loc[gene]),
                (10, 0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", ec="red"),
                ha="left",
                va="center",
            )

    x += 1

# ax.set_yscale("log")
ax.set_ylim(0.8, 2.0)
ax.set_yticks([0.8, 1.0, 1.5, 2])
ax.set_yticklabels(["-20%", "Baseline", "+50%", "+100%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks([])

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(chd.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iloc[1:].iterrows():
    n_frequencies = design_row[features.index]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)

    features_oi = features.loc[n_frequencies]
    ax.scatter([x] * len(features_oi), features_oi["ix"], color="#333333", linewidth=0.5, s=s)
    ax.plot(
        [x] * len(features_oi),
        features_oi["ix"],
        color="black",
        linewidth = 1.5,
        zorder = -10
    )

    x += 1
ax.set_xticks([])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "2/snote", "libsize")

# %%
y = pd.DataFrame({
    "diff":plotdata_individual_rela.loc[134].sort_values()
})
y["celltype"] = pd.DataFrame(transcriptome.X[:], index = transcriptome.adata.obs_names, columns = transcriptome.adata.var_names).groupby(transcriptome.obs.celltype).mean().idxmax()
y.groupby("celltype")["diff"].mean().sort_values()

# %%
import scanpy as sc
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")

# %%
transcriptome.obs["fragment_lib"] = np.bincount(fragments.mapping[:, 0])
transcriptome.obs["lib"] = transcriptome.X[:].sum(1)
transcriptome.obs.groupby("celltype")["fragment_lib"].mean().plot()
fig, ax = plt.subplots()
transcriptome.obs.groupby("celltype")["lib"].mean().plot()

# %%
fig, ax = plt.subplots()
for celltype in transcriptome.obs["celltype"].unique():
    sns.ecdfplot(transcriptome.obs.loc[transcriptome.obs["celltype"] == celltype, "lib"])
ax.set_xscale("log")

# %%
fig, ax = plt.subplots()
for celltype in transcriptome.obs["celltype"].unique():
    sns.ecdfplot(transcriptome.obs.loc[transcriptome.obs["celltype"] == celltype, "fragment_lib"])
ax.set_xscale("log")

# %%
fig, ax = plt.subplots(figsize = (2.5, 2.5))
sns.ecdfplot(transcriptome.obs["n_counts"], label = "RNA")
sns.ecdfplot(transcriptome.obs["fragment_lib"], label = "ATAC")
ax.legend()
ax.set_xscale("log")
ax.set_xlabel("Library size")
ax.set_ylabel("Fraction of cells")

manuscript.save_figure(fig, "2/snote", "libsize_variability")

# %% [markdown]
# ### Early stopping

# %%
earlystopping_design = pd.DataFrame({
    "earlystopping":[False, True],
    "method":["v33_noearlystopping", "v33"]
}).set_index("method")

# %%
design_oi = design.query("regions == '100k100k'")
design_oi.index.name = "design_ix"
design_oi = design_oi.loc[design_oi["method"].isin([*earlystopping_design.index, "counter"])]
design_oi = design_oi.iloc[design_oi.set_index("method").index.get_indexer(["counter"] + list(earlystopping_design.index))]
# design_oi = design_oi.loc[(design_oi["method"].map(len)).sort_values().index]
earlystopping_design["design_ix"] = design_oi.index[pd.Index(design_oi["method"]).get_indexer(earlystopping_design.index)]
design_oi = design_oi.join(earlystopping_design.set_index("design_ix"))

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "cor":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["cor"].mean("phase").mean("gene").to_pandas(),
    "r2":(scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"]).mean("phase").mean("gene").to_pandas(),
    "time":scores_oi.sel(design_ix = designs_oi)["time"].mean("gene").to_pandas(),
})
plotdata = design_oi.join(plotdata)
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata

# %%
plotdata_individual = scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["r2"].mean("phase").to_pandas()
# plotdata_individual = plotdata_individual.loc[design_oi.index, plotdata_individual.max(0) > 0.0]
plotdata_individual = plotdata_individual.reindex(design_oi.index, fill_value = 0.)

a = design_oi.query("method == 'v33'").index[0]
b = design_oi.query("method == 'v33_noearlystopping'").index[0]
ref = design_oi.query("method == 'counter'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]
plotdata_individual_rela = np.clip((plotdata_individual - plotdata_individual.loc[ref]) / (plotdata_individual.loc[b] - plotdata_individual.loc[ref]), 1/5.1, 5.1)

# %%
features = pd.DataFrame([
    ["earlystopping", "Early stopping"],
], columns = ["value", "label"])
features["ix"] = features.index
features.index = features.value
features

design_oi[features.index] = design_oi[features.index].fillna(False)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(chd.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iloc[1:].iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_rela.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    x += 1

# ax.set_yscale("log")
ax.set_ylim(0.8, 5.0)
ax.set_yticks([0.8, 1.0, 2, 5])
ax.set_yticklabels(["-20%", "Baseline", "+200%", "+500%"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
ax.set_xticks([])

ax.axhline(1.0, color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Relative performance\n(method - counter) / \n(baseline - counter)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(chd.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iloc[1:].iterrows():
    n_frequencies = design_row[features.index]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)

    features_oi = features.loc[n_frequencies]
    ax.scatter([x] * len(features_oi), features_oi["ix"], color="#333333", linewidth=0.5, s=s)
    ax.plot(
        [x] * len(features_oi),
        features_oi["ix"],
        color="black",
        linewidth = 1.5,
        zorder = -10
    )

    x += 1
ax.set_xticks([])
ax.set_xlim(-0.5, len(design_oi) - 1.5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "2/snote", "earlystopping")

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
ax.plot(regions_design.loc[plotdata_mean.index, "size"], plotdata_mean**2, marker = "o", color = "#0074D9")
ax.set_xscale("log")
lines = []
for i, row in plotdata.T.iterrows():
    row = row.dropna()
    line = ax.plot(regions_design.loc[row.index, "size"], row, color = "#0074D9", alpha = 0.1)
    lines.append(line[0])

plotdata_train = plotdata_train.reindex(region_progression)
plotdata_train_mean = plotdata_train.mean(1)
plotdata_train_mean = plotdata_train_mean.dropna()
ax.plot(regions_design.loc[plotdata_train_mean.index, "size"], plotdata_train_mean**2, marker = "o", color = "orange")

x = np.log10(regions_design.loc[plotdata_mean.index, "size"])
y = plotdata_mean
lm = scipy.stats.linregress(x, y)
ax.plot(10**x, (lm.intercept + x * lm.slope)**2, color = "#0074D9", linestyle = "--")


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
ax.plot(x, y**2, marker = "o", color = "#0074D9")
ax.plot(x, f(x, *popt)**2, color = "black", linestyle = "--")

# %%
