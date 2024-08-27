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

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
from design import (
    dataset_splitter_method_combinations as design_methods,
)
design_methods["group"] = "chd"

design = pd.concat([design_methods], axis=0, ignore_index=True)

# %%
scores = {}

for design_ix, design_row in design.query("group == 'chd'").iterrows():
    prediction = chd.flow.Flow(
        chd.get_output() / "diff" / design_row.dataset / design_row.regions / design_row.splitter / design_row.clustering / design_row.method,
    )
    performance = chd.models.diff.interpret.Performance(prediction.path / "scoring" / "performance")

    try:
        scores[design_ix] = performance.scores.sel_xr()
        # raise ValueError("Scores already exist")
    except FileNotFoundError:
        continue

scores = xr.concat(scores.values(), dim = pd.Index(scores.keys(), name = "design_ix"))

# %% [markdown]
# ## Overall

# %%
design_oi = design
# design_oi = design.query("regions == '10k10k'")
design_oi = design.query("regions == '100k100k'")

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi.mean("fold").sel(phase = "test")["scored"].mean("gene").to_pandas()
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata.plot.barh()

# %%
import scipy.stats

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi.mean("fold").sel(phase = "validation")["likelihood"].mean("gene").to_pandas()
plotdata = plotdata[plotdata != 0]
# plotdata = plotdata - plotdata.min()
plotdata.index = design.iloc[plotdata.index]["method"].values
fig, ax = plt.subplots()
ax.scatter(plotdata, plotdata.index)

# %% [markdown]
# ## Comparison

# %% [markdown]
# ### Time

# %%
colors = ["#FF4136", "#0074D9", "#B10DC9", "#2ECC40"]
plotdata = pd.DataFrame(
    [
        ["v31", 3.13, "ChromatinHD", colors[1]],
        ["macs2_summits/t-test", 1/6.19, "MACS2 summits t-test", colors[0]],
        ["macs2_summits/wilcoxon", 1/4.3, "MACS2 summits wilcoxon (Signac)", colors[0]],
        ["macs2_summits/snap", 1/5.42, "MACS2 summits edgeR (snapATAC)", colors[0]],
        ["macs2_summits/logreg", 1/1.4, "MACS2 summits logreg", colors[0]],
        ["rolling_50/t-test", 3.18, "50bp window t-test", colors[2]],
        ["rolling_50/wilcoxon", 4.23, "50bp window wilcoxon (Signac)", colors[2]],
        ["rolling_50/snap", 3.72, "50bp window edgeR (snapATAC)", colors[2]],
        ["rolling_50/logreg", 38.02, "50bp window logreg", colors[2]],
        ["rolling_100/t-test", 2.6, "100bp window t-test", colors[2]],
        ["rolling_100/wilcoxon", 3.6, "100bp window wilcoxon (Signac)", colors[2]],
        ["rolling_100/snap", 2.4, "100bp window edgeR (snapATAC)", colors[2]],
        ["rolling_100/logreg", 15.90, "100bp window logreg", colors[2]],
        ["rolling_500/t-test", 1/2.21, "500bp window t-test", colors[2]],
        ["rolling_500/wilcoxon", 1/1.5, "500bp window wilcoxon (Signac)", colors[2]],
        ["rolling_500/snap", 1/1.80, "500bp window edgeR (snapATAC)", colors[2]],
        ["rolling_500/logreg", 1.60, "500bp window logreg", colors[2]],
        ["encoder_screen/t-test", 1/2.72, "Encoder screen t-test", colors[3]],
        ["encoder_screen/wilcoxon", 1/3.95, "Encoder screen wilcoxon (Signac)", colors[3]],
        ["encoder_screen/snap", 1/3.12, "Encoder screen edgeR (snapATAC)", colors[3]],
        ["encoder_screen/logreg", 1/1.5, "Encoder screen logreg", colors[3]],
    ], columns = ["method", "time", "label", "color"]
)
plotdata["time"] = plotdata["time"] * 5000 / 60 / 60
plotdata["time_peakcalling"] = [0.24 if "macs2" in x else 0 for x in plotdata["method"]]

# %%
from chromatinhd_manuscript.methods import differential_methods

# %%
plotdata["ix"] = np.arange(len(plotdata))

fig, ax = plt.subplots(figsize = (3, 5))

ax.barh(plotdata["ix"], plotdata["time_peakcalling"], color = plotdata["color"], lw = 0., hatch = "\\\\\\")
ax.barh(plotdata["ix"], plotdata["time"], left = plotdata["time_peakcalling"], color = plotdata["color"], lw = 0.)

xlim = 7
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
for i, row in plotdata.iterrows():
    time = row["time"] + row["time_peakcalling"]
    if time < xlim:
        ax.text(time, row["ix"], f" {format_time(time)}", ha = "left", va = "center", fontsize = 10)
    else:
        ax.text(xlim, row["ix"], "→", ha = "left", va = "center", fontsize = 10, color = row["color"], fontweight = "bold")
        ax.text(xlim, row["ix"], f" {format_time(time)}", ha = "right", va = "center", fontsize = 10, color = "white")

ax.set_yticks(plotdata["ix"])
ax.set_yticklabels(plotdata["label"])

ax.set_xlabel("Total time (hours)")

handles = [
    ["Peak calling", mpl.patches.Rectangle((0, 0), 1, 1, fc = "#333", hatch = "\\\\\\")],
    ["Training +\ninterpretation", mpl.patches.Rectangle((0, 0), 1, 1, fc = "#333")],
]
ax.legend([handle for label, handle in handles], [label for label, handle in handles], loc = "lower right")

sns.despine()
ax.set_ylim(len(plotdata)-0.5, -0.5)
ax.set_xlim(0, xlim)

manuscript.save_figure(fig, "3", "snote", "time")

# %%
plotdata["ix"] = np.arange(len(plotdata))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (4, 5))
fig.subplots_adjust(wspace=0.05)  # adjust space between axes

ax1.barh(plotdata["ix"], plotdata["time"], color = plotdata["color"], lw = 0.)
ax2.barh(plotdata["ix"], plotdata["time"], color = plotdata["color"], lw = 0.)

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
for i, row in plotdata.iterrows():
    if row["time"] < 6.2:
        ax1.text(row["time"], row["ix"], f" {format_time(row['time'])}", ha = "left", va = "center", fontsize = 10, clip_on = False)
    else:
        ax2.text(row["time"], row["ix"], f" {format_time(row['time'])}", ha = "left", va = "center", fontsize = 10, clip_on = False)

ax1.set_yticks(plotdata["ix"])
ax1.set_yticklabels(plotdata["label"])

ax1.tick_params(labeltop=True, labelbottom = False)
ax1.set_xlabel("Total time (hours)")

ax1.set_ylim(len(plotdata)-0.5, -0.5)
ax2.set_ylim(len(plotdata)-0.5, -0.5)

# hide the spines between ax and ax2
ax1.spines.right.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
ax1.xaxis.tick_top()
ax2.xaxis.tick_top()
ax2.set_yticks([])

d = 2.  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

ax1.set_xlim(0, 8.2)
ax2.set_xlim(21, 53)

manuscript.save_figure(fig, "3", "snote", "time")

# %% [markdown]
# ### W Delta regularisation

# %%
design_oi = design.query("regions == '100k100k'")
design_oi = design_oi.loc[design_oi["method"].isin(["v31", "v31_wdeltareg-no"])]

# %%
a = design_oi.index[design_oi["method"] == "v31"][0]
b = design_oi.index[design_oi["method"] == "v31_wdeltareg-no"][0]

# %%
fig, ax = plt.subplots(figsize = (1.5, 1.))
ax.axvline(0, color = "black", linestyle = "--")
sns.ecdfplot(scores.mean("fold").sel(design_ix = a).sel(phase = "train")["likelihood"].values - scores.mean("fold").sel(design_ix = b).sel(phase = "train")["likelihood"].values)
sns.ecdfplot(scores.mean("fold").sel(design_ix = a).sel(phase = ["test", "validation"]).mean("phase")["likelihood"].values - scores.mean("fold").sel(design_ix = b).sel(phase = ["test", "validation"]).mean("phase")["likelihood"].values)

ax.set_xlim(-0.7, 0.7)
ax.set_xticks(np.log([1/1.5, 1., 1.5]))
ax.set_xticklabels(["2/3", "1", "1.5"])

ax.text(0.1, 0.9, "Train", ha = "left", va = "top", transform = ax.transAxes, color = sns.color_palette()[0])
ax.text(0.9, 0.1, "Test", ha = "right", va = "bottom", transform = ax.transAxes, color = sns.color_palette()[1])

ax.set_ylabel("Genes")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_xlabel("Test likelihood ratio\nwith and without regularization", loc = "right")

manuscript.save_figure(fig, "3/snote", "regularization")

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi.mean("fold").sel(phase = "test")["likelihood"].mean("gene").to_pandas()
plotdata.index = design.iloc[plotdata.index]["method"].values

# %%
plotdata

# %% [markdown]
# ### W delta scale

# %%
from params import w_delta_p_scale_design, w_delta_p_scale_titration_ids
w_delta_p_scale_design.index = w_delta_p_scale_design.label

# %%
design_oi = design.query("regions == '100k100k'")
# design_oi = design_oi.loc[design_oi["method"].isin(w_delta_p_scale_design.label)]
design_oi = design_oi.loc[design_oi["method"].isin(w_delta_p_scale_titration_ids)]

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "likelihood":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["likelihood"].mean("phase").mean("gene").to_pandas(),
})

# %%
plotdata["method"] = design_oi.loc[plotdata.index, "method"]

# %%
plotdata["likelihood"].plot()

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
designs_oi = scores_oi["scored"].any(["gene", "fold"])
plotdata = pd.DataFrame({
    "test_likelihood":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["test", "validation"])["likelihood"].mean("phase").mean("gene").to_pandas(),
    "train_likelihood":scores_oi.sel(design_ix = designs_oi).mean("fold").sel(phase = ["train"])["likelihood"].mean("phase").mean("gene").to_pandas(),
})
plotdata["design_ix"] = plotdata.index
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata = plotdata.join(w_delta_p_scale_design)
plotdata["w_delta_p_scale"] = plotdata.index.str.split("-").str[-1].astype(float)
plotdata

# %%
fig, ax = plt.subplots(figsize = (2.7, 1.))
ax.plot(plotdata["w_delta_p_scale"], plotdata["test_likelihood"], color = sns.color_palette()[1], linestyle = "-", marker = "o")
ax.plot(plotdata["w_delta_p_scale"], plotdata["train_likelihood"], color = sns.color_palette()[0], linestyle = "-", marker = "o")
ax.set_xscale("log")
ax.set_xticks(plotdata["w_delta_p_scale"])
ax.set_xlabel("$\sigma_a$")
ax.set_ylabel("Log-likelihood")
ax.set_xticklabels(plotdata["w_delta_p_scale"], rotation = 90)

manuscript.save_figure(fig, "3", "snote", "w_delta_p_scale")

# %% [markdown]
# ### Multiscale

# %%
from params import binwidth_combinations
from params import binwidth_titration

# %%
design_oi = design.loc[design.reset_index().groupby("method").first().reset_index().set_index("index").reset_index().set_index("method").loc[[*binwidth_titration["label"][::-1]]]["index"]]
design_oi = design_oi.loc[design_oi["regions"] == "100k100k"]

# %%
binwidth_titration.index = [design_oi.index[design_oi["method"] == method][0] for method in binwidth_titration["label"]]
design_oi["binwidths"] = binwidth_titration["binwidths"]

# %%
features = pd.DataFrame({
    "value":sorted(list(set([x for fset in binwidth_titration["binwidths"].dropna() for x in fset])))
})
features["label"] = [chd.plot.tickers.format_distance(x, add_sign = False) for x in features["value"]]
features["ix"] = features.index
features

# %%
plotdata_individual = scores.reindex(design_ix = design_oi.index, fill_value = np.nan).mean("fold").sel(phase = ["validation", "test"])["likelihood"].mean("phase").to_pandas()
# plotdata_individual.index = design_oi["method"]
b = design_oi.query("method == 'v31_(25,)bw'").index[0]
plotdata_individual_diff = plotdata_individual - plotdata_individual.loc[b]

# %%
plotdata = plotdata_individual_diff.mean(1).to_frame().reset_index()

# %%
import chromatinhd.plot.quasirandom

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

resolution = 0.35
width = plotdata.shape[0] * resolution

panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 2)))

color = "#333"

xs = []
ys = []
x = 0.0
for method_id, row in plotdata.iterrows():
    xs.append(x)
    design_ix = row["design_ix"]
    plotdata_oi = plotdata_individual_diff.loc[design_ix]

    x2 = chd.plot.quasirandom.offsetr(plotdata_oi.values)
    ax.scatter(x2 * 0.75 + x, [plotdata_oi], color="#888", linewidth=0.5, s=1.0)

    y = plotdata_oi.mean()
    ys.append(y)
    ax.scatter([x], [y], color=color, linewidth=0.5)

    x += 1
ax.plot(xs, ys, color = color)

ax.set_ylim(-0.7, 0.7)
ax.set_yticks(np.log([1/1.5, 1/1.2, 1/1.1, 1., 1.1, 1.2, 1.5]))
ax.set_yticklabels(["2/3", "1/1.2", "1/1.1", "1", "1.1", "1.2", "1.5"])

# ax.set_yticks([0, 0.5, 0.8, 1])
# ax.set_yticklabels(["Baseline", "50%", "Full model"])
ax.set_xlim(-0.5, len(design_oi) - .5)
ax.set_xticks([])

ax.axhline(0., color="black", linestyle="--", zorder=-10)

ax.set_ylabel("Test-cell\nlog-likelihood ratio\n(average per cell)", rotation=0, ha="right", va="center")

sns.despine(ax=ax)

resolution_features = 0.14
s = 50

panel, ax = fig.main.add_under(polyptich.grid.Panel((width, len(features) * resolution_features)), padding=0.1)

ax.set_ylim(-0.5, features["ix"].max() + 0.5)
ax.set_yticks(features["ix"])
ax.set_yticklabels(features["label"])

x = 0.0
for design_ix, design_row in design_oi.iterrows():
    binwidths = design_row["binwidths"]
    ax.scatter([x] * len(features), features["ix"], color="#33333333", linewidth=0.5, s=s)

    features_oi = features.loc[features["value"].isin(binwidths)]
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
ax.set_xlim(-0.5, len(design_oi) - .5)
sns.despine(ax=ax, bottom=True)
ax.set_ylabel("Included resolutions", rotation=0, ha="right", va="center")

fig.plot()

manuscript.save_figure(fig, "3/snote", "multiscale")

# %% [markdown]
# ## Pairwise comparison

# %%
design_oi = design.query("regions == '100k100k'")
# design_oi = design.query("regions == '10k10k'")

# %%
# a = design_oi.query("method == 'binary_50bw'").index[0]
a = design_oi.query("method == 'binary_shared_[5k,1k,500,100,50,25]bw'").index[0]
# a = design_oi.query("method == 'binary_1000-50bw_earlystop-no'").index[0]
b = design_oi.query("method == 'binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop_100epochs'").index[0]

plotdata = scores["likelihood"].sel(phase = "test").mean("fold").to_pandas().T
# plotdata = plotdata.loc[(plotdata[b] != 0) & (plotdata[a] != 0)] # remove zero values

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

fig = polyptich.grid.Figure(polyptich.grid.Wrap(padding_width = 0.7, ncol = 4))

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
plotdata["kurtosis"] = (scipy.stats.kurtosis(transcriptome.layers["magic"]))
plotdata["kurtosis_rank"] = scipy.stats.rankdata(plotdata["kurtosis"])

a_label = str(a)
b_label = str(b)

cmap = mpl.colormaps["Set1"]

# rank vs diff
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
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
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
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
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
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
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["dispersions"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("dispersion")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["dispersions"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions vs diff
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["log10means"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10means")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10means"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions_norm vs diff
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["log10dispersions_norm"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10dispersions_norm")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10dispersions_norm"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["log10n_fragments"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10n_fragments")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10n_fragments"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments std
panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["kurtosis"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("kurtosis")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["kurtosis"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

fig.plot()

# %%
plotdata.loc[transcriptome.gene_id("EBF1")]["log10n_fragments"]
