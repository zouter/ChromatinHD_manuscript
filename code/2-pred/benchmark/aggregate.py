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
from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_splitter_peakcaller_predictor_combinations as design_cre2,
)
design_cre2["dataset"] = design_cre2["testdataset"]
design_cre = pd.concat([design_cre, design_cre2], axis=0, ignore_index=True)
design_cre["method"] = design_cre["peakcaller"] + "/" + design_cre["predictor"]
design_cre["group"] = "cre"

from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_method_combinations as design_methods,
)
from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_splitter_method_combinations as design_methods2,
)
design_methods2["dataset"] = design_methods2["testdataset"]
design_methods = pd.concat([design_methods, design_methods2], axis=0, ignore_index=True)
design_methods["group"] = "chd"
design_methods = design_methods.loc[design_methods["method"].isin(["v33", "v33_additive"])]


from chromatinhd_manuscript.designs_pred import (
    dataset_baseline_combinations as design_baseline,
)
design_baseline["group"] = "baseline"

design = pd.concat([design_cre, design_methods, design_baseline], axis=0, ignore_index=True)
design.index.name = "design_ix"

# %%
# design = design.loc[design["regions"].isin(["100k100k"])]
# design = design.loc[design["dataset"].isin(["pbmc10k"])]
# design = design.loc[design["dataset"].isin(["pbmc10k_gran-pbmc10k"])]
# design = design.loc[design["dataset"].isin(["pbmc3k-pbmc10k"])]
# design = design.loc[design["dataset"].isin(["pbmc3k-pbmc10k", "pbmc10k", "pbmc10k_gran-pbmc10k"])]

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
design["label"] = design["dataset"] + "/" + design["regions"] + "/" + design["method"]

# %%
scores = scores.sel(design_ix = scores.coords["design_ix"].isin(design.index))

# %%
# scores["scored"].all("fold").sum("gene").to_dataframe().join(design).query("dataset == 'pbmc3k-pbmc10k'").query("regions == '100k100k'").style.bar(subset = ["scored"])
scores["scored"].all("fold").sum("gene").to_dataframe().join(design).query("method == 'v33_additive'").style.bar(subset = ["scored"])
# scores["scored"].all("fold").sum("gene").to_dataframe().join(design).query("method == 'rolling_500/xgboost'").style.bar(subset = ["scored"])
# scores["scored"].all("fold").sum("gene").to_dataframe().join(design).groupby("method")["scored"].mean().sort_values()

# %%
main_features = ["dataset", "regions", "splitter", "layer"]
reference_method = "v33"

# %%
design["reference"] = design.query("method == @reference_method").reset_index().set_index(main_features).loc[pd.MultiIndex.from_frame(design[main_features])]["design_ix"].values

# %%
r2s = scores["r2"].sel(phase = "test").mean("fold")
r2s.values[~scores["scored"].all("fold").sel(design_ix = design["reference"].values).values] = np.nan
nonan = (~np.isnan(r2s)).sum("gene")
r2 = r2s.mean("gene")

# %%
cors = scores["cor"].sel(phase = "test").mean("fold")
cors.values[~scores["scored"].all("fold").sel(design_ix = design["reference"].values).values] = np.nan
nonan = (~np.isnan(cors)).sum("gene")
cor = cors.mean("gene")

# %%
scored = (scores["scored"].all("fold").values * scores["scored"].all("fold").sel(design_ix = design["reference"].values)).sum("gene").to_pandas()
scored.index = design.index

# %%
from chromatinhd_manuscript.methods import prediction_methods
import chromatinhd.plot.quasirandom

# %%
# TO IMPUTE
# Also uncomment line in next cell
# plotdata = pd.DataFrame(
#     {
#         "r2": r2.to_pandas(),
#         "scored": scored,
#         "cor": np.sqrt(r2.to_pandas()),
#         "nonan": nonan.to_pandas(),
#     }
# ).dropna()
# plotdata["design_ix"] = plotdata.index
# plotdata = plotdata.join(design)

# import sklearn.impute
# import sklearn.experimental.enable_iterative_imputer

# x = plotdata.set_index([*main_features, "method"])["cor"].drop_duplicates().unstack()
# cor_imputed = pd.DataFrame(sklearn.impute.IterativeImputer().fit_transform(x), index = x.index, columns = x.columns)

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
plotdata.index.name = "design_ix"
plotdata = plotdata.join(design)
plotdata["label"] = design.loc[plotdata.index, "label"].values

# IMPUTE
# plotdata["cor"] = design.reset_index().set_index([*main_features, "method"]).join(cor_imputed.stack().rename("cor")).reset_index().set_index("design_ix")["cor"]

plotdata_mean_all = plotdata.groupby("method").agg({"r2": "mean", "design_ix": "first", "cor":"mean", "scored":"mean"}).reset_index()
plotdata_mean_all["n_datasets"] = plotdata.groupby("method").size().values
plotdata_mean_all["datasets"] = plotdata.groupby("method").agg({"dataset": lambda x: ", ".join(x)}).values
plotdata_mean_all[main_features] = "mean"
plotdata_mean_all["label"] = "Mean"
plotdata_mean = plotdata_mean_all.loc[(plotdata_mean_all["n_datasets"] == plotdata_mean_all["n_datasets"].max()) | (plotdata_mean_all["method"].str.startswith("baseline"))]

plotdata = pd.concat([plotdata, plotdata_mean], axis=0, ignore_index=True)

plotdata_grouped = plotdata.groupby(main_features, sort = False)

dataset_info = pd.DataFrame(plotdata_grouped.groups.keys(), columns=main_features).set_index(main_features, drop = False)
dataset_info["label"] = dataset_info["regions"] + " | " + dataset_info["dataset"]
dataset_info.loc[dataset_info["dataset"] == "mean", "label"] = "Mean"
def determine_setting(x):
    if "-" in x:
        return "test"
    elif x == "mean":
        return "mean"
    else:
        return "test_cells"
dataset_info["setting"] = [determine_setting(x) for x in dataset_info["dataset"]]
dataset_info["sorter"] = pd.Series({"test":0, "test_cells":1, "mean":2})[dataset_info["setting"]].values +  plotdata.query("method == @reference_method").set_index(main_features)["r2"].loc[dataset_info.index]
dataset_info = dataset_info.sort_values("sorter")
dataset_info["ix"] = np.arange(len(dataset_info))

plotdata["setting"] = [determine_setting(x) for x in plotdata["dataset"]]

# %%
plotdata_mean_all.sort_values(["n_datasets", "cor"]).style.bar(subset = ["n_datasets", "scored"])

# %%
for settings_oi in [["mean", "test_cells"], ["test"]]:
    dataset_info_oi = dataset_info.loc[dataset_info["setting"].isin(settings_oi)]
    plotdata_grouped = plotdata.loc[plotdata["setting"].isin(settings_oi)].groupby(main_features, sort=False)
    fig, ax = plt.subplots(figsize=(4, len(plotdata_grouped) * 0.25))

    # score = "r2"
    # score = "scored"
    score = "cor"

    for dataset, plotdata_dataset in plotdata_grouped:
        x = dataset_info.loc[dataset]["ix"]

        plotdata_dataset[score] = plotdata_dataset[score].fillna(0.0)
        y = (
            np.array(chd.plot.quasirandom.offsetr(np.array(plotdata_dataset[score].values.tolist()), adjust=0.1)) * 0.8
            + x
        )
        plotdata_dataset["y"] = y

        plotdata_dataset["color"] = prediction_methods.loc[
            design.loc[plotdata_dataset["design_ix"], "method"], "color"
        ].values
        plotdata_dataset.loc[plotdata_dataset["method"] == "v32", "color"] = "pink"
        plotdata_dataset.loc[plotdata_dataset["method"] == "v31", "color"] = "turquoise"

        ax.axhspan(
            x - 0.5,
            x + 0.5,
            color="#33333308" if dataset[0] != "mean" else "#33333315",
            ec="white",
            zorder=-2,
        )
        ax.scatter(plotdata_dataset[score], plotdata_dataset["y"], s=5, color=plotdata_dataset["color"], lw=0)

        plotdata_dataset["type"] = prediction_methods.loc[
            design.loc[plotdata_dataset["design_ix"], "method"], "type"
        ].values
        plotdata_top = plotdata_dataset.sort_values(score, ascending=False).groupby("type").first()
        for i, (type, plotdata_top_) in enumerate(plotdata_top.groupby("type")):
            ax.plot(
                [plotdata_top_[score]] * 2,
                [x - 0.45, x + 0.45],
                color=plotdata_top_["color"].values[0],
                lw=2,
                zorder=0,
                alpha=0.9,
                solid_capstyle="butt",
            )
        plotdata_ours = plotdata_dataset.loc[plotdata_dataset["method"] == reference_method].iloc[0]
        plotdata_top_others = plotdata_top.loc[plotdata_top.index != "ours"]

        try:
            plotdata_others_max = plotdata_top_others.loc[plotdata_top_others[score].idxmax()]
            rectangle = mpl.patches.Rectangle(
                (plotdata_others_max[score], x - 0.45),
                plotdata_ours[score] - plotdata_others_max[score],
                0.9,
                fc=plotdata_ours["color"],
                ec="none",
                zorder=-1,
                alpha=1 / 3,
            )
            ax.add_patch(rectangle)
        except ValueError:
            pass

    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0)
    ax.set_yticks(dataset_info_oi["ix"])
    ax.set_yticklabels(dataset_info_oi["label"])
    for ticklabel in ax.get_yticklabels():
        if ticklabel.get_text() == "Mean":
            ticklabel.set_fontweight("bold")
        else:
            ticklabel.set_fontsize(9)

    if "test_cells" in settings_oi:
        ax.set_xlabel("Test cell $r$")
    else:
        ax.set_xlabel("Test dataset $r$")

    # ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals = 0))
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_ylim(dataset_info_oi["ix"].min() - 0.5, dataset_info_oi["ix"].max() + 0.5)

    if "test_cells" in settings_oi:
        manuscript.save_figure(fig, "2", "aggregate_scores_test_cells")
    else:
        manuscript.save_figure(fig, "2", "aggregate_scores_test_datasets")

# %%
plotdata_mean = plotdata_mean.copy()
plotdata_mean["peakcaller"] = [method.split("/")[0] if method != "v33" else "chd" for method in plotdata_mean["method"]]
plotdata_mean.to_csv(chd.get_output() / "aggregate_prediction.csv")

# %%
plotdata_datasets = plotdata.groupby(["method", "dataset"])[["cor"]].mean().reset_index()
plotdata_datasets["peakcaller"] = [method.split("/")[0] if method != "v33" else "chd" for method in plotdata_datasets["method"]]
plotdata_datasets.to_csv(chd.get_output() / "aggregate_prediction_datasets.csv")

# %%
plotdata_mean

# %%
# plotdata.query("dataset == 'hspc'").query("regions == '10k10k'").sort_values("cor").style.bar(subset=["cor", "r2", "scored"], color="#d65f5f", vmin=0)
plotdata.query("method == 'macs2_improved/lasso'").sort_values("cor").style.bar(subset=["cor", "r2", "scored"], color="#d65f5f", vmin=0)

# %% [markdown]
# ## Detailed analysis for A-B

# %%
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran-pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "hspc"
# dataset_name = "lymphoma"
# dataset_name = "liver"
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
# a = "macs2_leiden_0.1_merged/lasso"
# a = "encode_screen/xgboost"
a = "v33_additive"
# a = "macs2_leiden_0.1_merged/lasso"
# a = "rolling_500/lasso"
# a = "rolling_500/xgboost"
# a = "rolling_100/lasso"
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

a_label = chdm.methods.prediction_methods.loc[a]["label"]
b_label = chdm.methods.prediction_methods.loc[b]["label"]

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
fig, ax = plt.subplots(figsize = (2, 2))
plotdata_oi = plotdata.loc[plotdata[[a, b]].min(1) > 0.05]
ax.hist(np.log((plotdata_oi[b] / plotdata_oi[a])), bins = np.linspace(-1., 1., 50), lw = 0)
ax.set_xticks(np.log([1., 1.5, 2, 1/1.5, 1/2, ]))
ax.set_xticklabels(["1", "1.5", "2",  "⅔", "½"])
ax.set_xlabel(f"R² ChromatinHD /\nR² {a}")

# %%
plotdata_oi = (plotdata["v33"] - plotdata[["rolling_500/xgboost", "macs2_leiden_0.1_merged/xgboost"]].mean(1)).sort_values(ascending = True).to_frame(name = "diff")
plotdata_oi["v33"] = plotdata["v33"]
plotdata_oi[[a, b]] = plotdata[[a, b]]
plotdata_oi["symbol"] = transcriptome.symbol(plotdata_oi.index)
plotdata_oi.head(20)

# %%
plotdata_oi.query("v33 > 0.4").sort_values("diff", ascending = True).head(50)

# %%
plotdata_oi.sort_values("diff", ascending = False).head(50)

# %%
plotdata["diff_abs"] = plotdata["diff"].abs()
plotdata["ratio"] = plotdata[b] / plotdata[a]
# plotdata.reindex(transcriptome.var.query("dispersions_norm > 1.").index).sort_values("diff_abs")[[a, b, "diff"]]
plotdata.reindex(transcriptome.var.query("dispersions_norm > 1.").index).sort_values("diff", ascending = False)[[a, b, "diff"]].dropna().iloc[80]

# %%
# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))

norm = mpl.colors.CenteredNorm(halfrange=1)
cmap = mpl.cm.get_cmap("RdYlBu_r")
ax.scatter(
    plotdata[a],
    plotdata[b],
    # c=cmap(norm(np.log2(plotdata["ratio"]))),
    color = "#3338",
    s=0.1,
)

symbols_oi = []
offsets = {}
if a == "macs2_leiden_0.1_merged":
    symbols_oi = [
        "IRF1",
        "ANXA2R",
        "FOXN2",
    ]
    offsets = {
        "IRF1": (0., 0.18),
        "ANXA2R": (0., 0.4),
        "FOXN2": (0.15, -0.15),
    }
elif a == "v33_additive":
    symbols_oi = [
        "TNFAIP2",
        "CD74",
        "KLF12",
        "BCL2",
    ]
    offsets = {
        "BCL2": (-0.2, 0.),
        "KLF12": (-0.15, 0.),
        "CD74": (-0.15, 0.),
    }
for symbol in symbols_oi:
    if symbol not in offsets:
        offsets[symbol] = (-0.1, 0.1)

genes_oi = transcriptome.gene_id(symbols_oi)
texts = []
for symbol_oi, gene_oi in zip(symbols_oi, genes_oi):
    if gene_oi not in plotdata.index:
        continue
    x, y = (
        plotdata.loc[gene_oi, a],
        plotdata.loc[gene_oi, b],
    )
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

if a == "macs2_leiden_0.1_merged":
    ratio_cutoff = 1.2
elif a == "v33_additive":
    ratio_cutoff = 1.1
percs = (
    (plotdata["ratio"] > ratio_cutoff).mean(),
    (plotdata["ratio"] < 1/ratio_cutoff).mean(),
    1 - (np.abs(np.log(plotdata["ratio"])) > np.log(ratio_cutoff)).mean(),
)
ax.axline((0, 0), (1., 1.*ratio_cutoff), color="black", linestyle="--", lw=0.5)
ax.axline((0, 0), (1., 1./ratio_cutoff), color="black", linestyle="--", lw=0.5)
bbox = dict(boxstyle="round", fc="white", ec="black", lw=0.5)
ax.annotate(
    f"{percs[0]:.1%}",
    (0.03, 0.97),
    (0.0, 0.0),
    textcoords="offset points",
    ha="left",
    va="top",
)
ax.annotate(
    f"{percs[1]:.1%}",
    (0.97, 0.03),
    (0.0, 0.0),
    textcoords="offset points",
    ha="right",
    va="bottom",
)
text = ax.annotate(
    f"{percs[2]:.1%}",
    (0.97, 0.97),
    (0.0, 0.0),
    textcoords="offset points",
    ha="right",
    va="top",
)
text.set_path_effects(
    [mpl.patheffects.Stroke(linewidth=3, foreground="white"), mpl.patheffects.Normal()]
)

polygon = mpl.patches.Polygon(
    [
        (0, 0),
        (1., 1.0*ratio_cutoff),
        # ((1., 1./ratio_cutoff), 1.0),
        (0, 1),
        (0, 0),
    ],
    closed=True,
    fill=True,
    edgecolor="black",
    lw=0.,
    facecolor="#00000022",
    zorder=0,
)
ax.add_patch(polygon)

ax.set_xlabel(f"OOS-$R^2$\n{a_label}")
ax.set_ylabel(f"OOS-$R^2$\n{b_label}", rotation=0, ha="right", va="center")
ax.set_aspect(1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

if a == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "2", "compare_genes_macs2", dpi=300)
elif a == "v33_additive":
    manuscript.save_figure(fig, "2", "compare_genes_v33_additive", dpi=300)

# %%
import pickle
if a == "v33_additive":
    pickle.dump(plotdata, (chd.get_output() / "additive_vs_nonadditive_gene_scores.pkl").open("wb"))

# %% [markdown]
# ### Take individual example

# %%
# dataset_name = "pbmc10k_gran-pbmc10k"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "hspc"
# dataset_name = "lymphoma"
# dataset_name = "liver"
regions_name = "100k100k"
# regions_name = "10k10k"
transcriptome = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
# region_id = "ENSG00000174996"
# region_id = "ENSG00000125347" # nice multiscale IRF1
# region_id = "ENSG00000177721" # >>
# region_id = "ENSG00000197061" # multiscale HIST1H4C
# region_id = "ENSG00000160963" # vs encode screen
# region_id = transcriptome.gene_id("HABP4")
# region_id = transcriptome.gene_id("ZEB2")
# region_id = transcriptome.gene_id("ANXA2R")
# region_id = transcriptome.gene_id("IL4R")
region_id = transcriptome.gene_id("TCF4")
region_id = transcriptome.gene_id("PKIA")
region_id = transcriptome.gene_id("ITM2C")
region_id = transcriptome.gene_id("IRF8")

# examples where both perform equally well
# region_id = "ENSG00000196526"
# region_id = "ENSG00000184307"

# custom
# region_id = transcriptome.gene_id("FOXN2")

print(transcriptome.symbol(region_id))

# %%
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
if dataset_name == "pbmc10k/subsets/top250":
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")

regions_name = "100k100k"
# regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
# dataset_name = "pbmc10k"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")
splitter = "5x1"
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)

model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / splitter / "magic" / "v33"
# model_folder = chd.get_output() / "pred" / dataset_name / "500k500k" / splitter / "magic" / "v34"
# models = chd.models.pred.model.better.Models(model_folder)

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    model_folder / "scoring" / "regionmultiwindow",
)

# %%
# select relevant regions to look at
regions = regionpositional.select_regions(region_id, prob_cutoff = np.exp(-1.5), padding = 800)

# %%
regionmultiwindow.interpolate([region_id], force = True)

# %%
regionmultiwindow.interpolation["interpolated"].sel_xr().to_pandas()[region_id]

# %%
import chromatinhd.data.associations
associations = chd.data.associations.Associations(
    chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / "gwas_immune_main"
    # chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / "gtex_immune"
    # chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / "gtex_caviar_immune"
    # chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / "gtex_caveman_immune"
)

# %%
symbol = transcriptome.var.loc[region_id, "symbol"]

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

score_panel_width = 0.8

breaking = chd.grid.broken.Breaking(regions, 0.05)

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

if "plotdata" in globals():
    pane_predictivity_score, ax = fig.main.add_right(
        chd.grid.Panel((score_panel_width, 0.5)), panel_predictivity
    )
    ax.axis("off")
    ax.barh(0., plotdata.loc[region_id].reindex(["v33"]), lw = 0, fc = chdm.methods.prediction_methods.loc["v33", "color"])
    ax.set_xlim(0, plotdata.loc[region_id, b])
    ax.set_ylim(-0.5, 2.8)

peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name, add_rolling = True)
peakcallers["label"] = chdm.peakcallers["label"]
panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
    fragments.regions.coordinates.loc[region_id], peakcallers, breaking, label_rows = False, label_methods_side = "left"
)
fig.main.add_under(panel_peaks)

if "plotdata" in globals():
    panel_peaks_score, ax = chd.grid.Panel((score_panel_width, panel_peaks[0, 0].height))
    fig.main[fig.main.get_panel_position(panel_peaks)[0], fig.main.get_panel_position(panel_peaks)[1]+1] = panel_peaks_score
    ax.set_ylim(np.array(panel_peaks[0, 0].ax.get_ylim()) -0.5)
    bars = ax.barh(
        np.arange(len(peakcallers)),
        plotdata.loc[region_id].reindex(peakcallers.index + "/lasso"),
        lw = 0,
        height = .9,
    )
    for i, bar, color in zip(range(len(bars)), bars, chdm.peakcallers.loc[peakcallers.index, "color"]):
        bar.set_color(color)
    ax.set_xlim(0, plotdata.loc[region_id, "v33"])
    ax.set_yticks([])
    sns.despine(ax = ax)
    ax.set_xlabel("Test OOS-$R^2$")

# panel_associations = chd.data.associations.plot.AssociationsBroken(associations, region_id, breaking)
# fig.main.add_under(panel_associations)

fig.plot()

# %%
manuscript.save_figure(fig, "2", f"example_raw_{symbol}")

# %% [markdown]
# ## Consistency with test

# %%
traindataset_name = "pbmc10k"
dataset_name = "pbmc10k_gran-pbmc10k"
regions_name = "100k100k"
splitter = "5x5"

dataset_name2 = "pbmc3k-pbmc10k"
dataset_name3 = "pbmc10kx-pbmc10k"

# %%
print(f"{traindataset_name=} {dataset_name=} {regions_name=}")
traindataset_folder = chd.get_output() / "datasets" / traindataset_name

train_fragments = chromatinhd.data.Fragments(traindataset_folder / "fragments" / regions_name)
train_transcriptome = chromatinhd.data.Transcriptome(traindataset_folder / "transcriptome")

folds = chd.data.folds.Folds(traindataset_folder / "folds" / splitter)


dataset_folder = chd.get_output() / "datasets" / dataset_name
fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

dataset_folder2 = chd.get_output() / "datasets" / dataset_name2
fragments2 = chromatinhd.data.Fragments(dataset_folder2 / "fragments" / regions_name)
transcriptome2 = chromatinhd.data.Transcriptome(dataset_folder2 / "transcriptome")

# %%
method_name = "v33"

# %%
# folds = [
#     {
#         "cells_train": np.concatenate([fold["cells_train"], fold["cells_test"]]),
#         # "cells_train": np.concatenate([fold["cells_train"], fold["cells_test"]]),
#         "cells_validation": fold["cells_validation"],
#         "cells_test": fold["cells_test"],
#     }
#     for fold in folds
# ]

folds_test = [
    {
        "cells_test": np.arange(len(fragments.obs)),
        "cells_train": np.array([], dtype = int),
        "cells_validation": np.array([], dtype = int),
    }
    for i in range(len(folds))
]
folds_test2 = [
    {
        "cells_test": np.arange(len(fragments2.obs)),
        "cells_train": np.array([], dtype = int),
        "cells_validation": np.array([], dtype = int),
    }
    for i in range(len(folds))
]

# %%
import chromatinhd.models.pred.model.better
from chromatinhd_manuscript.pred_params import params
method_info = params[method_name]

# %%
layer = "magic"

# %%
models = chd.models.pred.model.better.Models.create(
    fragments=train_fragments,
    transcriptome=train_transcriptome,
    folds=folds,
    model_params={**method_info["model_params"], "layer": layer},
    train_params=method_info["train_params"],
    path = chd.get_output() / "tmp" / "models",
    # overwrite = True,
)
if not models.trained(region_id):
    models.train_models(device="cuda:0", regions_oi = [region_id])

# %%
censorer = chd.models.pred.interpret.censorers.MultiWindowCensorer(fragments.regions.window)
censorer.design = censorer.design.loc[~(
    (censorer.design["window_start"].values[None, :] > regions["end"].values[:, None]) | 
    (censorer.design["window_end"].values[None, :] < regions["start"].values[:, None])).all(0)
]

# %%
regionmultiwindow_train = chd.models.pred.interpret.RegionMultiWindow.create(
    path=chd.get_output() / "tmp" / "regionmultiwindow_train",
    folds=folds,
    transcriptome=train_transcriptome,
    fragments=train_fragments,
    censorer=censorer,
    # overwrite = True
)
if not regionmultiwindow_train.scores["scored"].sel_xr(region_id).all():
    regionmultiwindow_train.score(models, device="cpu", regions = [region_id])

# %%
regionmultiwindow_train.interpolate([region_id], force = True)

# %%
regionmultiwindow_test = chd.models.pred.interpret.RegionMultiWindow.create(
    path=chd.get_output() / "tmp" / "regionmultiwindow_test",
    folds=folds_test,
    transcriptome=transcriptome,
    fragments=fragments,
    censorer=censorer,
    overwrite = True
)
if not regionmultiwindow_test.scores["scored"].sel_xr(region_id).all():
    regionmultiwindow_test.score(models, device="cpu", regions = [region_id])
# regionmultiwindow_test.score(models, fragments = fragments, transcriptome = transcriptome, folds = folds_test, device="cuda:0")

# %%
regionmultiwindow_test.interpolate([region_id], force = True)

# %%
regionmultiwindow_test2 = chd.models.pred.interpret.RegionMultiWindow.create(
    path=chd.get_output() / "tmp" / "regionmultiwindow_test2",
    folds=folds_test2,
    transcriptome=transcriptome2,
    fragments=fragments2,
    censorer=censorer,
    # overwrite = True
)
# regionmultiwindow_test2.score(models, fragments = fragments2, transcriptome = transcriptome2, folds = folds_test2, device="cuda:0")

# %%
regionmultiwindow_test2.interpolate([region_id], force = True)

# %%
import scanpy as sc

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

breaking.resolution = 10000

panel_pileup = fig.main.add_under(
    chd.models.pred.plot.PileupBroken.from_regionmultiwindow(regionmultiwindow_train,region_id,  breaking, height = 0.5)
)
panel_pileup[0, 0].ax.set_ylabel("# fragments per 1kb per 1k cells\nTrain dataset, test cells", ha = "right")
panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow_train,region_id,  breaking, height = 0.5, ymax = -0.1)
)
panel_predictivity[0, 0].ax.set_ylabel("$\Delta$ cor\nTrain dataset, test cells", ha = "right")
panel_predictivity[0, 0].ax.set_yticks([0, -0.08])

panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow_test,region_id,  breaking, height = 0.5, ymax = -0.1), padding = 0.
)
panel_predictivity[0, 0].ax.set_ylabel("$\Delta$ cor\nTest dataset, 10k cells", ha = "right")
panel_predictivity[0, 0].ax.set_yticks([0, -0.08])

panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow_test2,region_id,  breaking, height = 0.5, ymax = -0.1), padding = 0.
)
panel_predictivity[0, 0].ax.set_ylabel("$\Delta$ cor\nTest dataset, 3k cells", ha = "right")
panel_predictivity[0, 0].ax.set_yticks([0, -0.08])

fig.plot()

# %%
windows_oi = censorer.design.query("(window_start > 8500) & (window_end < 14975)").query("window_size == 500").sort_values("window_start")

# %%
plt.plot(regionmultiwindow_train.scores["deltacor"].sel_xr(region_id).mean("fold").sel(phase = "test").sel(window = windows_oi.index))
plt.plot(regionmultiwindow_test.scores["deltacor"].sel_xr(region_id).mean("fold").sel(phase = "test").sel(window = windows_oi.index))
plt.plot(regionmultiwindow_test2.scores["deltacor"].sel_xr(region_id).mean("fold").sel(phase = "test").sel(window = windows_oi.index))

# %%
fig, ax = plt.subplots()
ax.scatter(regionmultiwindow_train.get_plotdata(region_id)["lost"], regionmultiwindow_test.get_plotdata(region_id)["lost"])
ax.scatter(regionmultiwindow_train.get_plotdata(region_id)["lost"], regionmultiwindow_test2.get_plotdata(region_id)["lost"])
# ax.set_xscale("log")
# ax.set_yscale("log")

# %%
fig, ax = plt.subplots(figsize = (1.1, 1.1))
ax.scatter(regionmultiwindow_train.get_plotdata(region_id)["deltacor"], regionmultiwindow_test.get_plotdata(region_id)["deltacor"], s = 3, color = "black", lw = 0, alpha = 0.5)
ax.scatter(regionmultiwindow_train.get_plotdata(region_id)["deltacor"], regionmultiwindow_test2.get_plotdata(region_id)["deltacor"], s = 3, color = "black", lw = 0, alpha = 0.5)

import scipy.stats
lm = scipy.stats.linregress(regionmultiwindow_train.get_plotdata(region_id)["deltacor"], regionmultiwindow_test.get_plotdata(region_id)["deltacor"])
print(lm.rvalue**2)

lm = scipy.stats.linregress(regionmultiwindow_train.get_plotdata(region_id)["deltacor"], regionmultiwindow_test2.get_plotdata(region_id)["deltacor"])
print(lm.rvalue**2)

# %%
transcriptome2.var["all_zero"] = (transcriptome2.layers["normalized"][:] == 0).all(0)

# %%
symbol = transcriptome.var.loc[region_id, "symbol"]

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

score_panel_width = 0.45

regions = regionpositional.select_regions(region_id, prob_cutoff = np.exp(-1.5), padding = 800)
breaking = chd.grid.broken.Breaking(regions, 0.05, resolution = 7500)

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
    chd.models.pred.plot.PileupBroken.from_regionmultiwindow(regionmultiwindow_train,region_id,  breaking, height = 0.5)
)
panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow_train,region_id,  breaking, height = 0.5, ymax = -0.1)
)
pane_predictivity_score, ax = fig.main.add_right(
    chd.grid.Panel((score_panel_width, 0.5)), panel_predictivity
)
ax.barh(0., plotdata.loc[region_id].reindex(["v33"]), lw = 0, fc = chdm.methods.prediction_methods.loc["v33", "color"])

cor_max = plotdata.loc[region_id, b]
cor_lim = 0.5

ax.set_xlim(0, cor_lim)
ax.set_ylim(-0.5, 2.8)
ax.set_yticks([])
ax.set_xticklabels([])
sns.despine(ax = ax)
# ax.axvline(cor_max, color = prediction_methods.loc["v33", "color"], dashes = (1, 1), lw = 1)

peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name, add_rolling = True)
peakcallers["label"] = chdm.peakcallers["label"]
panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
    fragments.regions.coordinates.loc[region_id], peakcallers, breaking, label_rows = False, label_methods_side = "left"
)
fig.main.add_under(panel_peaks)

panel_peaks_score, ax = chd.grid.Panel((score_panel_width, panel_peaks[0, 0].height))
fig.main[fig.main.get_panel_position(panel_peaks)[0], fig.main.get_panel_position(panel_peaks)[1]+1] = panel_peaks_score
ax.set_ylim(np.array(panel_peaks[0, 0].ax.get_ylim()) -0.5)
bars = ax.barh(
    np.arange(len(peakcallers)),
    plotdata.loc[region_id].reindex(peakcallers.index + "/lasso"),
    lw = 0,
    height = .9,
)
for i, bar, color in zip(range(len(bars)), bars, chdm.peakcallers.loc[peakcallers.index, "color"]):
    bar.set_color(color)
ax.set_xlim(0, cor_lim)
ax.set_xticklabels([])
ax.set_yticks([])
sns.despine(ax = ax)
# ax.axvline(cor_max, color = prediction_methods.loc["v33", "color"], dashes = (1, 1), lw = 1)


# add test delta cor
panel_predictivity = fig.main.add_under(
    chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow_test,region_id,  breaking, height = 0.5, ymax = -0.1), padding = 0.
)
panel_predictivity[0, 0].ax.set_ylabel("$\Delta$ cor\nTest dataset, 10k cells", ha = "right")

# add test score
design_ix = design.query("(dataset == @dataset_name) & (regions == @regions_name) & (splitter == '5x1') & (layer == 'magic') & (method == 'v33')").index
print(design_ix)
test_cor = scores.sel(gene = region_id, phase = "test", design_ix = design_ix)["cor"].mean()

panel_chd_score, ax = chd.grid.Panel((score_panel_width, 0.5))
fig.main[fig.main.get_panel_position(panel_predictivity)[0], fig.main.get_panel_position(panel_predictivity)[1]+1] = panel_chd_score
ax.set_yticks([])
ax.set_xticklabels([])
sns.despine(ax = ax)
ax.barh(0., test_cor, lw = 0, fc = chdm.methods.prediction_methods.loc["v33", "color"])
ax.set_xlim(0, cor_lim)
ax.set_ylim(-0.5, 2.8)
# ax.axvline(cor_max, color = prediction_methods.loc["v33", "color"], dashes = (1, 1), lw = 1)

if not transcriptome2.var.loc[region_id, "all_zero"]:
    # add test delta cor
    panel_predictivity = fig.main.add_under(
        chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(regionmultiwindow_test2,region_id,  breaking, height = 0.5, ymax = -0.1), padding = 0.
    )
    panel_predictivity[0, 0].ax.set_ylabel("$\Delta$ cor\nTest dataset, 3k cells", ha = "right")

    # add test score
    design_ix = design.query("(dataset == @dataset_name2) & (regions == @regions_name) & (splitter == '5x1') & (layer == 'magic') & (method == 'v33')").index
    print(design_ix)
    test_cor = scores.sel(gene = region_id, phase = "test", design_ix = design_ix)["cor"].mean()

    panel_chd_score, ax = chd.grid.Panel((score_panel_width, 0.5))
    fig.main[fig.main.get_panel_position(panel_predictivity)[0], fig.main.get_panel_position(panel_predictivity)[1]+1] = panel_chd_score
    ax.barh(0., test_cor, lw = 0, fc = chdm.methods.prediction_methods.loc["v33", "color"])
    ax.set_yticks([])
    ax.set_xlim(0, cor_lim)
    ax.set_ylim(-0.5, 2.8)
    ax.set_xlabel("Test $r$")
    sns.despine(ax = ax)
    # ax.set_xticklabels(ax.get_xticklabels())
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    # ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals = 0))
    # ax.axvline(cor_max, color = prediction_methods.loc["v33", "color"], dashes = (1, 1), lw = 1)

fig.plot()

# %%
manuscript.save_figure(fig, "2", f"example_{symbol}")

# %% [markdown]
# ### Kurtosis

# %%
plotdata = regionmultiwindow.get_plotdata(region_id)

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

# %% [markdown]
# ## Train - test

# %%
dataset_name1 = "pbmc10k"
# dataset_name2 = "pbmc10k_gran-pbmc10k"
dataset_name2 = "pbmc10kx-pbmc10k"
regions_name = "100k100k"

# %%
design_oi1 = design.query("dataset == @dataset_name1").query("regions == @regions_name").query("splitter == '5x1'").query("layer == 'magic'")
design_oi2 = design.query("dataset == @dataset_name2").query("regions == @regions_name").query("splitter == '5x1'").query("layer == 'magic'")

# %%
plotdata1 = r2s.sel(design_ix = design_oi1.index).to_pandas().T
plotdata1.columns = design_oi1["method"]
plotdata1 = plotdata1.loc[~pd.isnull(plotdata1["v33"])]

plotdata2 = r2s.sel(design_ix = design_oi2.index).to_pandas().T
plotdata2.columns = design_oi2["method"]
plotdata2 = plotdata2.loc[~pd.isnull(plotdata2["v33"])]

# %%
a = "macs2_leiden_0.1_merged/lasso"
# a = "encode_screen/xgboost"
# a = "macs2_leiden_0.1_merged/xgboost"
# a = "rolling_500/lasso"
# a = "rolling_500/xgboost"
# a = "rolling_100/lasso"
# a = "rolling_100/lasso"
b = "v33"
 
# plotdata = plotdata.loc[(plotdata[a] > 0.) & (plotdata[b] > 0.)]
plotdata1.shape[0], plotdata2.shape[0]

# %%
fig, ax = plt.subplots()
ax.scatter(
    transcriptome.var["means"].values, plotdata2[b] - plotdata2[a], s = 1
)
ax.set_xscale("symlog")
fig, ax = plt.subplots()
ax.scatter(
    transcriptome.var["means"].values, plotdata1[b] - plotdata1[a], s = 1
)
ax.set_xscale("symlog")
# ax.scatter(plotdata1[a], plotdata1[b], s = 1)
# ax.scatter(plotdata2[a], plotdata2[b], s = 1)
