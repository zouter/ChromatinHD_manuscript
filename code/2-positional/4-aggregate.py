# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

import pickle

import scanpy as sc
import pathlib

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
from chromatinhd_manuscript.designs import dataset_splitter_peakcaller_predictor_combinations as design_peaks
design_peaks = design_peaks.loc[design_peaks["predictor"] != "xgboost"].copy()

from chromatinhd_manuscript.designs import dataset_splitter_method_combinations as design_methods

from chromatinhd_manuscript.designs import traindataset_testdataset_splitter_method_combinations as design_methods_traintest
design_methods_traintest["dataset"] = design_methods_traintest["testdataset"]
design_methods_traintest["splitter"] = "all"

from chromatinhd_manuscript.designs import traindataset_testdataset_splitter_peakcaller_predictor_combinations as design_peaks_traintest
design_peaks_traintest["dataset"] = design_peaks_traintest["testdataset"]
design_peaks_traintest["splitter"] = "all"

# %%
design_peaks["method"] = design_peaks["peakcaller"] + "/" + design_peaks["predictor"]
design_peaks_traintest["method"] = design_peaks_traintest["peakcaller"] + "/" + design_peaks_traintest["predictor"]

# %%
design = pd.concat([design_peaks, design_methods, design_methods_traintest, design_peaks_traintest])
design.index = np.arange(len(design))
design.index.name = "design_ix"

# %%
# design = design.query("dataset == 'pbmc10k'").copy()
# design = design.query("dataset == 'pbmc10k_gran-pbmc10k'").copy()
# design = design.query("dataset == 'pbmc10k_gran-pbmc10k'").copy()

# %%
design = design.query("splitter in ['random_5fold', 'all']").copy()
design = design.query("method != 'v20_initdefault'").copy()

# %%
design["traindataset"] = [x["dataset"] if pd.isnull(x["traindataset"]) else x["traindataset"] for _, x in design.iterrows()]


# %%
class Prediction(chd.flow.Flow):
    pass


# %%
scores = {}
design["found"] = False
for design_ix, design_row in design.iterrows():
    prediction = Prediction(
        chd.get_output()
        / "prediction_positional"
        / design_row["dataset"]
        / design_row["promoter"]
        / design_row["splitter"]
        / design_row["method"]
    )
    if (prediction.path / "scoring" / "overall" / "genescores.pkl").exists():
        # print(prediction.path)
        genescores = pd.read_pickle(
            prediction.path / "scoring" / "overall" / "genescores.pkl"
        )

        genescores["design_ix"] = design_ix
        scores[design_ix] = genescores.reset_index()
        design.loc[design_ix, "found"] = True
scores = pd.concat(scores, ignore_index = True)
scores = pd.merge(design, scores, on = "design_ix")
# scores = scores.set_index(["dataset", "method", "phase", "gene"])

design["found"].mean()

# %%
method_info = pd.DataFrame(
    [
    ],
    columns=["method", "label", "type"],
).set_index("method")
missing_methods = pd.Index(design["method"].unique()).difference(method_info.index).tolist()
missing_method_info = pd.DataFrame(
    {"method": missing_methods, "label": missing_methods}
).set_index("method")

method_info = pd.concat([method_info, missing_method_info]).reset_index()
method_info = method_info.set_index("method").join(design_peaks.groupby("method")[["peakcaller", "predictor"]].first())

method_info.loc[method_info["peakcaller"].isin(["cellranger", "macs2_improved", "macs2_leiden_0.1_merged", "macs2_leiden_0.1", "genrich"]), "type"] = "peak"
method_info.loc[method_info["peakcaller"].str.startswith("rolling").fillna(False), "type"] = "rolling"
method_info.loc[method_info["peakcaller"].str.startswith("encode").fillna(False), "type"] = "predefined"
method_info.loc[method_info["peakcaller"].isin(["1k1k", "gene_body"]), "type"] = "predefined"
method_info.loc[method_info.index.str.startswith("v2"), "type"] = "ours"
method_info["ix"] = -np.arange(method_info.shape[0])

method_info.loc[method_info["type"] == "baseline", "color"] = "#888888"
method_info.loc[method_info["type"] == "ours", "color"] = "#0074D9"
method_info.loc[method_info["type"] == "rolling", "color"] = "#FF851B"
method_info.loc[method_info["type"] == "peak", "color"] = "#FF4136"
method_info.loc[method_info["type"] == "predefined", "color"] = "#2ECC40"
method_info.loc[pd.isnull(method_info["color"]), "color"] = "#DDDDDD"

# %%
dataset_info = pd.DataFrame(index = design.groupby(["dataset", "promoter", "splitter"]).first().index)
dataset_info["label"] = dataset_info.index.get_level_values("dataset")
dataset_info["ix"] = np.arange(len(dataset_info))
dataset_info["label"] = dataset_info["label"].str.replace("-", "→\n")

# %%
dummy_method = "counter"

# %%
metric_ids = ["cor"]

group_ids = [*method_info.index.names, *dataset_info.index.names, "phase"]

# %%
scores = scores.reset_index().set_index([*group_ids, "gene"])

# %%
scores["cor_diff"] = (scores["cor"] - scores.xs(dummy_method, level="method")["cor"]).reorder_levels(scores.index.names)

# %%
deltacor_cutoff = 0.005

# %%
scores_all = scores.groupby(group_ids)[metric_ids].mean()
meanscores = scores.groupby(group_ids)[["cor", "design_ix"]].mean()

diffscores = meanscores - meanscores.xs(dummy_method, level="method")
diffscores.columns = diffscores.columns + "_diff"

score_relative_all = meanscores.join(diffscores)

score_relative_all["better"] = (
    (
        (scores["cor_diff"] - scores.xs(dummy_method, level="method")["cor_diff"])
        > deltacor_cutoff
    )
    .groupby(group_ids)
    .mean()
)
score_relative_all["worse"] = (
    (
        (scores["cor_diff"] - scores.xs(dummy_method, level="method")["cor_diff"])
        < -deltacor_cutoff
    )
    .groupby(group_ids)
    .mean()
)
score_relative_all["same"] = (
    ((scores["cor_diff"] - scores.xs(dummy_method, level="method")["cor_diff"]).abs() < deltacor_cutoff)
    .groupby(group_ids)
    .mean()
)

# %%
import textwrap

# %%
metric = "cor_diff"
metric_multiplier = 1
metric_limits = (-0.015, 0.015)
metric_label = "$\Delta$ cor"

# %%
panel_width = 4/4

fig, axes = plt.subplots(
    2,
    len(dataset_info),
    figsize=(len(dataset_info) * panel_width, 2 * len(method_info) / 4),
    sharey=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    squeeze=False,
)

for dataset_, plotdata in score_relative_all.groupby(dataset_info.index.names):
    dataset_info_ = dataset_info.loc[dataset_]
    axes_dataset = axes[:, dataset_info_["ix"]].tolist()
    
    ## PANEL 1
    ax = axes_dataset.pop(0)
    # plotdata = score_relative_all.loc[dataset_].reset_index()
    plotdata = plotdata.query("phase in ['validation', 'test']").reset_index()
    plotdata["ratio"] = plotdata["better"] / (plotdata["worse"] + plotdata["better"])
    plotdata.loc[pd.isnull(plotdata["ratio"]), "ratio"] = 0.5

    ax.barh(
        width=plotdata[metric],
        y=method_info.loc[plotdata["method"]]["ix"],
        color=method_info.loc[plotdata["method"]]["color"] + [{"validation":"88", "test":"FF"}[phase] for phase in plotdata["phase"]],
        lw=0,
        zorder = 0,
    )
    
    plotdata_annotate = plotdata.loc[(plotdata[metric] < metric_limits[0]) & (plotdata["phase"] == 'test')]
    transform = mpl.transforms.blended_transform_factory(
        ax.transAxes,
        ax.transData
    )
    for _, plotdata_row in plotdata_annotate.iterrows():
        ax.text(
            x = 0.01,
            y = method_info.loc[plotdata_row["method"]]["ix"],
            s = f"{plotdata_row[metric]:.2f}",
            transform = transform,
            va = "center",
            ha = "left",
            color = "#FFFFFFCC",
            fontsize = 6
        )
    
    ax.axvline(0.0, dashes=(2, 2), color="#888888", zorder=-10, lw = 1)
    ax.set_yticks(method_info["ix"])
    ax.set_yticklabels(method_info["label"])
    ax.xaxis.tick_top()
    ax.set_xlim(*metric_limits)
    
    ## PANEL 2
    ax = axes_dataset.pop(0)
    ax.scatter(
        x=plotdata["ratio"],
        y=method_info.loc[plotdata["method"]]["ix"],
        c=method_info.loc[plotdata["method"]]["color"] + [{"validation":"88", "test":"FF"}[phase] for phase in plotdata["phase"]],
        zorder = 10,
        lw = 0
    )
    ax.axvline(0.5, dashes=(2, 2), color="#888888", zorder=-10, lw = 1)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.5, method_info["ix"].min()-0.5)

for ax, (dataset_, dataset_info_) in zip(axes[0], dataset_info.iterrows()):
    ax.set_title(dataset_info_["label"], fontsize = 8)

for i, ax in enumerate(axes[0]):
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(metric_limits)
    if i == 0:
        ax.xaxis.tick_top()
        ax.set_xlabel(metric_label, fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    else:
        ax.set_xlabel("  \n  ", fontsize=8)
        ax.set_xticks([])
    ax.xaxis.set_label_position("top")

for i, ax in enumerate(axes[1]):
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=8)
    if i == 0:
        ax.set_xlabel("# improved genes\n over # changing genes", fontsize=8)
    else:
        ax.set_xlabel("  \n  ", fontsize=8)
        ax.set_xticks([])
    ax.xaxis.set_label_position("bottom")

for ax in fig.axes:
    ax.legend([], [], frameon=False)

# %%
meanscores.loc["cellranger/linear"].loc["lymphoma-pbmc10k"]

# %% [markdown]
# ## Dependency on # of cells and predictive power

# %%
traindataset_info = pd.DataFrame(index = design["traindataset"].unique())
traindataset_info["n_trained_cells"] = [len(chd.data.Transcriptome(chd.get_output() / "data" / dataset_name / "transcriptome").obs) for dataset_name in traindataset_info.index]
traindataset_info

# %%
score_relative_all["traindataset"] = design["traindataset"][score_relative_all.design_ix].values
score_relative_all["n_trained_cells"] = traindataset_info.loc[score_relative_all["traindataset"], "n_trained_cells"].values

# %%
plotdata = score_relative_all.xs("v20", level = "method").xs("test", level = "phase")

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["n_trained_cells"], plotdata["cor_diff"])
ax.set_xscale("log")

# %%
plotdata.sort_values("n_trained_cells")

# %% [markdown]
# ------

# %%
datasets = pd.DataFrame({"dataset": dataset_names}).set_index("dataset")
datasets["color"] = sns.color_palette("Set1", datasets.shape[0])

# %%
plotdata = pd.DataFrame(
    {
        "cor_total": scores.xs("v14_50freq_sum_sigmoid_initdefault", level="method").xs(
            "validation", level="phase"
        )["cor"],
        "cor_a": scores.xs("v14_50freq_sum_sigmoid_initdefault", level="method").xs(
            "validation", level="phase"
        )["cor_diff"],
        "cor_b": scores.xs("cellranger_linear", level="method").xs(
            "validation", level="phase"
        )["cor_diff"],
        "dataset": scores.xs("cellranger_linear", level="method")
        .xs("validation", level="phase")
        .index.get_level_values("dataset"),
    }
)
plotdata = plotdata.query("cor_total > 0.05")
plotdata["diff"] = plotdata["cor_a"] - plotdata["cor_b"]
plotdata = plotdata.sample(n=plotdata.shape[0])

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.axline((0, 0), slope=1, dashes=(2, 2), zorder=1, color="#333")
ax.axvline(0, dashes=(1, 1), zorder=1, color="#333")
ax.axhline(0, dashes=(1, 1), zorder=1, color="#333")
plt.scatter(
    plotdata["cor_b"],
    plotdata["cor_a"],
    c=datasets.loc[plotdata["dataset"], "color"],
    alpha=0.5,
    s=1,
)
ax.set_xlabel("$\Delta$ cor Cellranger linear")
ax.set_ylabel("$\Delta$ cor Positional NN", rotation=0, ha="right", va="center")

# %%
plotdata.sort_values("diff", ascending=False).head(20)

# %%

# %%

# %%

# %%

# %%
