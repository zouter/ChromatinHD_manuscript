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
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

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
from chromatinhd_manuscript.designs import (
    dataset_splitter_peakcaller_predictor_combinations as design_peaks,
)

# design_peaks = design_peaks.loc[design_peaks["predictor"] != "xgboost"].copy()

from chromatinhd_manuscript.designs import (
    dataset_splitter_method_combinations as design_methods,
)

from chromatinhd_manuscript.designs import (
    traindataset_testdataset_splitter_method_combinations as design_methods_traintest,
)

design_methods_traintest["dataset"] = design_methods_traintest["testdataset"]
design_methods_traintest["splitter"] = "all"

from chromatinhd_manuscript.designs import (
    traindataset_testdataset_splitter_peakcaller_predictor_combinations as design_peaks_traintest,
)

design_peaks_traintest["dataset"] = design_peaks_traintest["testdataset"]
design_peaks_traintest["splitter"] = "all"

# %%
design_peaks["method"] = design_peaks["peakcaller"] + "/" + design_peaks["predictor"]
design_peaks_traintest["method"] = (
    design_peaks_traintest["peakcaller"] + "/" + design_peaks_traintest["predictor"]
)

# %%
design = pd.concat(
    [design_peaks, design_methods, design_methods_traintest, design_peaks_traintest]
)
design.index = np.arange(len(design))
design.index.name = "design_ix"

# %%
design = design.query("dataset != 'alzheimer'").copy()
# design = design.query("dataset == 'pbmc10k_gran-pbmc10k'").copy()
# design = design.query("dataset == 'pbmc10k_gran-pbmc10k'").copy()

# %%
design = design.query("splitter in ['random_5fold', 'all']").copy()
design = design.query("promoter in ['10k10k']").copy()
design = design.query("method != 'v20_initdefault'").copy()

# %%
design["traindataset"] = [
    x["dataset"] if pd.isnull(x["traindataset"]) else x["traindataset"]
    for _, x in design.iterrows()
]


# %%
class Prediction(chd.flow.Flow):
    pass


# %%
scores = {}
design["found"] = False
for design_ix, design_row in design.iterrows():
    prediction = chd.flow.Flow(
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
scores = pd.concat(scores, ignore_index=True)
scores = pd.merge(design, scores, on="design_ix")

scores = scores.reset_index().set_index(
    ["method", "dataset", "promoter", "phase", "gene"]
)

dummy_method = "counter"
scores["cor_diff"] = (
    scores["cor"] - scores.xs(dummy_method, level="method")["cor"]
).reorder_levels(scores.index.names)

design["found"].mean()

# %%
metric_ids = ["cor"]

group_ids = ["method", "dataset", "promoter", "phase"]

meanscores = scores.groupby(group_ids)[["cor", "design_ix"]].mean()
diffscores = meanscores - meanscores.xs(dummy_method, level="method")
diffscores.columns = diffscores.columns + "_diff"

scores_all = meanscores.join(diffscores)

# %%
methods_info = chdm.methods.prediction_methods.reindex(design["method"].unique())

methods_info["type"] = pd.Categorical(
    methods_info["type"], ["peak", "predefined", "rolling", "ours"]
)
methods_info["predictor"] = pd.Categorical(
    methods_info["predictor"], ["linear", "lasso", "xgboost"]
)
methods_info["subgroup"] = methods_info["type"] != "ours"
methods_info = methods_info.sort_values(["subgroup", "predictor", "type"])

methods_info["ix"] = -np.arange(methods_info.shape[0])

methods_info.loc[pd.isnull(methods_info["color"]), "color"] = "black"
methods_info.loc[pd.isnull(methods_info["label"]), "label"] = methods_info.index[
    pd.isnull(methods_info["label"])
]

# %%
metrics_info = pd.DataFrame(
    [
        {
            "label": "$\\Delta$ cor\n(method-baseline)",
            "metric": "cor_diff",
            "limits": (-0.01, 0.01),
            "transform": lambda x: x,
            "ticks": [-0.01, 0, 0.01],
            "ticklabels": ["-0.01", "0", "0.01"],
        },
    ]
).set_index("metric")
metrics_info["ix"] = np.arange(len(metrics_info))
metrics_info["ticks"] = metrics_info["ticks"].fillna(
    metrics_info.apply(
        lambda metric_info: [metric_info["limits"][0], 0, metric_info["limits"][1]],
        axis=1,
    )
)
metrics_info["ticklabels"] = metrics_info["ticklabels"].fillna(
    metrics_info.apply(lambda metric_info: metric_info["ticks"], axis=1)
)

# %%
datasets_info = pd.DataFrame(
    index=design.groupby(["dataset", "promoter"]).first().index
)
datasets_info["label"] = datasets_info.index.get_level_values("dataset")
datasets_info = datasets_info.sort_values("label")
datasets_info["ix"] = np.arange(len(datasets_info))
datasets_info["label"] = datasets_info["label"].str.replace("-", "→\n")

# %% [markdown]
# ### Across datasets and metrics

# %%

score_relative_all = scores_all

# %%
panel_width = 5 / 4
panel_resolution = 1 / 4

fig, axes = plt.subplots(
    len(metrics_info),
    len(datasets_info),
    figsize=(
        len(datasets_info) * panel_width,
        len(metrics_info) * len(methods_info) * panel_resolution,
    ),
    gridspec_kw={"wspace": 0.05, "hspace": 0.2},
    squeeze=False,
)

for dataset, dataset_info in datasets_info.iterrows():
    axes_dataset = axes[:, dataset_info["ix"]].tolist()
    for metric, metric_info in metrics_info.iterrows():
        ax = axes_dataset.pop(0)
        ax.set_xlim(metric_info["limits"])
        plotdata = (
            pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [dataset], names=datasets_info.index.names
                )
            )
            .join(score_relative_all)
            .reset_index()
            .query("phase == 'test'")
        )
        plotdata = pd.merge(
            plotdata,
            methods_info,
            on="method",
        )

        ax.barh(
            width=plotdata[metric],
            y=plotdata["ix"],
            color=plotdata["color"],
            lw=0,
            zorder=0,
            # height=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # out of limits values
        metric_limits = metric_info["limits"]
        plotdata_annotate = plotdata.loc[
            (plotdata[metric] < metric_limits[0])
            | (plotdata[metric] > metric_limits[1])
        ]
        transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        metric_transform = metric_info.get("transform", lambda x: x)
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x=0.03 if left else 0.97,
                y=plotdata_row["ix"],
                s=f"{metric_transform(plotdata_row[metric]):+.2f}",
                transform=transform,
                va="center",
                ha="left" if left else "right",
                color="#FFFFFFCC",
                fontsize=6,
            )

    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Datasets
for dataset, dataset_info in datasets_info.iterrows():
    ax = axes[0, dataset_info["ix"]]
    ax.set_title(dataset_info["label"], fontsize=8)

# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[metric_info["ix"], 0]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])

    ax = axes[metric_info["ix"], 1]
    ax.set_xlabel(metric_info["label"])

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"])

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

manuscript.save_figure(fig, "2", "positional_all_scores_datasets")

# %% [markdown]
# ### Averaged per dataset group

# %%
datagroups_info = pd.DataFrame(
    {
        "datagroup": ["within_dataset", "across_dataset", "across_celltypes"],
        "ix": [0, 1, 2],
        "label": [
            "Within dataset",
            "Across datasets\nSame cell types",
            "Across datasets\nDifferent cell types",
        ],
        "datasets": [
            ["brain", "e18brain", "lymphoma", "pbmc10k", "pbmc10k_gran"],
            ["pbmc10k_gran-pbmc10k", "pbmc3k-pbmc10k"],
            ["lymphoma-pbmc10k"],
        ],
    }
).set_index("datagroup")
datasets_info["datagroup"] = (
    datagroups_info.explode("datasets")
    .reset_index()
    .set_index("datasets")
    .loc[datasets_info.index.get_level_values("dataset"), "datagroup"]
    .values
)

# %%
group_ids = [*methods_info.index.names, "datagroup", "phase"]
scores_all["datagroup"] = (
    datasets_info["datagroup"]
    .reindex(scores_all.reset_index()[datasets_info.index.names])
    .values
)
score_relative_all = scores_all.groupby(group_ids).mean()

# %%
panel_width = 5 / 4
panel_resolution = 1 / 8

fig, axes = plt.subplots(
    len(metrics_info),
    len(datagroups_info),
    figsize=(
        len(datagroups_info) * panel_width,
        len(metrics_info) * len(methods_info) * panel_resolution,
    ),
    gridspec_kw={"wspace": 0.2, "hspace": 0.2},
    squeeze=False,
)

for datagroup, datagroup_info in datagroups_info.iterrows():
    axes_datagroup = axes[:, datagroup_info["ix"]].tolist()
    for metric, metric_info in metrics_info.iterrows():
        ax = axes_datagroup.pop(0)
        ax.set_xlim(metric_info["limits"])
        plotdata = (
            score_relative_all.xs(datagroup, level="datagroup")
            .reset_index()
            .query("phase == 'test'")
        )
        plotdata = pd.merge(
            plotdata,
            methods_info,
            on="method",
        )

        ax.barh(
            width=plotdata[metric],
            y=plotdata["ix"],
            color=plotdata["color"],
            lw=0,
            zorder=0,
            # height=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # out of limits values
        metric_limits = metric_info["limits"]
        plotdata_annotate = plotdata.loc[
            (plotdata[metric] < metric_limits[0])
            | (plotdata[metric] > metric_limits[1])
        ]
        transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        metric_transform = metric_info.get("transform", lambda x: x)
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x=0.03 if left else 0.97,
                y=plotdata_row["ix"],
                s=f"{metric_transform(plotdata_row[metric]):+.2f}",
                transform=transform,
                va="center",
                ha="left" if left else "right",
                color="#FFFFFFCC",
                fontsize=6,
            )

        # individual values
        plotdata = scores_all.loc[
            scores_all.index.get_level_values("dataset").isin(
                datagroup_info["datasets"]
            )
        ].query("phase == 'test'")
        plotdata = pd.merge(
            plotdata,
            methods_info,
            on="method",
        )
        ax.scatter(
            plotdata[metric],
            plotdata["ix"],
            # color=plotdata["color"],
            s=2,
            zorder=1,
            marker="|",
            color="#33333388",
        )

    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Datasets
for datagroup, datagroup_info in datagroups_info.iterrows():
    ax = axes[0, datagroup_info["ix"]]
    ax.set_title(datagroup_info["label"], fontsize=8)

# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[metric_info["ix"], 0]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])

    ax = axes[metric_info["ix"], 1]
    ax.set_xlabel(metric_info["label"])

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"], fontsize=8)

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

manuscript.save_figure(fig, "2", "positional_all_scores_datagroups")


# %% [markdown]
# ### Averaged over all datasets
group_ids = [*methods_info.index.names, "phase"]
score_relative_all = scores_all.groupby(group_ids).mean()

# %%
panel_width = 5 / 4
panel_resolution = 1 / 8

fig, axes = plt.subplots(
    1,
    len(metrics_info),
    figsize=(
        len(metrics_info) * panel_width,
        len(methods_info) * panel_resolution,
    ),
    gridspec_kw={"wspace": 0.2},
    squeeze=False,
)

for metric_ix, (metric, metric_info) in enumerate(metrics_info.iterrows()):
    ax = axes[0, metric_ix]
    ax.set_xlim(metric_info["limits"])
    plotdata = score_relative_all.reset_index().query("phase == 'test'")
    plotdata = pd.merge(
        plotdata,
        methods_info,
        on="method",
    )

    ax.barh(
        width=plotdata[metric],
        y=plotdata["ix"],
        color=plotdata["color"],
        lw=0,
        zorder=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # out of limits values
    metric_limits = metric_info["limits"]
    plotdata_annotate = plotdata.loc[
        (plotdata[metric] < metric_limits[0]) | (plotdata[metric] > metric_limits[1])
    ]
    transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    metric_transform = metric_info.get("transform", lambda x: x)
    for _, plotdata_row in plotdata_annotate.iterrows():
        left = plotdata_row[metric] < metric_limits[0]
        ax.text(
            x=0.03 if left else 0.97,
            y=plotdata_row["ix"],
            s=f"{metric_transform(plotdata_row[metric]):+.2f}",
            transform=transform,
            va="center",
            ha="left" if left else "right",
            color="#FFFFFFCC",
            fontsize=6,
        )


# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[0, metric_info["ix"]]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])
    ax.set_xlabel(metric_info["label"])
    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"], fontsize=8)

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

manuscript.save_figure(fig, "2", "positional_all_scores")

# %% [markdown]
# ## Dependency on # of cells and predictive power

# %%
traindataset_info = pd.DataFrame(index=design["traindataset"].unique())
traindataset_info["n_trained_cells"] = [
    len(
        chd.data.Transcriptome(
            chd.get_output() / "data" / dataset_name / "transcriptome"
        ).obs
    )
    for dataset_name in traindataset_info.index
]
traindataset_info

# %%
score_relative_all["traindataset"] = design["traindataset"][
    score_relative_all.design_ix
].values
score_relative_all["n_trained_cells"] = traindataset_info.loc[
    score_relative_all["traindataset"], "n_trained_cells"
].values

# %%
plotdata = score_relative_all.xs("v20", level="method").xs("test", level="phase")

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["n_trained_cells"], plotdata["cor_diff"])
ax.set_xscale("log")

# %%
plotdata.sort_values("n_trained_cells")

# %% [markdown]
# ## Compare against CRE methods


# %%
method_oi1 = "v20"
method_oi2 = "cellranger/linear"
# method_oi2 = "rolling_500/xgboost"
plotdata = pd.DataFrame(
    {
        "cor_b": scores.xs(method_oi2, level="method").xs("validation", level="phase")[
            "cor_diff"
        ],
        "dataset": scores.xs(method_oi2, level="method")
        .xs("validation", level="phase")
        .index.get_level_values("dataset"),
    }
)
plotdata["cor_total"] = scores.xs(method_oi1, level="method").xs(
    "validation", level="phase"
)["cor"]
plotdata["cor_a"] = scores.xs(method_oi1, level="method").xs(
    "validation", level="phase"
)["cor_diff"]
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
    # c=datasets_info.loc[plotdata["dataset"], "color"],
    alpha=0.5,
    s=1,
)
ax.set_xlabel(f"$\Delta$ cor {method_oi2}")
ax.set_ylabel("$\Delta$ cor ChromatinHD", rotation=0, ha="right", va="center")

# %%
plotdata.loc["pbmc10k"].sort_values("diff", ascending=False).head(20)

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / "pbmc10k"
    / "10k10k"
    / "random_5fold"
    / "v20"
)

scores_dir = prediction.path / "scoring"


# %%
# if you're interested in genes where the difference is small
plotdata.loc["pbmc10k"].assign(abs_diff=lambda x: np.abs(x["diff"]).abs()).query(
    "cor_a > 0.1"
).sort_values("abs_diff", ascending=True).to_csv(
    scores_dir / "difference_with_peakcalling_small.csv"
)

plotdata.loc["pbmc10k"].assign(abs_diff=lambda x: np.abs(x["diff"]).abs()).query(
    "cor_a > 0.1"
).sort_values("abs_diff", ascending=True).head(20)

# %%
# if you're interested in genes where the difference is large
plotdata.loc["pbmc10k"].query("cor_a > 0.1").sort_values(
    "diff", ascending=False
).to_csv(scores_dir / "difference_with_peakcalling_large.csv")

plotdata.loc["pbmc10k"].query("cor_a > 0.1").sort_values(
    "diff", ascending=False
).head(20)

# %%

# %%
