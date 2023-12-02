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
# dataset_name = "pbmc10k";promoter_name = "10k10k";latent_name = "leiden_0.1";method_name = "v9_128-64-32";motifscan_name = "cutoff_0001"
# scores_dir = (
#     chd.get_output()
#     / "prediction_likelihood"
#     / dataset_name
#     / promoter_name
#     / latent_name
#     / method_name / "scoring" / "significant_up" / motifscan_name
# )
# scores = pickle.load((scores_dir / "scores.pkl").open("rb"))

# %%
import chromatinhd as chd

# %%
import itertools

# %% [markdown]
# ### Method info

# %%
promoter_name = "10k10k"
design = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["lymphoma"],
            ["celltype"],
            ["v9_128-64-32"],
            [
                "xgboost_100",
                # "lasso",
                # "lm",
                "svr",
            ],
        ),
        itertools.product(
            [
                "lymphoma"
            ],
            ["celltype"],
            [
                "baseline"
            ],
            [""],
        ),
        itertools.product(
            [
                "pbmc10k",
                "e18brain",
                "brain",
            ],
            ["leiden_0.1"],
            [
                "v9_128-64-32",
                "stack",
                "cellranger",
                "macs2_improved",
                "macs2_leiden_0.1",
                "macs2_leiden_0.1_merged",
                "genrich",
                "rolling_500",
                "rolling_100",
                "rolling_50",
            ],
            [
                "xgboost_100",
                # "lasso",
                # "lm",
                "svr",
            ],
        ),
        itertools.product(
            [
                "pbmc10k",
                "e18brain",
                "brain",
            ],
            ["leiden_0.1"],
            [
                "baseline"
            ],
            [""],
        ),
    ),
    columns=["dataset", "latent", "method", "predictor"],
)

# %%
method_info = pd.DataFrame(
    [
        ["v9_128-64-32", "ChromatinHD", "ours"],
        ["cellranger", "Cellranger", "peak"],
        ["macs2_improved", "MACS2", "peak"],
        ["macs2_leiden_0.1", "MACS2 cluster", "peak"],
        ["macs2_leiden_0.1_merged", "MACS2 cluster merged", "peak"],
        ["genrich", "Genrich", "peak"],
        ["stack", "-10kb → +10kb", "rolling"],
        ["rolling_500", "Window 500bp", "rolling"],
        ["rolling_100", "Window 100bp", "rolling"],
        ["rolling_50", "Window 50bp", "rolling"],
    ],
    columns=["method", "label", "type"],
).set_index("method")
missing_methods = pd.Index(design["method"].unique()).difference(method_info.index).tolist()
missing_method_info = pd.DataFrame(
    {"method": missing_methods, "label": missing_methods}
).set_index("method")
method_info = pd.concat([method_info, missing_method_info])
method_info["ix"] = -np.arange(method_info.shape[0])

method_info.loc[method_info["type"] == "baseline", "color"] = "grey"
method_info.loc[method_info["type"] == "ours", "color"] = "#0074D9"
method_info.loc[method_info["type"] == "rolling", "color"] = "#FF851B"
method_info.loc[method_info["type"] == "peak", "color"] = "#FF4136"
method_info.loc[pd.isnull(method_info["color"]), "color"] = "black"

method_info["opacity"] = "88"
method_info.loc[method_info.index[-1], "opacity"] = "FF"

# %%
method_info["baseline"] = [
    method_name + "_baseline" if not method_name.endswith("_baseline") else method_name
    for method_name in method_info.index
]


# %% [markdown]
# ### Load scores

# %%
class Prediction(chd.flow.Flow):
    pass


# %%
scores = {}
for _, (dataset_name, latent_name, method_name, predictor) in design.iterrows():
    scores_dir = (
        chd.get_output()
        / "prediction_expression_pseudobulk"
        / dataset_name
        / promoter_name
        / latent_name
        / method_name
        / predictor
    )
    if (scores_dir / "scores.pkl").exists():
        scores_ = pd.read_pickle(
            scores_dir / "scores.pkl"
        )

        scores[(dataset_name, latent_name, method_name, predictor)] = scores_
    else:
        print(scores_dir)
scores = pd.concat(scores, names=["dataset", "latent", "method", "predictor", *scores_.index.names])

# %% [markdown]
# ### Process scores

# %%
scores["mse"] = scores["rmse"]**2

# %%
baseline_scores = scores.xs("baseline", level = "method").xs("", level = "predictor")

# %%
# fill all missing values, typically because some genes were not predicted because they had no peaks
fill_values = baseline_scores
desired_index = baseline_scores.index
scores = (
    scores
    .groupby(["method", "predictor"])
    .apply(lambda x:x.reset_index().set_index(desired_index.names).reindex(desired_index).fillna(fill_values).drop(columns = ["method", "predictor"]))
)

# %%
# scores["rmse_diff"] = (scores["rmse"] + -baseline_scores["rmse"]).reorder_levels(scores.index.names).loc[scores.index]
# scores["mse_diff"] = (scores["mse"] + -baseline_scores["mse"]).reorder_levels(scores.index.names).loc[scores.index]

# %%
meanscores = scores.groupby(["dataset", "latent", "method", "predictor"]).mean()
baseline_meanscores = meanscores.xs("baseline", level = "method").xs("", level = "predictor")
meanscores["rmse_diff"] = (meanscores["rmse"] + -baseline_meanscores["rmse"]).reorder_levels(meanscores.index.names).loc[meanscores.index]
meanscores["mse_diff"] = (meanscores["mse"] + -baseline_meanscores["mse"]).reorder_levels(meanscores.index.names).loc[meanscores.index]

# %%
# scores["mse_baseline"] = baseline_scores["mse"].loc[scores.index.droplevel(["method", "predictor"])].values
# scores["mse_fraction"] = scores["mse"] / scores["mse_baseline"]

# %%
meanscores.sort_values(["dataset", "predictor"]).style.bar()

# %%
plotdata = scores.query("gene_ix < 500").groupby(["dataset", "latent", "method", "predictor"]).mean()

# %%
dataset_latent_info = plotdata.index.to_frame(index = False).groupby(["dataset", "latent"]).first().index.to_frame(index = False)
dataset_latent_info["label"] = dataset_latent_info["dataset"]
dataset_latent_info = dataset_latent_info.set_index(["dataset", "latent"])

# %%
predictor_metric_info = pd.DataFrame([
    ["xgboost_100", "rmse_diff"],
    ["svr", "rmse_diff"]
], columns = ["predictor", "metric"]
).set_index(["predictor", "metric"])
predictor_metric_info["label"] = predictor_metric_info.index.get_level_values("predictor")

# %% [markdown]
# ### Plot

# %%
panel_width = 6/6
fig, axes = plt.subplots(
    2,
    len(dataset_latent_info),
    figsize=(len(dataset_latent_info) * panel_width, 2 * len(method_info) / 4),
    sharey=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    squeeze=False,
)

for axes_dataset, dataset_latent_name in zip(axes.T, dataset_latent_info.index):
    axes_dataset = axes_dataset.tolist()
    
    for ax, (predictor, metric) in zip(axes_dataset, predictor_metric_info.index):
        plotdata = meanscores.loc[dataset_latent_name].xs(predictor, level = "predictor").reset_index()
        ax.barh(
            method_info.loc[plotdata["method"], "ix"],
            plotdata[metric],
            color = method_info.loc[plotdata["method"]]["color"]
        )
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_yticks(method_info["ix"])
        ax.set_yticklabels(method_info["label"])
        ax.xaxis.tick_top()
        
        ax.set_xlim(plotdata[metric].min() * (-0.1), plotdata[metric].min() * 1.05)
    
for ax, dataset_name in zip(axes[0], dataset_latent_info.label):
    ax.set_title("-\n".join(dataset_name.split("-")))

for ax in axes[0]:
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("  \n  ", fontsize=8)
    ax.set_xticks([])
    ax.xaxis.set_label_position("top")
for ax in axes[1]:
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("  \n  ", fontsize=8)
    ax.set_xticks([])
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
for ax, ((predictor, metric), predictor_metric) in zip(axes[:, 0], predictor_metric_info.iterrows()):
    ax.set_xlabel(predictor_metric["label"], fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    
    
#     plotdata["ratio"] = plotdata["better"] / (plotdata["worse"] + plotdata["better"])
#     plotdata.loc[pd.isnull(plotdata["ratio"]), "ratio"] = 0.5

#     ax.barh(
#         width=plotdata[metric],
#         y=method_info.loc[plotdata["method"]]["ix"],
#         color=method_info.loc[plotdata["method"]]["color"],
#         # color = "#FF4136" + method_info.loc[plotdata["method"]]["opacity"],
#         lw=0,
#     )
#     ax.axvline(0.0, dashes=(2, 2), color="#333333", zorder=-10)
#     ax.set_yticks(method_info["ix"])
#     ax.set_yticklabels(method_info["label"])
#     ax.xaxis.tick_top()
#     ax.set_xlim(*metric_limits)
#     ax = axes_dataset.pop(0)
#     ax.scatter(
#         x=plotdata["ratio"],
#         y=method_info.loc[plotdata["method"]]["ix"],
#         c=method_info.loc[plotdata["method"]]["color"].values
#     )
#     ax.axvline(0.5, dashes=(2, 2), color="#333333", zorder=-10)
#     ax.set_xlim(0, 1)
#     ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# for ax, dataset_name in zip(axes[0], dataset_names):
#     ax.set_title("-\n".join(dataset_name.split("-")))

# for i, ax in enumerate(axes[0]):
#     ax.tick_params(axis="y", length=0)
#     ax.tick_params(axis="x", labelsize=8)
#     if i == 0:
#         ax.set_xlabel(metric_label, fontsize=8)
#         ax.xaxis.set_major_locator(plt.MaxNLocator(3))
#     else:
#         ax.set_xlabel("  \n  ", fontsize=8)
#         ax.set_xticks([])
#     ax.xaxis.set_label_position("top")

# for i, ax in enumerate(axes[1]):
#     ax.tick_params(length=0)
#     ax.tick_params(axis="x", labelsize=8)
#     if i == 0:
#         ax.set_xlabel("# improved genes\n over # changing genes", fontsize=8)
#     else:
#         ax.set_xlabel("  \n  ", fontsize=8)
#         ax.set_xticks([])
#     ax.xaxis.set_label_position("bottom")

# for ax in fig.axes:
#     ax.legend([], [], frameon=False)
# ax_all.set_xlim(
# ax_cor_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_cor_all.get_yticklabels()]]["label"])

# %%
sns.ecdfplot(np.log(scores.loc["baseline", "", "pbmc10k", "leiden_0.1"]["mse"]))
sns.ecdfplot(np.log(scores.loc["v9_128-64-32", "xgboost_100", "pbmc10k", "leiden_0.1"]["mse"]))
sns.ecdfplot(np.log(scores.loc["cellranger", "xgboost_100", "pbmc10k", "leiden_0.1"]["mse"]))

# %%
from scipy.stats.mstats import hmean

# %%
scores.groupby(["dataset", "latent", "method", "predictor"]).mean().style.bar()

# %%
scores["rmse_diff_oi"] = (scores["rmse"].xs("v9_128-64-32", level = "method") - scores["rmse"]).reorder_levels(scores.index.names).loc[scores.index]
scores["improved"] = (scores["rmse_diff_oi"] < 0)

# %%
scores.query("gene_ix < 500").groupby("method")["improved"].mean()

# %%
top_counts = scores["rmse"].unstack(level = ["dataset", "latent", "gene_ix", "validation_cluster"]).idxmin().value_counts()
top_counts/top_counts.sum()

# %%
baselines = pd.MultiIndex.from_frame(
    scores.index.to_frame().assign(
        method=lambda x: method_info["baseline"][
            x.index.get_level_values("method").tolist()
        ].tolist()
    )
)
try:
    scores = scores.drop(columns=["lr"])
except:
    pass
scores["lr"] = scores["likelihood"].values - scores["likelihood"].loc[baselines].values

# %%
cutoff = 0.005
metric = "likelihood_diff"
metric_multiplier = 1
metric_limits = (-100, 100)
metric_label = "LR"
# metric = "likelihood_diff";metric_multiplier = 1;metric_limits = (-1000, 1000); metric_label = "LR"
# metric = "cor";metric_multiplier = 1;metric_limits = (0, 0.1); metric_label = "cor"

# %%
scores_all = scores.groupby(["dataset", "method", "phase"])[["likelihood"]].mean()

meanscores = scores.groupby(["dataset", "method", "phase"])[["likelihood"]].mean()

baselines = pd.MultiIndex.from_frame(
    meanscores.index.to_frame().assign(
        method=lambda x: method_info["baseline"][
            x.index.get_level_values("method").tolist()
        ].tolist()
    )
)

diffscores = meanscores - meanscores.loc[baselines].values
diffscores.columns = diffscores.columns + "_diff"
diffscores["likelihood_diff"] = diffscores["likelihood_diff"]

score_relative_all = meanscores.join(diffscores)

# score_relative_all["better"] = ((scores - scores.xs(dummy_method, level = "method"))[metric] * metric_multiplier > cutoff).groupby(["dataset", "method", "phase"]).mean()
# score_relative_all["worse"] = ((scores - scores.xs(dummy_method, level = "method"))[metric] * metric_multiplier < -cutoff).groupby(["dataset", "method", "phase"]).mean()
# score_relative_all["same"] = ((scores - scores.xs(dummy_method, level = "method"))[metric].abs() < cutoff).groupby(["dataset", "method", "phase"]).mean()

# mean_scores = scores.query("gene in @genes_oi").groupby(["dataset", "method", "phase"])[["cor"]].mean()
# scores_relative_able = (mean_scores - mean_scores.xs(dummy_method, level = "method"))

# scores_relative_able["better"] = ((scores - scores.xs(dummy_method, level = "method")).query("gene in @genes_oi")["cor"] > cutoff).groupby(["dataset", "phase", "method"]).mean()
# scores_relative_able["worse"] = ((scores - scores.xs(dummy_method, level = "method")).query("gene in @genes_oi")["cor"] < -cutoff).groupby(["dataset", "phase", "method"]).mean()
# scores_relative_able["same"] = ((scores - scores.xs(dummy_method, level = "method").query("gene in @genes_oi"))["cor"].abs() < cutoff).groupby(["dataset", "phase", "method"]).mean()

# %%
import textwrap

# %%
method_info["is_baseline"] = method_info.index.str.endswith("baseline")

# %%
fig, axes = plt.subplots(
    2,
    len(dataset_names),
    figsize=(len(dataset_names) * 6 / 4, 2 * len(method_names) / 4),
    sharey=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    squeeze=False,
)

for axes_dataset, dataset_name in zip(axes.T, dataset_names):
    axes_dataset = axes_dataset.tolist()
    ax = axes_dataset.pop(0)
    plotdata = score_relative_all.loc[dataset_name].reset_index()
    # plotdata = plotdata.query("phase in ['train', 'test']")
    plotdata = plotdata.query("phase in ['validation', 'test']")

    ax.barh(
        width=plotdata[metric],
        y=method_info.loc[plotdata["method"]]["ix"],
        color=method_info.loc[plotdata["method"]]["color"],
        # color = "#FF4136" + method_info.loc[plotdata["method"]]["opacity"],
        lw=0,
    )
    ax.axvline(0.0, dashes=(2, 2), color="#333333", zorder=-10)
    ax.set_yticks(method_info["ix"])
    ax.set_yticklabels(method_info["label"])
    ax.xaxis.tick_top()
    ax.set_xlim(*metric_limits)

for ax, dataset_name in zip(axes[0], dataset_names):
    ax.set_title("-\n".join(dataset_name.split("-")))

for i, ax in enumerate(axes[0]):
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    if i == 0:
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
# ax_all.set_xlim(
# ax_cor_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_cor_all.get_yticklabels()]]["label"])

# %%
dataset_name = "pbmc10k"
# dataset_name = "e18brain"

# %%
folder_data_preproc = folder_data / dataset_name
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
sc.tl.leiden(transcriptome.adata)
sc.tl.rank_genes_groups(transcriptome.adata, "leiden")
# lfc = sc.get.rank_genes_groups_df(transcriptome.adata, None).query("pvals < 0.05").sort_values("logfoldchanges", ascending = False).groupby("names").first()["logfoldchanges"]
lfc = (
    sc.get.rank_genes_groups_df(transcriptome.adata, None)
    .sort_values("pvals", ascending=False)
    .groupby("names")
    .first()["pvals"]
)

# %%
for method in method_names:
    if method.endswith("baseline"):
        continue
    s = pd.DataFrame(
        {"lr": scores.loc[dataset_name].loc[method].loc["validation"]["lr"]}
    )
    s["lfc"] = lfc
    s = s.dropna()
    cor = np.corrcoef(s["lfc"], s["lr"])[0, 1]
    print(method, cor)

# %%
sns.scatterplot(x="lr", y="lfc", data=s)

# %%
transcriptome.adata.obs["leiden"]

# %%
# these are genes that are horribly modeled by the mixture
# typically because most fragments are centered e.g. on the promoter
# and then some small number of fragments far away are not well modeled
# but then the differential mixture model "improves" this by slightly moving one component to the gene body
# so it "seems" that the likelihood is very much improved
# but the model is just horrible
# => small "improvements" are extremely magnified
(
    scores.loc["lymphoma"].loc["v4"].loc["train"]
    - scores.loc["lymphoma"].loc["v2"].loc["train"]
).sort_values("lr")

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
