# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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

dataset_names = [
    "pbmc10k",
    "lymphoma",
    "e18brain",
    # "pbmc3k-pbmc10k",
    # "lymphoma-pbmc10k",
    # "pbmc10k_gran-pbmc10k"
]
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_names[0]
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (-10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (-10000, 0)

# %% tags=[]
method_names = [
    # "v4",
    # "v4_baseline",
    # "v4_decoder1",
    "v5",
    "v5_baseline",
    "v5_8",
    "v5_32",
    "v5_s0.8",
    # "v5_1decoder",
    # "v5_norescale",
    # "v5_encoder32",
    # "v5_regularizefragmentcounts",
    # "v5_regularizefragmentcounts_400epoch",
    # "v5_s0.5",
    # "v5_s0.3",
    # "v5_mixtureautoscale",
    # "v5_mixturescale0.1",
    # "v5_mixturelaplace",
    # "v6",
    # "cellranger/pca_50",
    # "macs2/pca_50",
    # "stack/pca_50",
    "cellranger/pca_20",
    "macs2/pca_20",
    "stack/pca_20",
    # "cellranger/pca_200",
    # "macs2/pca_200",
    # "stack/pca_200",
    # "cellranger/pca_5",
    # "macs2/pca_5",
    # "stack/pca_5",
]


# %%
class Prediction(chd.flow.Flow):
    pass


# %%
scores = {}
for dataset_name in dataset_names:
    for method_name in method_names:
        prediction = Prediction(
            chd.get_output()
            / "prediction_vae"
            / dataset_name
            / promoter_name
            / method_name
        )
        if (prediction.path / "scoring" / "overall" / "scores.pkl").exists():
            scores_ = pd.read_pickle(
                prediction.path / "scoring" / "overall" / "scores.pkl"
            )

            scores[(dataset_name, method_name)] = scores_
        else:
            print(prediction.path)
scores = pd.concat(scores, names=["dataset", "method", *scores_.index.names])

# %%
# scores = scores.query("phase != 'all'")

# %%
method_info = pd.DataFrame(
    [
        ["counter", "Counter (baseline)", "baseline"],
        ["counter_binary", "Counter binary", "baseline"],
        ["v9", "Ours", "ours"],
        ["v11", "Ours", "ours"],
        ["v14_50freq_sum_sigmoid_initdefault", "Positional NN", "ours"],
        ["macs2_linear", "Linear", "peak"],
        ["macs2_polynomial", "Quadratic", "peak"],
        ["macs2_xgboost", "XGBoost", "peak"],
        ["cellranger_linear", "Linear", "peak"],
        ["cellranger_polynomial", "Quadratic", "peak"],
        ["cellranger_xgboost", "XGBoost", "peak"],
        ["rolling_500_linear", "Linear", "rolling"],
        ["rolling_500_xgboost", "XGBoost", "rolling"],
    ],
    columns=["method", "label", "type"],
).set_index("method")
missing_methods = pd.Index(method_names).difference(method_info.index).tolist()
missing_method_info = pd.DataFrame(
    {"method": missing_methods, "label": missing_methods}
).set_index("method")
method_info = pd.concat([method_info, missing_method_info])
method_info = method_info.loc[method_names]
method_info["ix"] = -np.arange(method_info.shape[0])

method_info.loc[method_info["type"] == "baseline", "color"] = "grey"
method_info.loc[method_info["type"] == "ours", "color"] = "#0074D9"
method_info.loc[method_info["type"] == "rolling", "color"] = "#FF851B"
method_info.loc[method_info["type"] == "peak", "color"] = "#FF4136"
method_info.loc[pd.isnull(method_info["color"]), "color"] = "black"

method_info["opacity"] = "88"
method_info.loc[method_info.index[-1], "opacity"] = "FF"

# %%
method_info["baseline"] = "v5_baseline"

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
# scores["lr"] = (scores["likelihood"].values - scores["likelihood"].loc[baselines].values)

# %%
metric = {"name": "woverlap_diff", "limits": (-0.1, 0.1), "label": "woverlap_diff"}
metric2 = {"name": "overlap_diff", "limits": (-0.1, 0.1), "label": "woverlap_diff"}
metric3 = {"name": "ari_diff", "limits": (-0.1, 0.1), "label": "woverlap_diff"}
metric4 = {"name": "ami_diff", "limits": (-0.1, 0.1), "label": "ami_diff"}
metrics = [metric, metric2, metric3, metric4]

# %% tags=[]
scores_all = scores.groupby(["dataset", "method", "phase"])[
    ["overlap", "woverlap", "ari", "ami"]
].mean()

meanscores = scores.groupby(["dataset", "method", "phase"])[
    ["overlap", "woverlap", "ari", "ami"]
].mean()

baselines = pd.MultiIndex.from_frame(
    meanscores.index.to_frame().assign(
        method=lambda x: method_info["baseline"][
            x.index.get_level_values("method").tolist()
        ].tolist()
    )
)

diffscores = meanscores - meanscores.loc[baselines].values
diffscores.columns = diffscores.columns + "_diff"
diffscores["overlap_diff"] = diffscores["overlap_diff"]

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
plotdata = meanscores.unstack(level="dataset").unstack("phase")
plotdata = plotdata - plotdata.loc["v5_baseline"]
plotdata.style.bar(cmap=mpl.cm.RdBu_r, align="zero", vmin=-0.02, vmax=0.02)

# %%
import textwrap

# %%
method_info["is_baseline"] = method_info.index.str.endswith("baseline")

# %%
phase_info = pd.DataFrame(
    {"phase": ["train", "validation", "all"], "marker": [".", ".", "o"]}
).set_index("phase")

# %% tags=[]
fig, axes = plt.subplots(
    len(metrics),
    len(dataset_names),
    figsize=(len(dataset_names) * len(metrics) / 2, 2 * len(method_names) / 2),
    sharey=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.2},
    squeeze=False,
)

for axes_dataset, dataset_name in zip(axes.T, dataset_names):
    axes_dataset = axes_dataset.tolist()

    for metric, ax in zip(metrics, axes_dataset):
        plotdata = score_relative_all.loc[dataset_name].reset_index()
        for phase in ["train", "validation", "all"]:
            plotdata_ = plotdata.loc[plotdata["phase"] == phase]
            ax.scatter(
                x=plotdata_[metric["name"]],
                y=method_info.loc[plotdata_["method"]]["ix"],
                color=method_info.loc[plotdata_["method"]]["color"],
                marker=phase_info.loc[phase]["marker"],
            )

        ax.axvline(0.0, dashes=(2, 2), color="#333333", zorder=-10)
        ax.set_yticks(method_info["ix"])
        ax.set_yticklabels(method_info["label"])
        ax.xaxis.tick_top()
        ax.set_xlim(*metric["limits"])

for ax, dataset_name in zip(axes[0], dataset_names):
    ax.set_title("-\n".join(dataset_name.split("-")))

for metric, ax in zip(metrics, axes[:, 0]):
    ax.set_xlabel(metric["name"])

# for i, ax in enumerate(axes[0]):
#     ax.tick_params(axis = "y", length  = 0)
#     ax.tick_params(axis = "x", labelsize=8)
#     if i == 0:
#         ax.set_xlabel(metric_label, fontsize = 8)
#         ax.xaxis.set_major_locator(plt.MaxNLocator(3))
#     else:
#         ax.set_xlabel("  \n  ", fontsize = 8)
#         ax.set_xticks([])
#     ax.xaxis.set_label_position('top')

# for i, ax in enumerate(axes[1]):
#     ax.tick_params(length  = 0)
#     ax.tick_params(axis = "x", labelsize=8)
#     if i == 0:
#         ax.set_xlabel("# improved genes\n over # changing genes", fontsize = 8)
#     else:
#         ax.set_xlabel("  \n  ", fontsize = 8)
#         ax.set_xticks([])
#     ax.xaxis.set_label_position('bottom')

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
