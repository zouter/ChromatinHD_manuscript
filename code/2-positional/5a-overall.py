# ---
# jupyter:
#   jupytext:
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
import IPython

if IPython.get_ipython():
    IPython.get_ipython().magic("load_ext autoreload")
    IPython.get_ipython().magic("autoreload 2")

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

import torch_scatter
import torch

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
dataset_name = "pbmc10k"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
outcome_source = "magic"
prediction_name = "v20"

folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# create design to run
from design import get_design, get_folds_inference


class Prediction(chd.flow.Flow):
    pass


# %%
# prediction_name = "counter"
prediction_name = "v20"
# prediction_name = "v20_initdefault"

# %%
baseline_prediction_name = "counter"
# baseline_prediction_name = "cellranger/lasso"
# baseline_prediction_name = "rolling_100/linear"

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

scores_dir = prediction.path / "scoring" / "overall"

scores = pd.read_pickle(scores_dir / "scores.pkl")
genescores = pd.read_pickle(scores_dir / "genescores.pkl")

baseline_prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / baseline_prediction_name
)
baseline_scores_dir = baseline_prediction.path / "scoring" / "overall"
scores_baseline = pd.read_pickle(baseline_scores_dir / "scores.pkl")
genescores_baseline = pd.read_pickle(baseline_scores_dir / "genescores.pkl")

# %%
diffscores = (scores - scores_baseline).rename(columns=lambda x: x + "_diff")
# diffscores["mse_diff"] *= -1
scores[diffscores.columns] = diffscores

diffgenescores = (genescores - genescores_baseline).rename(
    columns=lambda x: x + "_diff"
)
# diffgenescores["mse_diff"] *= -1
genescores[diffgenescores.columns] = diffgenescores

# %%
genescores.to_pickle(scores_dir / "genescores.pkl")

# %%
genescores_baseline.loc["test"]["cor"].mean(), genescores.loc["test"]["cor"].mean()

# %% [markdown]
# ### Global view

# %%
phases = chd.plotting.phases

# %%
cor_cutoff = 0.01

# %%
(genescores.sort_values("cor_diff")["cor_diff"] > 0).mean()

# %%
sumscores = {
    "same": (
        (genescores["cor_diff"] > -cor_cutoff) & (genescores["cor_diff"] < cor_cutoff)
    ).mean(),
    "higher": ((genescores["cor_diff"] > cor_cutoff)).mean(),
    "lower": ((genescores["cor_diff"] < -cor_cutoff)).mean(),
}

# %%
fig, ax = plt.subplots(figsize=(3, 2))
sns.ecdfplot(genescores.sort_values("cor_diff")["cor_diff"].loc["validation"])
mean = genescores["cor_diff"].loc["validation"].mean()
ax.axvline(mean, color="red")
ax.set_xlim(-0.1, 0.1)
ticks = ax.set_xticks([-0.1, 0, 0.1])
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax.xaxis.tick_top()
ax.axvline(0, color="#333333", dashes=(2, 2))
ax.annotate(
    f"{mean:.2f}",
    (mean, -0.05),
    xycoords="data",
    ha="center",
    va="top",
    clip_on=False,
    color="red",
    annotation_clip=False,
)
ax.set_xlabel("Δcor")
ax.xaxis.set_label_position("top")

# %%
fig, ax = plt.subplots(figsize=(3, 2))
sns.ecdfplot(genescores.sort_values("cor_diff")["cor_diff"].loc["validation"])
ax.axvspan(-0.1, -cor_cutoff, color="#333333", alpha=0.1)
ax.axvspan(+0.1, cor_cutoff, color="green", alpha=0.1)
mean = genescores["cor_diff"].loc["validation"].mean()
ax.axvline(cor_cutoff, color="#333333", dashes=(2, 2))
ax.axvline(-cor_cutoff, color="#333333", dashes=(2, 2))
ax.set_xlim(-0.1, 0.1)
ticks = ax.set_xticks([-0.1, cor_cutoff, 0.1])
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax.xaxis.tick_top()
ax.annotate(
    f"{sumscores['lower']:.0%} genes\n worse prediction",
    (0.25, 0.95),
    xycoords="axes fraction",
    ha="center",
    va="top",
    bbox=bbox_props,
    color="#333333",
)
# ax.annotate(f"{sumscores['same']:.0%} genes\n same prediction", (0.5, -0.45), xycoords = "axes fraction", ha = "center", va = "top", bbox = bbox_props, color = "grey")
ax.annotate(
    f"{sumscores['higher']:.0%} genes\n better prediction",
    (0.75, 0.05),
    xycoords="axes fraction",
    ha="center",
    va="bottom",
    bbox=bbox_props,
    color="green",
)
ax.set_xlabel("Δcor")
ax.xaxis.set_label_position("top")

# %%
genescores["cor_diff_improved"] = genescores["cor_diff"] > cor_cutoff

# %%
transcriptome.adata.var["n_fragments"] = torch.bincount(
    fragments.mapping[:, 1], minlength=fragments.n_genes
).numpy()

# %%
plotdata = pd.DataFrame(
    {
        "log_dispersions_norm": np.log(transcriptome.adata.var["dispersions_norm"]),
        "log_mean": np.log(transcriptome.adata.var["means"]),
        "log_n_fragments": np.log1p(transcriptome.adata.var["n_fragments"]),
        "cor_diff": genescores.loc["validation"]["cor_diff"],
    }
)

# %%
measures = pd.DataFrame(
    {
        "measure": ["log_dispersions_norm", "log_mean", "log_n_fragments"],
        "label": ["normalized dispersion", "mean expression", "# fragments"],
        "ix": [0, 1, 2],
        "ticks": [
            [np.log(0.1), np.log(1), np.log(10)],
            [np.log(0.01), np.log(0.1), np.log(1)],
            [np.log1p(0), np.log1p(150), np.log1p(20000)],
        ],
        "ticktransformer": [np.exp, np.exp, lambda x: np.exp(x) - 1],
    }
).set_index("measure")

# %%
import statsmodels.api as sm

# %%
fig, axes = plt.subplots(1, len(measures), figsize=(2 * len(measures), 2), sharey=True)

for (measure, measure_info), ax in zip(measures.iterrows(), axes):
    plotdata_oi = plotdata.copy()
    plotdata_oi["measure"] = plotdata_oi[measure]
    ax.scatter(plotdata[measure], plotdata["cor_diff"], s=1, alpha=0.5)

    # fit loess
    xvals = np.linspace(plotdata_oi["measure"].min(), plotdata_oi["measure"].max(), 100)
    loess = sm.nonparametric.lowess(
        plotdata_oi["cor_diff"], plotdata_oi["measure"], xvals=xvals, frac=0.1
    )
    loess = pd.DataFrame(np.stack([xvals, loess]).T, columns=["x", "y"])
    ax.set_xlabel(measure_info["label"])
    ax.plot(loess["x"], loess["y"], color="red")
    ax.set_xticks(measure_info["ticks"])
    # format as float but with no trailing zeros

    ax.set_xticklabels(
        [
            f"{('%.2f' % measure_info['ticktransformer'](x)).rstrip('0').rstrip('.')}"
            for x in measure_info["ticks"]
        ]
    )
axes[0].set_ylabel("Δ cor")

fig.savefig(chd.get_output() / "prediction_positional" / "accuracy_vs_gene_metrics.pdf")

# %%
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
manuscript.save_figure(fig, "2", "accuracy_vs_gene_metrics")

# %%
plt.scatter(genescores_baseline.loc["test"]["cor"], genescores.loc["test"]["cor_diff"])

# %%
genescores.loc["test"].loc[transcriptome.gene_id("IL1B")]

# %%
genescores["cor_diff"]

# %%
