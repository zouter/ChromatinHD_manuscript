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
import polyptich as pp
pp.setup_ipython()

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
peakcallers = chdm.peakcallers.copy().loc[["macs2_summits", "macs2_leiden_0.1_merged", "macs2_improved", "encode_screen", "rolling_500"]]
peakcallers.loc["chd"] = {"label":"ChromatinHD", "color":"#0074D9"}

peakcallers_peaks = peakcallers.loc[peakcallers["type"] == "peak"].copy()
peakcallers_peaks["marker"] = ["o", "p", "D"]
peakcallers_peaks["color"] = ["#FF7E7B", "#FF4136", "#B20909"]
peakcallers.loc[peakcallers_peaks.index, "color"] = peakcallers_peaks["color"]
peakcallers["marker"] = peakcallers_peaks["marker"]
peakcallers.loc["encode_screen", "marker"] = "s"
peakcallers.loc["rolling_500", "marker"] = "*"
peakcallers.loc["chd", "marker"] = "v"

# %% [markdown]
# ## Load

# %%
scores_gwas = pd.read_csv(chd.get_output() / "aggregate_gwas_enrichment.csv", index_col = 0)
scores_gwas = scores_gwas.sort_values("odds", ascending = False).groupby("peakcaller").first().reset_index()
scores_gwas["score"] = scores_gwas["odds"]-1

# %%
scores_eqtl = pd.read_csv(chd.get_output() / "aggregate_eqtl_enrichment.csv", index_col = 0)
scores_eqtl = scores_eqtl.sort_values("odds", ascending = False).groupby("peakcaller").first().reset_index()
scores_eqtl["score"] = scores_eqtl["odds"]-1

# %%
scores_crispri = pd.read_csv(chd.get_output() / "aggregate_crispr.csv", index_col = 0)
scores_crispri = scores_crispri.sort_values("cor", ascending = False).groupby("peakcaller").first().reset_index()
scores_crispri["score"] = scores_crispri["cor"]

# %%
scores_prediction = pd.read_csv(chd.get_output() / "aggregate_prediction.csv", index_col = 0)
scores_prediction = scores_prediction.sort_values("cor", ascending = False).groupby("peakcaller").first().reset_index()
scores_prediction["score"] = scores_prediction["cor"]

# %%
scores_motif = pd.read_csv(chd.get_output() / "aggregate_motif_enrichment.csv", index_col = 0)
scores_motif = scores_motif.sort_values("cor", ascending = False).groupby("peakcaller").first().reset_index()
scores_motif["score"] = scores_motif["log_avg_odds"]

# %%
scores_joined = pd.concat(
    [
        scores_eqtl[["peakcaller", "score"]].assign(task="eqtl"),
        scores_gwas[["peakcaller", "score"]].assign(task="gwas"),
        # scores_qtl[["peakcaller", "score"]].assign(task="qtl"),
        scores_crispri[["peakcaller", "score"]].assign(task="crispri"),
        scores_prediction[["peakcaller", "score"]].assign(task="prediction"),
        scores_motif[["peakcaller", "score"]].assign(task="motif"),
    ],
    axis=0,
)

# %%
scores = scores_joined.set_index(["peakcaller", "task"])["score"].unstack()
ranks = scores.rank(ascending=False, axis=0)

# %%
# tasks = pd.DataFrame({
#     "task": ["mean", "qtl", "crispri", "prediction", "motif"],
#     "label": ["All", "QTL", "CRISPRi", "Prediction", "Motif"],
#     "ix": [4, 3, 1, 0, 2],
# })
tasks = pd.DataFrame({
    "task": ["mean", "eqtl", "gwas", "crispri", "prediction", "motif"],
    "label": ["All", "eQTL", "GWAS", "CRISPRi", "Pred.", "TFBS"],
    "ix": [5, 3, 4,1, 0, 2],
})
tasks = tasks.set_index("task").sort_values("ix")

# %%
normscores = scores / scores.max()
normscores["mean"] = normscores.mean(axis=1)
normscores = normscores.reindex(index = peakcallers.index, columns = tasks.index)

# %%
fig, ax = plt.subplots(figsize = (4., 2))

for task_id, task in tasks.iterrows():
    points = []
    for peakcaller, score in normscores[task_id].items():
        color = peakcallers.loc[peakcaller, "color"]
        marker = "o"
        if not pd.isnull(peakcallers.loc[peakcaller, "marker"]):
            marker = peakcallers.loc[peakcaller, "marker"]
        point = ax.scatter(task["ix"], score, marker=marker, color=color, s = 50)
        points.append(point)

for (task_id, task), (task_id2, task2) in zip(tasks.iloc[:-1].iterrows(), tasks.iloc[1:].iterrows()):
    for (peakcaller, score), (_, score2) in zip(normscores[task_id].items(), normscores[task_id2].items()):
        ax.plot([task["ix"], task2["ix"]], [score, score2], color=peakcallers.loc[peakcaller, "color"], zorder = -1)

ax.set_xticks(tasks["ix"])
ax.set_xticklabels(tasks["label"])
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
sns.despine(ax=ax, left=False, bottom=True)
ax.xaxis.set_tick_params(length = 0.)

ax.set_ylim(0.22, 1.05)
ax.set_yticks([1., 0.75, 0.5, 0.25])
ax.set_yticklabels(["Max.", "75%", "50%", "25%"])

manuscript.save_figure(fig, "4", "aggregate_joined")

# %%
# sns.heatmap(ranks, cmap = "viridis_r", annot = True, fmt = ".0f", cbar = False)

# %% [markdown]
# ## Across datasets

# %%
datasets_oi = ["pbmc10k", "pbmc10k_gran", "lymphoma", "hspc", "liver", "e18brain", "alzheimer"]
datasets = pd.DataFrame({
    "dataset": datasets_oi,
    # "label": ["PBMC", "E18 Brain", "Lymphoma", "HSPC", "Liver"],
    "ix": range(len(datasets_oi)),
}).set_index("dataset")
datasets["label"] = datasets.index

# %%
scores_motif = pd.read_csv(chd.get_output() / "aggregate_motif_enrichment_datasets.csv", index_col = 0)
scores_motif = scores_motif.sort_values("log_avg_odds", ascending = False).groupby(["dataset", "peakcaller"]).first().reset_index()
scores_motif = scores_motif.set_index(["peakcaller", "dataset"])["log_avg_odds"].unstack()
scores_motif = scores_motif.reindex(index = peakcallers.index, columns = datasets_oi)
scores_motif = scores_motif / scores_motif.max()

# %%
scores_qtl = pd.read_csv(chd.get_output() / "aggregate_qtl_enrichment_datasets.csv", index_col = 0)
scores_qtl = scores_qtl.sort_values("odds", ascending = False).groupby(["dataset", "peakcaller"]).first().reset_index()
scores_qtl["score"] = scores_qtl["odds"]-1
scores_qtl = scores_qtl.set_index(["peakcaller", "dataset"])["odds"].unstack()
scores_qtl = scores_qtl.reindex(index = peakcallers.index, columns = datasets_oi)
scores_qtl = (scores_qtl / scores_qtl.max())

# %%
scores_prediction = pd.read_csv(chd.get_output() / "aggregate_prediction_datasets.csv", index_col = 0)
scores_prediction = scores_prediction.sort_values("cor", ascending = False).groupby(["dataset", "peakcaller"]).first().reset_index()
scores_prediction["score"] = scores_prediction["cor"]
scores_prediction = scores_prediction.set_index(["peakcaller", "dataset"])["score"].unstack()
scores_prediction = scores_prediction.reindex(index = peakcallers.index, columns = datasets_oi)
scores_prediction = scores_prediction / scores_prediction.max()

# %%
overall_dataset_scores = pd.DataFrame(np.nanmean(np.stack([scores_motif.values, scores_qtl.values, scores_prediction.values], axis = -1), axis = -1, ), index = peakcallers.index, columns = datasets_oi)

# %%
fig, ax = plt.subplots(figsize = (4., 2))

for dataset_id, dataset in datasets.iterrows():
    points = []
    for peakcaller, score in overall_dataset_scores[dataset_id].items():
        marker = "o"
        if not pd.isnull(peakcallers.loc[peakcaller, "marker"]):
            marker = peakcallers.loc[peakcaller, "marker"]
        point = ax.scatter(dataset["ix"], score, marker=marker, color=peakcallers.loc[peakcaller, "color"], s = 50)
        points.append(point)

for (dataset_id, dataset), (dataset_id2, dataset2) in zip(datasets.iloc[:-1].iterrows(), datasets.iloc[1:].iterrows()):
    for (peakcaller, score), (_, score2) in zip(overall_dataset_scores[dataset_id].items(), overall_dataset_scores[dataset_id2].items()):
        ax.plot([dataset["ix"], dataset2["ix"]], [score, score2], color=peakcallers.loc[peakcaller, "color"], zorder = -1)

datasets["even"] = (datasets["ix"] % 2) == 0
ax.set_xticks(datasets.query("even")["ix"])
ax.set_xticklabels(datasets.query("even")["label"])
ax.xaxis.set_tick_params(length = 0.)

ax.set_xticks(datasets.query("~even")["ix"], minor = True)
ax.set_xticklabels(datasets.query("~even")["label"], minor = True)
ax.xaxis.set_tick_params(length = 15., which = "minor")

ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
sns.despine(ax=ax, left=False, bottom=True)

ax.set_ylim(0.2, 1.05)
ax.set_yticks([1., 0.75, 0.5])
ax.set_yticklabels(["Max", "75%", "50%"])

manuscript.save_figure(fig, "4", "aggregate_datasets")

# %%
