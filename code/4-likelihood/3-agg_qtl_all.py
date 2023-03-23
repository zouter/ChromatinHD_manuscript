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
import itertools

# %% [markdown]
# ### Method info

# %%
from chromatinhd_manuscript.designs import dataset_latent_peakcaller_diffexp_method_qtl_enricher_combinations as design
design = design.query("dataset in ['pbmc10k', 'brain']").copy()

# %%
promoter_name = "10k10k"


# %%
def get_score_folder(x):
    return chd.get_output() / "prediction_likelihood" / x.dataset / promoter_name / x.latent / x.method / "scoring" / x.peakcaller / x.diffexp / x.motifscan / x.enricher
design["score_folder"] = design.apply(get_score_folder, axis = 1)


# %%
def aggregate_motifscores(motifscores):
    motifscores_oi = motifscores
    ratio_significant = (motifscores_oi.query("n_region > 0")["pval_region"] < 0.05).sum() / (motifscores_oi.query("n_peak > 0")["pval_peak"] < 0.05).sum()
    ratio_captured = (motifscores_oi.query("n_region > 0")["in_region"]).mean() / (motifscores_oi.query("n_region > 0")["in_peak"]).mean()
    
    return {
        "ratio_significant":ratio_significant,
        "ratio_captured":ratio_captured,
    }


# %%
design["force"] = False

# %%
for design_ix, subdesign in tqdm.tqdm(design.iterrows(), total = len(design)):    
    design_row = subdesign
    
    desired_outputs = [(subdesign["score_folder"] / "aggscores.pkl")]
    force = subdesign["force"]
    if not all([desired_output.exists() for desired_output in desired_outputs]):
        force = True

    if force:
        # load motifscores
        try:
            score_folder = design_row["score_folder"]
            scores_peaks = pd.read_pickle(
                score_folder / "scores_peaks.pkl"
            )
            scores_regions = pd.read_pickle(
                score_folder / "scores_regions.pkl"
            )

            # scores[ix] = scores_peaks
            motifscores = pd.merge(scores_peaks, scores_regions, on = scores_peaks.index.names, suffixes = ("_peak", "_region"), how = "outer")
        except BaseException as e:
            print(e)
            continue

        scores = aggregate_motifscores(motifscores)
        pickle.dump(scores, (subdesign["score_folder"] / "aggscores.pkl").open("wb"))
# scores = pd.DataFrame(scores)

# %% [markdown]
# ## Summarize

# %%
scores = []
for design_ix, design_row in tqdm.tqdm(design.iterrows(), total = len(design)):    
    try:
        score = pickle.load((design_row["score_folder"] / "aggscores.pkl").open("rb"))
        score["design_ix"] = design_ix
        scores.append(score)
    except FileNotFoundError as e:
        pass
scores = pd.DataFrame(scores)

# %% [markdown]
# ## Plot all combinations

# %%
methods_info = design.groupby(["peakcaller", "diffexp"]).first().index.to_frame(index = False)

methods_info["type"] = "predefined"
methods_info.loc[methods_info["peakcaller"].isin(["cellranger", "macs2_improved", "macs2_leiden_0.1_merged", "macs2_leiden_0.1", "genrich"]), "type"] = "peak"
methods_info.loc[methods_info["peakcaller"].str.startswith("rolling").fillna(False), "type"] = "rolling"

methods_info.loc[methods_info["type"] == "baseline", "color"] = "#888888"
methods_info.loc[methods_info["type"] == "ours", "color"] = "#0074D9"
methods_info.loc[methods_info["type"] == "rolling", "color"] = "#FF851B"
methods_info.loc[methods_info["type"] == "peak", "color"] = "#FF4136"
methods_info.loc[methods_info["type"] == "predefined", "color"] = "#2ECC40"
methods_info.loc[pd.isnull(methods_info["color"]), "color"] = "#DDDDDD"

methods_info["type"] = pd.Categorical(methods_info["type"], ["peak", "predefined", "rolling"])

methods_info = methods_info.set_index(["peakcaller", "diffexp"])
methods_info["label"] = methods_info.index.get_level_values("peakcaller")

methods_info = methods_info.sort_values(["diffexp", "type", "peakcaller"])
                                        
methods_info["ix"] = -np.arange(methods_info.shape[0])

# %%
datasets_info = pd.DataFrame(index = design.groupby(["dataset", "promoter", "latent", "motifscan"]).first().index)
datasets_info["label"] = datasets_info.index.get_level_values("dataset") + "-\n" + datasets_info.index.get_level_values("motifscan")
datasets_info["ix"] = np.arange(len(datasets_info))

# %%
group_ids = [*methods_info.index.names, *datasets_info.index.names]

# %%
scores["logratio_significant"] = np.log(scores["ratio_significant"]).fillna(0)
scores["logratio_captured"] = np.log(scores["ratio_captured"]).fillna(0)

meanscores = scores.set_index("design_ix").join(design)

meanscores = meanscores.set_index(group_ids)

# %%
import textwrap

# %%
metrics_info = pd.DataFrame([
    {"label":r"ratio significant", "metric":"logratio_significant", "limits":(np.log(1/2), np.log(2)), "ticks":[np.log(1/2), 0, np.log(2)], "ticklabels":["½", "1", "2"]},
    {"label":r"ratio captured", "metric":"logratio_captured", "limits":(np.log(1/2), np.log(2)), "ticks":[np.log(1/2), 0, np.log(2)], "ticklabels":["½", "1", "2"]},
]).set_index("metric")
metrics_info["ix"] = np.arange(len(metrics_info))

metrics_info["ticks"] = metrics_info["ticks"].fillna(metrics_info.apply(lambda metric_info: [metric_info["limits"][0], 0, metric_info["limits"][1]], axis = 1))
metrics_info["ticklabels"] = metrics_info["ticklabels"].fillna(metrics_info.apply(lambda metric_info: metric_info["ticks"], axis = 1))

# %%
panel_width = 4/4
panel_resolution = 1/4

fig, axes = plt.subplots(
    len(metrics_info),
    len(datasets_info),
    figsize=(len(datasets_info) * panel_width, len(metrics_info) * len(methods_info) * panel_resolution),
    gridspec_kw={"wspace": 0.05, "hspace": 0.2},
    squeeze=False,
)

for dataset, dataset_info in datasets_info.iterrows():
    axes_dataset = axes[:, dataset_info["ix"]].tolist()
    for metric, metric_info in metrics_info.iterrows():
        ax = axes_dataset.pop(0)
        ax.set_xlim(metric_info["limits"])
        plotdata = pd.DataFrame(index = pd.MultiIndex.from_tuples([dataset], names = datasets_info.index.names)).join(meanscores).reset_index()
        plotdata = pd.merge(plotdata, methods_info, on = methods_info.index.names)
        
        ax.barh(
            width=plotdata[metric],
            y=plotdata["ix"],
            color=plotdata["color"],
            lw=0,
            zorder = 0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        
        # out of limits values
        metric_limits = metric_info["limits"]
        plotdata_annotate = plotdata.loc[(plotdata[metric] < metric_limits[0]) | (plotdata[metric] > metric_limits[1])]
        transform = mpl.transforms.blended_transform_factory(
            ax.transAxes,
            ax.transData
        )
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x = 0.03 if left else 0.97,
                y = plotdata_row["ix"],
                s = f"{plotdata_row[metric]:+.2f}",
                transform = transform,
                va = "center",
                ha = "left" if left else "right",
                color = "#FFFFFFCC",
                fontsize = 6
            )

# Datasets
for dataset, dataset_info in datasets_info.iterrows():
    ax = axes[0, dataset_info["ix"]]
    ax.set_title(dataset_info["label"], fontsize = 8)
    
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
    ax.set_ylim(methods_info["ix"].min()-0.5, 0.5)

# %%
np.exp(0.14)

# %%
meanscores.groupby("peakcaller").mean()[metrics_info.index].mean()

# %%

# %%
