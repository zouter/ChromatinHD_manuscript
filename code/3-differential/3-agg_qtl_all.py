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
#     language: pythondatasets_info
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

import pickle

import scanpy as sc
import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

import itertools

# %% [markdown]
# ### Method info

# %%
from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_method_qtl_enricher_combinations as design,
)

design = design.query("dataset in ['pbmc10k', 'brain']").copy()
design = design.query("motifscan in ['gtex_immune', 'gwas_immune2']")
design = design.query("promoter in ['10k10k']")


# %%
def get_score_folder(x):
    return (
        chd.get_output()
        / "prediction_likelihood"
        / x.dataset
        / x.promoter
        / x.latent
        / x.method
        / "scoring"
        / x.peakcaller
        / x.diffexp
        / x.motifscan
        / x.enricher
    )


design["score_folder"] = design.apply(get_score_folder, axis=1)


# %%
def aggregate_motifscores(motifscores):
    motifscores_oi = motifscores
    ratio_significant = (
        motifscores_oi.query("n_region > 0")["pval_region"] < 0.05
    ).sum() / (motifscores_oi.query("n_peak > 0")["pval_peak"] < 0.05).sum()
    ratio_captured = (motifscores_oi.query("n_region > 0")["in_region"]).mean() / (
        motifscores_oi.query("n_region > 0")["in_peak"]
    ).mean()
    ratio_up = (
        (motifscores_oi.query("n_region > 0")["in_region"])
        > (motifscores_oi.query("n_region > 0")["in_peak"])
    ).mean() / (
        (motifscores_oi.query("n_region > 0")["in_region"])
        < (motifscores_oi.query("n_region > 0")["in_peak"])
    ).mean()
    ratio_detected = (motifscores_oi.query("n_region > 0")["in_region"] > 0).sum() / (
        motifscores_oi.query("n_region > 0")["in_peak"] > 0
    ).sum()

    return {
        "ratio_significant": ratio_significant,
        "ratio_captured": ratio_captured,
        "ratio_up": ratio_up,
        "ratio_detected": ratio_detected,
    }


# %%
design["force"] = True

# %%
for design_ix, subdesign in tqdm.tqdm(design.iterrows(), total=len(design)):
    design_row = subdesign

    desired_outputs = [(subdesign["score_folder"] / "aggscores.pkl")]
    force = subdesign["force"]
    if not all([desired_output.exists() for desired_output in desired_outputs]):
        force = True

    if force:
        # load motifscores
        try:
            score_folder = design_row["score_folder"]
            scores_peaks = pd.read_pickle(score_folder / "scores_peaks.pkl")
            scores_regions = pd.read_pickle(score_folder / "scores_regions.pkl")

            # scores[ix] = scores_peaks
            motifscores = pd.merge(
                scores_peaks,
                scores_regions,
                on=scores_peaks.index.names,
                suffixes=("_peak", "_region"),
                how="outer",
            )
        except BaseException as e:
            print(e)
            continue

        scores = aggregate_motifscores(motifscores)
        pickle.dump(scores, (subdesign["score_folder"] / "aggscores.pkl").open("wb"))
# scores = pd.DataFrame(scores)

# %%
motifscores_oi = motifscores.query("n_region > 0")
# ((motifscores_oi["in_region"])-(motifscores_oi["in_peak"])).groupby("").mean()

# %%
((motifscores_oi["in_region"]) - (motifscores_oi["in_peak"])).groupby(
    "gene"
).min().mean()

# %%
((motifscores_oi["in_region"]) - (motifscores_oi["in_peak"])).groupby(
    "gene"
).max().mean()

# %%
(
    (motifscores_oi.query("n_region > 0")["in_region"])
    == (motifscores_oi.query("n_region > 0")["in_peak"])
).mean()


# %% [markdown]
# ## Summarize

# %%
scores = []
for design_ix, design_row in tqdm.tqdm(design.iterrows(), total=len(design)):
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
# %%
methods_info = chdm.methods.peakcaller_diffexp_combinations

methods_info["label"] = np.where(
    ~pd.isnull(methods_info.reset_index()["peakcaller"]),
    (
        chdm.peakcallers.reindex(methods_info.reset_index()["peakcaller"])[
            "label"
        ].reset_index(drop=True)
        + " ("
        + chdm.diffexps.reindex(methods_info.reset_index()["diffexp"])[
            "label_short"
        ].reset_index(drop=True)
        + ")"
    ).values,
    "ChromatinHD differential",
)

# methods_info = methods_info.sort_values(["diffexp", "type", "peakcaller"])

# methods_info["ix"] = -np.arange(methods_info.shape[0])

# %%
datasets_info = pd.DataFrame(
    index=design.groupby(["dataset", "promoter", "latent", "motifscan"]).first().index
)
datasets_info["label"] = (
    datasets_info.index.get_level_values("dataset")
    + "-\n"
    + datasets_info.index.get_level_values("motifscan")
)
datasets_info["ix"] = np.arange(len(datasets_info))

# %%
group_ids = [*methods_info.index.names, *datasets_info.index.names]

# %%
scores["logratio_significant"] = np.log(scores["ratio_significant"]).fillna(0)
scores["logratio_captured"] = np.log(scores["ratio_captured"]).fillna(0)
scores["logratio_up"] = np.log(scores["ratio_up"]).fillna(0)
scores["logratio_detected"] = np.log(scores["ratio_detected"]).fillna(0)

scores_joined = scores.set_index("design_ix").join(design)

meanscores = scores.set_index("design_ix").join(design)

meanscores = meanscores.set_index(group_ids)
scores_all = meanscores.copy()

# %%
import textwrap

# %%
metrics_info = pd.DataFrame(
    [
        # {
        #     "label": r"ratio captured",
        #     "metric": "logratio_captured",
        #     "limits": (np.log(1 / 2), np.log(2)),
        #     "ticks": [np.log(1 / 2), 0, np.log(2)],
        #     "ticklabels": ["½", "1", "2"],
        # },
        {
            "label": "ratio genes\nwith QTLs\n(ChromatinHD\n/method)",
            "metric": "logratio_up",
            "limits": (np.log(1 / 2), np.log(2)),
            "ticks": [np.log(1 / 2), 0, np.log(2)],
            "ticklabels": ["½", "1", "2"],
        },
        # {
        #     "label": r"ratio detected",
        #     "metric": "logratio_detected",
        #     "limits": (np.log(1 / 2), np.log(2)),
        #     "ticks": [np.log(1 / 2), 0, np.log(2)],
        #     "ticklabels": ["½", "1", "2"],
        # },
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
panel_width = 4 / 4
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
            .join(meanscores)
            .reset_index()
        )
        plotdata = pd.merge(plotdata, methods_info, on=methods_info.index.names)

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
            (plotdata[metric] < metric_limits[0])
            | (plotdata[metric] > metric_limits[1])
        ]
        transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x=0.03 if left else 0.97,
                y=plotdata_row["ix"],
                s=f"{plotdata_row[metric]:+.2f}",
                transform=transform,
                va="center",
                ha="left" if left else "right",
                color="#FFFFFFCC",
                fontsize=6,
            )

# Datasets
for dataset, dataset_info in datasets_info.iterrows():
    ax = axes[0, dataset_info["ix"]]
    ax.set_title(dataset_info["label"], fontsize=8)

# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[metric_info["ix"], 0]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])

    ax = axes[metric_info["ix"], 0]
    ax.set_xlabel(metric_info["label"])

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"])

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

manuscript.save_figure(fig, "2", "likelihood_all_qtl_datasets")

# %%
np.exp(
    meanscores.groupby(["dataset", "motifscan", "diffexp", "peakcaller"]).mean()[
        metrics_info.index
    ]
).style.bar()

# %%

# %% [markdown]
# ### Averaged over all datasets

# %%
group_ids = [*methods_info.index.names]

ncell_cutoff = 0
average_n_positions_cutoff = 10**4
meanscores = scores_joined.groupby(group_ids)[
    [
        "logratio_up",
        "logratio_detected",
        "logratio_significant",
        "logratio_captured",
    ]
].mean()

# %%
panel_width = 4 / 4
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
    plotdata = meanscores.reset_index()
    plotdata = pd.merge(
        plotdata,
        methods_info,
        on=methods_info.index.names,
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

    # individual values
    plotdata = scores_all
    plotdata = pd.merge(
        plotdata,
        methods_info,
        on=[*methods_info.index.names],
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

# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[0, metric_info["ix"]]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])
    ax.set_xlabel(metric_info["label"])
    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Methods
ax = axes[0, 0]
ax.set_yticks(methods_info["ix"])
ax.set_yticklabels(methods_info["label"], fontsize=8)

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

ax.set_yticks([])
manuscript.save_figure(fig, "2", "likelihood_all_qtl")

# %%