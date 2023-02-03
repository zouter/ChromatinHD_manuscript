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

designs = []

# %%
import itertools

# %%
design_1 = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["pbmc10k", "e18brain"],
            ["leiden_0.1"],
            ["v4_128-64-32_30_rep"],
            [
                "cellranger",
                "macs2",
                "rolling_500",
                # None
            ],
            [
                "cutoff_0001",
                # "gwas_immune",
                # "onek1k_0.2"
            ],
        ),
        itertools.product(
            ["lymphoma"],
            ["celltype"],
            ["v4_128-64-32_30_rep"],
            [
                "cellranger",
                "macs2",
                "rolling_500",
                # None
            ],
            [
                "cutoff_0001",
                # "gwas_lymphoma"
            ],
        ),
    ),
    columns=["dataset", "latent", "method", "peaks", "motifscan"],
)
designs.append(design_1)

# %%
dataset_design = pd.Series(
    [
        "CDX1_7",
        "CDX2_7",
        "MSGN1_7",
        "KLF4_7",
        "KLF5_7",
        "FLI1_7",
        "NHLH1_7",
    ],
    name="dataset",
)

promoter_name, (padding_negative, padding_positive) = "10k10k", (-10000, 10000)

method_design = pd.Series(
    [
        # "v4_128-64-32",
        # "v4_128-64-32_rep",
        # "v4_128",
        # "v4_64",
        # "v4_32",
        # "v4_64-32",
        # "v4_256-128-64-32",
        "v4_128-64-32_30",
        # "v4_128-64-32_30_laplace0.05",
        # "v4_128-64-32_30_laplace0.1",
        # "v4_128-64-32_30_laplace1.0",
        # "v4_128-64-32_30_normal0.05",
    ],
    name="method",
)

peaks_design = pd.Series(
    [
        "cellranger",
        "macs2",
        "rolling_500",
    ],
    name="peaks",
)

motifscan_design = pd.Series(
    [
        "cutoff_0001",
    ],
    name="motifscan",
)

latent_design = pd.Series(["overexpression"], name="latent")
# designs.append(chd.utils.crossing(dataset_design, method_design, latent_design, peaks_design, motifscan_design))

# %%
design = pd.concat(designs)


# %%
class Prediction(chd.flow.Flow):
    pass


# %%
scores = {}
for _, (dataset_name, latent_name, method_name, peaks_name, motifscan_name) in design[
    ["dataset", "latent", "method", "peaks", "motifscan"]
].iterrows():
    prediction = Prediction(
        chd.get_output()
        / "prediction_likelihood"
        / dataset_name
        / promoter_name
        / latent_name
        / method_name
    )

    scores_dir = prediction.path / "scoring" / peaks_name / motifscan_name
    if (scores_dir / "scores.pkl").exists():
        scores_ = pd.read_pickle(scores_dir / "scores.pkl")
        scores[(dataset_name, method_name, peaks_name, motifscan_name)] = scores_
    else:
        print(scores_dir)
scores = pd.concat(
    scores, names=["dataset", "method", "peaks", "motifscan", *scores_.index.names]
)
scores["n_cells"] = [int(s) for s in scores["n_cells"].values]


# %%
def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j + 1 :: 2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


# %%
scores["slope"].values[scores["slope"].values < -0] = 1.0
plotdata = scores.reset_index()

# %%
# np.log(scores.xs("macs2", level = "peaks").xs("KLF4_7", level = "dataset").xs("cutoff_0001", level = "motifscan")["slope"].unstack()).style.bar()

# %%
# scores.xs("macs2", level = "peaks").xs("lymphoma", level = "dataset").xs("cutoff_0001", level = "motifscan")["slope"].unstack().style.bar()

# %%
# (scores.xs("cellranger", level = "peaks").xs("lymphoma", level = "dataset").xs("cutoff_0001", level = "motifscan")["slope"].unstack()).style.bar()

# %%
peaks_info = pd.DataFrame(
    [["macs2"], ["cellranger"], ["rolling_500"]], columns=["peaks"]
).set_index("peaks")
peaks_info["x"] = np.arange(peaks_info.shape[0])

dataset_info = pd.DataFrame({"dataset": design["dataset"].unique()}).set_index(
    "dataset"
)
dataset_info["color"] = sns.color_palette(n_colors=len(dataset_info))

# %%
from chromatinhd import quasirandom

# %%
method_info = pd.DataFrame({"method": design["method"].unique()}).set_index("method")

# %%
group_columns = ["peaks", "method"]

# %%
x_info = x_info = scores.groupby(group_columns)[[]].first()
x_info["x"] = np.arange(len(x_info))


# %%
def create_labelling(x_info, col):
    y = x_info.index.to_frame()[col]
    y = pd.DataFrame({"level": pd.Categorical(y)})
    y["x"] = x_info["x"].values
    y["group"] = np.hstack([[0], np.cumsum(np.diff(y["level"].cat.codes) != 0)])
    labelling = pd.DataFrame(
        {
            "right": y.groupby("group")["x"].max() + 0.5,
            "left": y.groupby("group")["x"].min() - 0.5,
        }
    )
    labelling["center"] = (
        labelling["left"] + (labelling["right"] - labelling["left"]) / 2
    )
    labelling["label"] = y["level"].cat.categories
    return labelling


# %%
fig, ax = plt.subplots(figsize=(4, (len(x_info) / 4)))
for idx, plotdata_ in plotdata.groupby(group_columns):
    x = x_info.loc[idx, "x"]
    # ax.scatter(peaks_info.loc[peaks, "x"] + simple_beeswarm(plotdata_peaks["slope"], nbins = 20)/5, plotdata_peaks["slope"], c = dataset_info.loc[plotdata_peaks["dataset"], "color"])
    offset = (
        quasirandom.offset(np.log(plotdata_["slope"]), nbins=100, method="quasirandom")
        / 2
    )
    ax.scatter(
        plotdata_["slope"],
        x + offset,
        c=dataset_info.loc[plotdata_["dataset"], "color"],
        s=2,
    )

    plotdata_agg = plotdata_.groupby("dataset")[["slope"]].mean()
    ax.scatter(
        plotdata_agg["slope"],
        np.repeat(x, len(plotdata_agg)),
        c=dataset_info.loc[plotdata_agg.index, "color"],
        s=30,
        marker="o",
        lw=1,
        ec="black",
    )

    # m = np.exp(np.log(plotdata_agg["slope"]).median())
    m = np.exp(np.log(plotdata_["slope"]).mean())
    ax.plot(np.repeat(m, 2), [x - 0.4, x + 0.4], c="#333333FF")
    txt = ax.text(
        m,
        x,
        "{:.2f}".format(m),
        ha="center",
        color="white",
        va="center",
        fontweight="bold",
    )
    txt.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=1, foreground="#333333FF"),
            mpl.patheffects.Normal(),
        ]
    )

ax.set_xlim(1 / 4, 8)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["¼", "½", "1", "2", "4"])
ax.axvline(1, zorder=0, dashes=(2, 2), color="#333333")
ax.set_yticks(x_info.x)
ax.set_yticklabels(x_info.index.get_level_values(group_columns[-1]), rotation=0)

ax.set_ylim(x_info["x"].min() - 0.5, x_info["x"].max() + 0.5)


ax3 = ax.twinx()
ax3.set_ylim(ax.get_ylim())

labelling = create_labelling(x_info, "peaks")

ax3.spines["right"].set_position(("axes", 1.05))
ax3.tick_params("both", length=0, width=0, which="minor")
ax3.tick_params("both", direction="in", which="major")
ax3.yaxis.set_ticks_position("right")
ax3.yaxis.set_label_position("right")

ax3.set_yticks(labelling[["right", "left"]].values.flatten())
ax3.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
ax3.yaxis.set_minor_locator(mpl.ticker.FixedLocator(labelling["center"].values))
ax3.yaxis.set_minor_formatter(mpl.ticker.FixedFormatter(labelling["label"]))
ax3.set_ylabel("peaks")

sns.despine(ax=ax)
sns.despine(ax=ax3, right=False)

ax.set_xlabel("overenrichment")

# %% [markdown]
# ### Correlation with # cells in cluster

# %%
import scipy

# %%
scores_oi = scores.query("peaks == 'cellranger'").query(
    "method == 'v4_128-64-32_30_rep'"
)

lm = scipy.stats.linregress(np.log(scores_oi["n_cells"]), np.log(scores_oi["slope"]))

x1 = 6
x2 = 5.0

xy1 = [np.exp(x1), np.exp(lm.intercept + x1 * lm.slope)]
xy2 = [np.exp(x2), np.exp(lm.intercept + x2 * lm.slope)]

print(f"p = {lm.pvalue}")
print(f"r = {lm.rvalue}")
print(f"r2 = {lm.rvalue**2}")

# %%
settings = [
    # {"peaks":"cellranger", "method":'v4_128-64-32'},
    # {"peaks":"macs2", "method":'v4_128-64-32'},
    # {"peaks":"cellranger", "method":'v4_64-32'},
    # {"peaks":"macs2", "method":'v4_64-32'},
    # {"peaks":"cellranger", "method":'v4_128-64-32_30'},
    # {"peaks":"macs2", "method":'v4_128-64-32_30'},
    {"peaks": "cellranger", "method": "v4_128-64-32_30_rep"},
    {"peaks": "macs2", "method": "v4_128-64-32_30_rep"},
]

# %%
plotdata = scores_oi.reset_index()

# %% tags=[]
fig, axes = plt.subplots(
    1, len(settings), figsize=(len(settings) * 2, 2), squeeze=False
)
for ax, setting in zip(axes[0], settings):
    scores_oi = scores
    for k, v in setting.items():
        scores_oi = scores_oi.query(f"{k} == '{v}'")

    lm = scipy.stats.linregress(
        np.log(scores_oi["n_cells"]), np.log(scores_oi["slope"])
    )

    x1 = 6
    x2 = 5.0

    xy1 = [np.exp(x1), np.exp(lm.intercept + x1 * lm.slope)]
    xy2 = [np.exp(x2), np.exp(lm.intercept + x2 * lm.slope)]

    plotdata = scores_oi.reset_index()

    ax.scatter(
        plotdata["n_cells"],
        plotdata["slope"],
        c=dataset_info.loc[plotdata["dataset"], "color"],
        s=5,
    )
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["¼", "½", "1", "2", "4"])
    ax.set_xscale("log")
    # ax.axline(np.array([1.0, 10.]), slope = 1.)
    txt = ax.axline(xy1, xy2, color="black")
    txt.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFF88"),
            mpl.patheffects.Normal(),
        ]
    )
    ax.set_xlabel("# of cells in cluster")

    ax.set_ylim(1 / 2, 8)

    ax.axhline(1, zorder=0, dashes=(2, 2), color="#333333")

    ax.annotate(
        f"$r={lm.rvalue:.2f}$\n$p={lm.pvalue:.2f}$",
        (0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        va="top",
    )

    ax.set_title(setting, fontsize=5)

axes[0][0].set_ylabel("overenrichment", rotation=0, va="bottom", ha="right")

# %% [markdown]
# ## Known motif

# %%
designs = []

dataset_design = pd.DataFrame(
    [
        ["FLI1_7", "FLI1"],
        # ["PAX2_7", "PAX3"],
        ["NHLH1_7", "HEN1"],
        ["CDX1_7", "CDX1"],
        ["CDX2_7", "CDX2"],
        # ["MSGN1_7",
        ["KLF4_7", "KLF4"],
        ["KLF5_7", "KLF5"],
        # ["PTF1A_4", "PTF1A"],
    ],
    columns=["dataset", "known_motif"],
)

promoter_name, (padding_negative, padding_positive) = "10k10k", (-10000, 10000)

method_design = pd.Series(
    [
        # "v4_128-64-32",
        # "v4_128-64-32_rep",
        # "v4_128",
        # "v4_64",
        # "v4_32",
        # "v4_64-32",
        # "v4_256-128-64-32",
        # "v4_128-64-32_30",
        # "v4_128-64-32_30_freescale",
        # "v4_128-64-32_30_scalelik",
        # "v4_128-64-32_30_laplace0.05",
        # "v4_128-64-32_30_laplace0.1",
        # "v4_128-64-32_30_laplace1.0",
        # "v4_128-64-32_30_normal0.05",
        "v4_128-64-32_30_rep",
    ],
    name="method",
)

peaks_design = pd.Series(
    [
        "cellranger",
        "macs2",
        "rolling_500",
    ],
    name="peaks",
)

motifscan_design = pd.Series(
    [
        "cutoff_0001",
    ],
    name="motifscan",
)

latent_design = pd.Series(["overexpression"], name="latent")
designs.append(
    chd.utils.crossing(
        dataset_design, method_design, latent_design, peaks_design, motifscan_design
    )
)

design = pd.concat(designs)

# %%
dataset_info = pd.DataFrame({"dataset": design["dataset"].unique()}).set_index(
    "dataset"
)
dataset_info["color"] = sns.color_palette(n_colors=len(dataset_info))


# %%
class Prediction(chd.flow.Flow):
    pass


# %%
scores = {}
scores_mean = {}
for _, (
    dataset_name,
    latent_name,
    method_name,
    peaks_name,
    motfiscan_name,
    known_motif,
) in design[
    ["dataset", "latent", "method", "peaks", "motifscan", "known_motif"]
].iterrows():
    prediction = Prediction(
        chd.get_output()
        / "prediction_likelihood"
        / dataset_name
        / promoter_name
        / latent_name
        / method_name
    )

    scores_dir = prediction.path / "scoring" / peaks_name / motifscan_name
    if (scores_dir / "scores.pkl").exists():
        scores_ = pd.read_pickle(scores_dir / "motifscores_all.pkl")

        motifs = scores_.index.get_level_values("motif")[
            scores_.index.get_level_values("motif").str.contains(known_motif)
        ].unique()
        scores[(dataset_name, method_name, peaks_name, motifscan_name)] = scores_.loc[
            motifs
        ]
        scores_mean[(dataset_name, method_name, peaks_name, motifscan_name)] = (
            scores_.groupby("group").mean().reset_index()
        )
    else:
        print(scores_dir)
scores = pd.concat(
    scores, names=["dataset", "method", "peaks", "motifscan", *scores_.index.names]
)
scores_mean = pd.concat(
    scores_mean, names=["dataset", "method", "peaks", "motifscan", *scores_.index.names]
)

# %%
plotdata = scores.query("group == 0").reset_index()
plotdata["diff"] = plotdata["logodds_peak"] - plotdata["logodds_region"]
plotdata = plotdata.sort_values("diff", ascending=False)

# %%
fig, ax = plt.subplots(figsize=(1.0, 2))
for i, (_, row) in enumerate(plotdata.iterrows()):
    ax.plot(
        [0, 1],
        row[["odds_peak", "odds_region"]].values,
        color=dataset_info.loc[row["dataset"], "color"],
        marker=".",
        zorder=-i,
        alpha=0.8,
    )
txt = ax.plot(
    [0, 1],
    np.exp(plotdata[["logodds_peak", "logodds_region"]].mean()),
    color="#333",
    marker="o",
)
txt[0].set_path_effects(
    [
        mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFF88"),
        mpl.patheffects.Normal(),
    ]
)
# ax.scatter(1, np.exp(plotdata["logodds_region"].mean()), color = "#333333", zorder = 10)
ax.set_ylim(1 / 2, 8)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["¼", "½", "1", "2", "4"])
ax.axhline(1, zorder=0, dashes=(2, 2), color="#333333")
ax.set_xticks([0, 1])
ax.set_xlim(-0.4, 1.4)
ax.set_xticklabels(["peak", "ours"])

# %%
sns.heatmap(
    scores.query("group == 0")[["logodds_peak", "logodds_region"]].T,
    vmin=-1,
    vmax=1,
    cmap=mpl.cm.PuOr_r,
)

# %%
scores.query("group == 0")[["logodds_region", "logodds_peak"]].T

# %%

# %%
