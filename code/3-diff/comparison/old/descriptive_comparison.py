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
import IPython

if IPython.get_ipython():
    IPython.get_ipython().magic("load_ext autoreload")
    IPython.get_ipython().magic("autoreload 2")
    IPython.get_ipython().magic("config InlineBackend.figure_format='retina'")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    default="pbmc10k",
    # default = "GSE198467_H3K27ac",
    # default = "brain",
    # default = "pbmc10k"
)
parser.add_argument("--promoter_name", default="10k10k")
# parser.add_argument("--latent_name", default = "celltype")
parser.add_argument("--latent_name", default="leiden_0.1")
parser.add_argument("--method_name", default="v9_128-64-32")

try:
    get_ipython().__class__.__name__
    in_jupyter = True
except:
    in_jupyter = False
parameters = vars(parser.parse_args("" if in_jupyter else None))
dataset_name = parameters["dataset_name"]
promoter_name = parameters["promoter_name"]
latent_name = parameters["latent_name"]
method_name = parameters["method_name"]

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / dataset_name


# %% [markdown]
# ### Load data

# %%
class Prediction(chd.flow.Flow):
    pass


prediction = Prediction(
    chd.get_output()
    / "prediction_likelihood"
    / dataset_name
    / promoter_name
    / latent_name
    / method_name
)
# model = chd.load((prediction.path / "model_0.pkl").open("rb"))

# %%
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

probs_diff = probs - probs.mean(1, keepdims=True)

# %%
design["gene_ix"] = design["gene_ix"]

# %%
window = {"10k10k": np.array([-10000, 10000])}[promoter_name]
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# Interpolate probs for individual positions

# %%
x = (
    (design["coord"].values)
    .astype(int)
    .reshape(
        (
            len(design["gene_ix"].cat.categories),
            len(design["active_latent"].cat.categories),
            len(design["coord"].cat.categories),
        )
    )
)
desired_x = torch.arange(*window)
probs_interpolated = chd.utils.interpolate_1d(
    desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
).numpy()

# %% [markdown]
# ----

# %% [markdown]
# Temporary look at local autocorrelation

# %%
probs_diff = probs_interpolated - probs_interpolated.mean(-1, keepdims=True)

# %%
gene_ix = transcriptome.gene_ix("CD74")

# %%
x = torch.from_numpy(probs_diff.transpose(2, 1, 0)[:, :, gene_ix])

# %%
cors = torch.cov(x) / (x.std(1) * (x.std(1)[:, None]))

# %%
pad = 200

# %%
cors_padded = torch.nn.functional.pad(cors, (pad, pad), value=torch.nan)

# %%
n_positions = cors_padded.shape[0]

# %%
indices = (
    torch.arange(0, pad * 2 + 1)[None, :]
    + torch.arange(0, n_positions)[:, None]
    + torch.arange(0, n_positions)[:, None] * cors_padded.shape[1]
)

# %%
resolution = 1 / 1000

# %%
fig, ax = plt.subplots(
    figsize=((window[1] - window[0]) * resolution, (pad * 2) * resolution)
)
ax.matshow(cors_padded.flatten()[indices].T, vmin=-1, vmax=1, cmap=mpl.cm.PiYG_r)
ax.axhline(pad, color="black", dashes=(2, 2), lw=1)

# %% [markdown]
# ----

# %% [markdown]
# ## Load slices

# %%
prob_cutoff = np.log(1.0)

# %%
scores_dir = prediction.path / "scoring" / "significant_up"
pureregionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %% [markdown]
# ### Relative to peaks

# %%
import chromatinhd.peakcounts

# peakcaller = "cellranger"
# peakcaller = "macs2"
# peakcaller = "macs2_improved";peakcaller_label = "MACS2"
# peakcaller = "encode_screen";peakcaller_label = "ENCODE SCREEN"

peakcaller = "macs2_leiden_0.1_merged"

# peakcaller = "rolling_500";peakcaller_label = "Sliding window 500"

# peakcaller = "rolling_100"

# diffexp = "signac"
diffexp = "scanpy"
# diffexp = "scanpy_wilcoxon"

# %%
scores_dir = (
    chd.get_output()
    / "prediction_differential"
    / dataset_name
    / promoter_name
    / latent_name
    / diffexp
    / peakcaller
)
peakresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %%
peakresult.get_slicescores()

# %%
scores_dir = prediction.path / "scoring" / peakcaller / diffexp
regionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %% [markdown]
# ## Positional overlap

# %%
methods_info = pd.DataFrame(
    [
        [
            "peak",
            "#FF4136",
            f"Unique to differential {chdm.peakcallers.loc[peakcaller, 'label']}",
        ],
        ["common", "#B10DC9", f"Common"],
        ["region", "#0074D9", f"Unique to ChromatinHD"],
    ],
    columns=["method", "color", "label"],
).set_index("method")

# %%
position_chosen_region = regionresult.position_chosen
position_chosen_peak = peakresult.position_chosen

# %%
(position_chosen_region & position_chosen_peak).sum() / (
    position_chosen_region | position_chosen_peak
).sum()

# %%
position_region = np.where(position_chosen_region)[0]
position_peak = np.where(position_chosen_peak)[0]
position_indices_peak_unique = np.where(
    position_chosen_peak & (~position_chosen_region)
)[0]
position_indices_region_unique = np.where(
    position_chosen_region & (~position_chosen_peak)
)[0]
position_indices_common = np.where(position_chosen_region & position_chosen_peak)[0]
position_indices_intersect = np.where(position_chosen_region | position_chosen_peak)[0]

# %%
positions_region_unique = (
    position_indices_region_unique % (window[1] - window[0])
) + window[0]
positions_region = (position_region % (window[1] - window[0])) + window[0]

positions_peak_unique = (
    position_indices_peak_unique % (window[1] - window[0])
) + window[0]
positions_peak = (position_peak % (window[1] - window[0])) + window[0]

positions_common = (position_indices_common % (window[1] - window[0])) + window[0]

positions_intersect = (position_indices_intersect % (window[1] - window[0])) + window[0]

# %%
# 200 bins => 100 bp bin size
binmids = np.linspace(*window, 200 + 1)
cuts = (binmids + (binmids[1] - binmids[0]) / 2)[:-1]

# %%
positions_region_unique_bincounts = np.bincount(
    np.digitize(positions_region_unique, cuts, right=True), minlength=len(cuts) + 1
)
positions_region_bincounts = np.bincount(
    np.digitize(positions_region, cuts), minlength=len(cuts) + 1
)

positions_peak_unique_bincounts = np.bincount(
    np.digitize(positions_peak_unique, cuts), minlength=len(cuts) + 1
)
positions_peak_bincounts = np.bincount(
    np.digitize(positions_peak, cuts), minlength=len(cuts) + 1
)

positions_common_bincounts = np.bincount(
    np.digitize(positions_common, cuts), minlength=len(cuts) + 1
)

positions_intersect_bincounts = np.bincount(
    np.digitize(positions_intersect, cuts), minlength=len(cuts) + 1
)

# %%
fig, ax = plt.subplots()
ax.plot(
    binmids,
    (positions_common_bincounts)
    / (positions_intersect_bincounts)
    / (positions_common_bincounts.sum() / positions_intersect_bincounts.sum()),
    label="common",
)
ax.plot(
    binmids,
    (positions_peak_unique_bincounts / positions_peak_bincounts)
    / (positions_peak_unique_bincounts.sum() / positions_peak_bincounts.sum()),
    label="peak_unique",
)
ax.plot(
    binmids,
    (positions_region_unique_bincounts / positions_region_bincounts)
    / (positions_region_unique_bincounts.sum() / positions_region_bincounts.sum()),
    label="region_unique",
)
ax.axhline(1, dashes=(2, 2), color="#333")
ax.set_yscale("log")
ax.legend()
ax.set_yticks([1 / 2, 1, 2, 1 / 4])
ax.set_yticklabels([1 / 2, 1, 2, 1 / 4])

# %%
plotdata = pd.DataFrame(
    {
        "common": positions_common_bincounts,
        "peak_unique": positions_peak_unique_bincounts,
        "region_unique": positions_region_unique_bincounts,
        "intersect": positions_intersect_bincounts,
        "position": binmids,
    }
)
plotdata["peak_unique_density"] = plotdata["peak_unique"] / plotdata["intersect"]
plotdata["common_density"] = plotdata["common"] / plotdata["intersect"]
plotdata["region_unique_density"] = plotdata["region_unique"] / plotdata["intersect"]

# %%
plotdata_last = (
    plotdata[["peak_unique_density", "common_density", "region_unique_density"]]
    .iloc[-10]
    .to_frame(name="density")
)
plotdata_last["cumulative_density"] = (
    np.cumsum(plotdata_last["density"]) - plotdata_last["density"] / 2
)
plotdata_mean = pd.Series(
    {
        "peak_unique_density": positions_peak_unique_bincounts.sum()
        / positions_intersect_bincounts.sum(),
        "region_unique_density": positions_region_unique_bincounts.sum()
        / positions_intersect_bincounts.sum(),
        "common_density": positions_common_bincounts.sum()
        / positions_intersect_bincounts.sum(),
    }
)

# %%
import textwrap

fig, ax = plt.subplots(figsize=(2.0, 2.0))
ax.stackplot(
    binmids,
    plotdata["peak_unique_density"],
    plotdata["common_density"],
    plotdata["region_unique_density"],
    baseline="zero",
    colors=methods_info["color"],
    lw=1.0,
    ec="#FFFFFF",
)
ax.set_xlim(*window)
ax.set_ylim(0, 1)
transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
x = 1.02

label_peak = "\n".join(
    textwrap.wrap(
        f"{plotdata_mean['peak_unique_density']:.0%} {methods_info.loc['peak', 'label']}",
        20,
        drop_whitespace=False,
    )
)
ax.text(
    x,
    plotdata_last.loc["peak_unique_density", "cumulative_density"],
    label_peak,
    transform=transform,
    ha="left",
    va="center",
    color=methods_info.loc["peak", "color"],
)

label_common = (
    f"{plotdata_mean['common_density']:.0%} {methods_info.loc['common', 'label']}"
)
ax.text(
    x,
    plotdata_last.loc["common_density", "cumulative_density"],
    label_common,
    transform=transform,
    ha="left",
    va="center",
    color=methods_info.loc["common", "color"],
)

label_region = "\n".join(
    textwrap.wrap(
        f"{plotdata_mean['region_unique_density']:.0%} {methods_info.loc['region', 'label']}",
        20,
        drop_whitespace=False,
    )
)
ax.text(
    x,
    plotdata_last.loc["region_unique_density", "cumulative_density"],
    label_region,
    transform=transform,
    ha="left",
    va="center",
    color=methods_info.loc["region", "color"],
)

ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_xlabel("Distance to TSS")

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "peak_vs_chhd_positional")

# %%
import textwrap

fig, axes = plt.subplots(2, 2, figsize=(4.0, 4.0), gridspec_kw={"hspace": 0.05})

ax = axes[0, 0]
ax.stackplot(
    binmids,
    plotdata["peak_unique_density"],
    plotdata["common_density"],
    plotdata["region_unique_density"],
    baseline="zero",
    colors=methods_info["color"],
    lw=0.0,
    # lw=1.0,
    # ec="#FFFFFF",
)
ax.set_xlim(*window)
ax.set_ylim(0, 1)
transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
x = 1.02

label_peak = "\n".join(
    textwrap.wrap(
        f"{plotdata_mean['peak_unique_density']:.0%} {methods_info.loc['peak', 'label']}",
        20,
        drop_whitespace=False,
    )
)
ax.text(
    x,
    plotdata_last.loc["peak_unique_density", "cumulative_density"],
    label_peak,
    transform=transform,
    ha="left",
    va="center",
    color=methods_info.loc["peak", "color"],
)

label_common = (
    f"{plotdata_mean['common_density']:.0%} {methods_info.loc['common', 'label']}"
)
ax.text(
    x,
    plotdata_last.loc["common_density", "cumulative_density"],
    label_common,
    transform=transform,
    ha="left",
    va="center",
    color=methods_info.loc["common", "color"],
)

label_region = "\n".join(
    textwrap.wrap(
        f"{plotdata_mean['region_unique_density']:.0%} {methods_info.loc['region', 'label']}",
        20,
        drop_whitespace=False,
    )
)
ax.text(
    x,
    plotdata_last.loc["region_unique_density", "cumulative_density"],
    label_region,
    transform=transform,
    ha="left",
    va="center",
    color=methods_info.loc["region", "color"],
)

ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_yticks([0, 0.5, 1])
ax.set_xticks([])

ax = axes[1, 0]
ax.fill_between(binmids, plotdata["common"], color=methods_info.loc["common", "color"])
ax.fill_between(
    binmids,
    plotdata["common"],
    plotdata["region_unique"],
    color=methods_info.loc["region", "color"],
    # lw = 1.0,
    # ec = "#FFFFFF",
)
ax.fill_between(
    binmids, -plotdata["peak_unique"], color=methods_info.loc["peak", "color"]
)
ax.axvline(0, dashes=(2, 2), color="#333", lw=1)
ax.set_xlim(*window)
ax.set_xticks([window[0], 0, window[1]])
ax.xaxis.set_major_formatter(chromatinhd.plot.gene_ticker)
ax.set_xlabel("Distance to TSS")

ax = axes[1, 1]
ax.fill_between(binmids, plotdata["common"], color=methods_info.loc["common", "color"])
ax.fill_between(
    binmids,
    plotdata["common"],
    plotdata["region_unique"],
    color=methods_info.loc["region", "color"],
    # lw = 1.0,
    # ec = "#FFFFFF",
)
ax.fill_between(
    binmids, -plotdata["peak_unique"], color=methods_info.loc["peak", "color"]
)
ax.axvline(0, dashes=(2, 2), color="#333", lw=1)
ax.set_xlim(*window)
ax.set_xlim(-3000, 3000)
ax.xaxis.set_major_formatter(chromatinhd.plot.gene_ticker)
ax.set_yticks([])

ax = axes[0, 1]
ax.axis("off")

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "peak_vs_chhd_positional")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
ax1.stackplot(
    binmids,
    plotdata["peak_unique"],
    plotdata["common"],
    plotdata["region_unique"],
    baseline="zero",
    colors=methods_info["color"],
    lw=1,
    ec="#FFFFFF33",
)
ax1.set_xlim(*window)

ax2.stackplot(
    binmids,
    plotdata["peak_unique"],
    plotdata["common"],
    plotdata["region_unique"],
    baseline="zero",
    colors=methods_info["color"],
    lw=1,
    ec="#FFFFFF33",
)
ax2.set_xlim(-3000, 3000)

# %% [markdown]
# ## Bystander regions

# %%
probs_diff_interpolated = probs_interpolated - probs_interpolated.mean(1, keepdims=True)

# %%
probs_interpolated.shape

# %%
print(
    probs_diff_interpolated.flatten()[position_indices_peak_unique].mean(),
    probs_diff_interpolated.flatten()[position_indices_region_unique].mean(),
    probs_diff_interpolated.flatten()[position_indices_common].mean(),
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
sns.ecdfplot(
    np.exp(probs_diff_interpolated.flatten()[position_indices_peak_unique][:1000000]),
    color=methods_info.loc["peak", "color"],
    ax=ax,
)
sns.ecdfplot(
    np.exp(probs_diff_interpolated.flatten()[position_indices_region_unique][:1000000]),
    color=methods_info.loc["region", "color"],
    ax=ax,
)
sns.ecdfplot(
    np.exp(probs_diff_interpolated.flatten()[position_indices_common][:1000000]),
    color=methods_info.loc["common", "color"],
    ax=ax,
)
ax.set_xscale("log")
ax.set_xlim(0.5, 8)
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["½", 1, 2, 4, 8])
ax.axvline(1, dashes=(2, 2), color="#333333")
ax.set_xlabel("Accessibility fold change")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "bystander_chhd_foldchange")

# %% [markdown]
# ### Per-peak look

# %%
peakscores = (
    peakresult.get_slicescores()
    .join(peakresult.get_slicedifferential(probs_interpolated))
    .join(peakresult.get_sliceaverages(probs_interpolated))
)

# %%
print(
    f"Overall, we found that the percentage of non-differential positions was very variable, with {(peakscores['differential_positions'] == 0).mean():.1%} of the differential MACS2 peaks containing no differential positions, while {(peakscores['differential_positions'] == 1).mean():.1%} was fully differential (FIG:bystander_number_of_differential_positions). Only {(peakscores['differential_positions'] >= 0.5).mean():.1%} of differential peaks contained less than 50% differential positions (FIG:bystander_number_of_differential_positions), a statistic particularly driven by larger peaks (>1kb)."
)

# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))
sns.ecdfplot(peakscores["differential_positions"], color="#333333")
peakscores["large"] = peakscores["length"] > 1000

plotdata = peakscores.loc[peakscores["large"]]
color = "orange"
label = "Peak size > 1kb"
sns.ecdfplot(plotdata["differential_positions"], color=color)
q = 0.8
median = np.quantile(plotdata["differential_positions"], q)
ax.annotate(
    label,
    (median, q),
    xytext=(-15, 15),
    ha="right",
    va="top",
    arrowprops=dict(arrowstyle="-", color=color),
    textcoords="offset points",
    fontsize=8,
    bbox=dict(boxstyle="round", fc="w", ec=color),
)

plotdata = peakscores.loc[~peakscores["large"]]
color = "green"
label = "Peak size < 1kb"
sns.ecdfplot(plotdata["differential_positions"], color=color)
q = 0.3
median = np.quantile(plotdata["differential_positions"], q)
ax.annotate(
    label,
    (median, q),
    xytext=(15, -15),
    ha="left",
    va="top",
    arrowprops=dict(arrowstyle="-", color=color),
    textcoords="offset points",
    fontsize=8,
    bbox=dict(boxstyle="round", fc="w", ec=color),
)


plotdata = peakscores
color = "black"
label = "All"
sns.ecdfplot(plotdata["differential_positions"], color=color)
q = 0.5
median = np.quantile(plotdata["differential_positions"], q)
ax.annotate(
    label,
    (median, q),
    xytext=(15, -15),
    ha="left",
    va="top",
    arrowprops=dict(arrowstyle="-", color=color),
    textcoords="offset points",
    fontsize=8,
    bbox=dict(boxstyle="round", fc="w", ec=color),
)


ax.set_xlabel("% differential positions\nwithin differential peak")
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_yticks([0, 0.5, 1])

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "bystander_number_of_differential_positions")
elif peakcaller in ["rolling_50", "rolling_100", "rolling_500"]:
    ax.set_title(chdm.peakcallers.loc[peakcaller, "label"])
    manuscript.save_figure(
        fig, "4", f"bystander_number_of_differential_positions_{peakcaller}"
    )

# %%
upstream_cutoff = -1000 - window[0]
downstream_cutoff = +1000 - window[0]


def classify_position_group(x, upstream_cutoff, downstream_cutoff):
    return np.array(["upstream", "downstream", "promoter"])[
        np.argmax(
            np.vstack(
                [
                    x.end < upstream_cutoff,
                    x.start > downstream_cutoff,
                    np.repeat(True, len(x)),
                ]
            ),
            0,
        )
    ]


position_group_info = pd.DataFrame(
    [
        ["upstream", "Upstream -10kb→-1kb"],
        ["promoter", "-1kb→+1kb"],
        ["downstream", "Downstream +1kb→+10kb"],
    ],
    columns=["position_group", "label"],
).set_index("position_group")

# %%
peakscores["position_group"] = classify_position_group(
    peakscores, upstream_cutoff, downstream_cutoff
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
for position_group, peakscores_group in peakscores.groupby("position_group"):
    sns.ecdfplot(
        peakscores_group["differential_positions"],
        label=position_group_info.loc[position_group, "label"],
    )
plt.legend(bbox_to_anchor=(0.5, -0.4), loc="upper center", fontsize=8, ncol=1)
ax.set_xlabel("% differential positions\nwithin peak")
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_yticks([0, 0.5, 1])

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(
        fig, "4", "bystander_number_of_differential_positions_positiongroups"
    )

# %%
fig, axes = plt.subplots(1, 3, figsize=(1.5 * 3, 1.5), sharey=True, sharex=True)

score = "max"
xlim = (0.1, 100)
xlabel = "Maximum accessibility"
for ax, (position_group, plotdata_positiongroup) in zip(
    axes, peakscores.groupby("position_group")
):
    sns.ecdfplot(
        peakscores["differential_positions"],
        ax=ax,
        color="#33333333",
    )
    sns.ecdfplot(plotdata_positiongroup["differential_positions"], ax=ax, color="black")
    ax.set_title(
        "\n".join(
            textwrap.wrap(position_group_info.loc[position_group]["label"], width=10)
        )
    )
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("")
ax = axes[0]
ax.set_xlabel("% differential positions\nwithin peak")
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_yticks([0, 0.5, 1])

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(
        fig, "4", "bystander_number_of_differential_positions_positiongroups"
    )

# %% [markdown]
# ### Examples

# %%
slices_oi = (
    peakscores.query("max_lfc > log(4)").sort_values("differential_positions").head(10)
)
# slices_oi = peakscores.query("max_lfc > log(4)").assign(group = lambda x:(x.passenger * 10)//1).groupby("group").first()
# https://media.tenor.com/wn2_Qq6flogAAAAM/magical-magic.gif
slices_oi = slices_oi.sort_values("length").iloc[
    np.concatenate(
        [np.arange(len(slices_oi), step=2), np.arange(len(slices_oi), 0, step=-2) - 1]
    )
]

# %%
probs_diff = probs - probs.mean(1, keepdims=True)

# %%
region_position_chosen = pureregionresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), (window[1] - window[0]))
)
peak_position_chosen = peakresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), (window[1] - window[0]))
)

# %%
import polyptich.grid

main = polyptich.grid.Grid()
wrap = main[0, 0] = polyptich.grid.Wrap(5, padding_width=0.1, padding_height=0.3)
fig = polyptich.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005

panel_height = 0.5

total_width_cutoff = 10

hatch_color = "#FFF4"

for slice_oi in slices_oi.to_dict(orient="records"):
    slice_oi = dict(slice_oi)

    expanded_slice_oi = slice_oi.copy()
    expanded_slice_oi["start"] = np.clip(
        slice_oi["start"] - 1000, *(window - window[0])
    )
    expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 1000, *(window - window[0]))

    window_oi = (
        np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
    )

    gene_oi = expanded_slice_oi["gene_ix"]
    cluster_ix = expanded_slice_oi["cluster_ix"]
    cluster_info_oi = cluster_info.iloc[[cluster_ix]]

    plotdata_atac = (
        design.query("gene_ix == @gene_oi")
        .copy()
        .rename(columns={"active_latent": "cluster"})
        .set_index(["coord", "cluster"])
        .drop(columns=["batch", "gene_ix"])
    )
    plotdata_atac["prob"] = probs[gene_oi].flatten()
    plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

    plotdata_atac["prob"] = (
        plotdata_atac["prob"]
        - np.log(
            plotdata_atac.reset_index()
            .groupby(["cluster"])
            .apply(
                lambda x: np.trapz(
                    np.exp(x["prob"]),
                    x["coord"].astype(float) / (window[1] - window[0]),
                )
            )
        ).mean()
    )
    plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

    resolution = 0.0005
    panel_width = (window_oi[1] - window_oi[0]) * resolution

    # differential atac
    wrap_differential = chd.models.diff.plot.Differential(
        plotdata_atac,
        plotdata_atac_mean,
        cluster_info_oi,
        window_oi,
        panel_width,
        panel_height,
        padding_height=padding_height,
        title=False,
    )
    wrap.add(wrap_differential)

    ax = wrap_differential.elements[0].ax

    # add region and peak unique spans
    region_position_chosen_oi = region_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]
    peak_position_chosen_oi = peak_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]

    chd.models.diff.plot.CommonUnique(
        ax,
        peak_position_chosen_oi,
        region_position_chosen_oi,
        expanded_slice_oi,
        window,
        methods_info,
    )

    # add labels
    gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
    cluster_label = cluster_info.query("dimension == @cluster_ix")["label"][0]
    position_label = str(int(slice_oi["start"] + slice_oi["length"] / 2) + window[0])

    chd.models.diff.plot.LabelSlice(ax, gene_label, cluster_label, slice_oi, window)

legend_panel = main[0, 1] = polyptich.grid.Panel((1, panel_height * 2))
legend_panel.ax.axis("off")

fig.plot()

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "bystander_examples")

# %% [markdown]
# ## Background

# %% [markdown]
# ### Examples

# %%
region_position_chosen = pureregionresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), (window[1] - window[0]))
)
peak_position_chosen = peakresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), (window[1] - window[0]))
)

# %%
slicetopologies = regionresult.get_slicetopologies(probs_interpolated).join(
    regionresult.get_slicescores()
)

# %%
slices_oi = (
    slicetopologies.query("dominance < 0.4")
    .query("(mid > (@window[0] + 500)) & (mid < (@window[1] - 500))")
    .query("length > 100")
    .sort_values("differentialdominance", ascending=False)
    .groupby("gene_ix", sort=False, as_index=False)
    .first()
    # .sort_values("differentialdominance", ascending=False)
    .head(30)
)

# %%
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)


# %%
peaks_folder = chd.get_output() / "peaks" / dataset_name

# %%
import polyptich.grid

main = wrap = polyptich.grid.Wrap(5, padding_width=0.1, padding_height=0.3)
fig = polyptich.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005

panel_height = 0.5

total_width_cutoff = 10

n = 0

for slice_ix, slice_oi in enumerate(slices_oi.to_dict(orient="records")):
    n_captured = peak_position_chosen[
        slice_oi["gene_ix"], slice_oi["cluster_ix"], slice_oi["start"] : slice_oi["end"]
    ].mean()
    if n_captured > 0.1:
        continue

    if n >= 10:
        break

    n += 1

    grid = polyptich.grid.Grid(
        margin_height=0, margin_width=0, padding_height=0, padding_width=0
    )
    main.add(grid)

    slice_oi = dict(slice_oi)

    expanded_slice_oi = slice_oi.copy()
    expanded_slice_oi["start"] = np.clip(
        slice_oi["start"] - 1000, *(window - window[0])
    )
    expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 1000, *(window - window[0]))

    window_oi = (
        np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
    )

    gene_oi = expanded_slice_oi["gene_ix"]
    cluster_ix = expanded_slice_oi["cluster_ix"]
    cluster_info_oi = cluster_info.iloc[[cluster_ix]]

    plotdata_atac = (
        design.query("gene_ix == @gene_oi")
        .copy()
        .rename(columns={"active_latent": "cluster"})
        .set_index(["coord", "cluster"])
        .drop(columns=["batch", "gene_ix"])
    )
    plotdata_atac["prob"] = probs[gene_oi].flatten()
    plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

    plotdata_atac["prob"] = (
        plotdata_atac["prob"]
        - np.log(
            plotdata_atac.reset_index()
            .groupby(["cluster"])
            .apply(
                lambda x: np.trapz(
                    np.exp(x["prob"]),
                    x["coord"].astype(float) / (window[1] - window[0]),
                )
            )
        ).mean()
    )
    plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

    resolution = 0.0005
    panel_width = (window_oi[1] - window_oi[0]) * resolution

    # differential atac
    wrap_differential = chd.models.diff.plot.Differential(
        plotdata_atac,
        plotdata_atac_mean,
        cluster_info_oi,
        window_oi,
        panel_width,
        panel_height,
        # padding_height=padding_height,
        title=False,
    )
    grid[0, 0] = wrap_differential

    ax = wrap_differential.elements[0].ax

    # add labels
    gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
    cluster_label = cluster_info.query("dimension == @cluster_ix")["label"][0]
    position_label = str(int(slice_oi["start"] + slice_oi["length"] / 2) + window[0])

    chd.models.diff.plot.LabelSlice(ax, gene_label, cluster_label, slice_oi, window)

    # peaks
    promoter = promoters.iloc[expanded_slice_oi["gene_ix"]]

    grid[1, 0] = chdm.plotting.Peaks(
        promoter,
        peaks_folder,
        window=window_oi,
        width=panel_width,
        row_height=0.4,
        label_methods=slice_ix == len(slices_oi) - 1,
        label_rows=slice_ix == 0,
    )

    # add common/unique annotation
    region_position_chosen_oi = region_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]
    peak_position_chosen_oi = peak_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]

    chd.models.diff.plot.CommonUnique(
        ax,
        peak_position_chosen_oi,
        region_position_chosen_oi,
        expanded_slice_oi,
        window,
        methods_info,
    )
fig.plot()

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "background_examples")

# %% [markdown]
# ### Enrichment

# %% [markdown]
# TODO

# %%
motifscan_name = "cutoff_0001"
# motifscan_name = "onek1k_0.2"
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_cns"
# motifscan_name = "gtex"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)

# %%
# position_indices = np.repeat(np.arange(fragments.n_genes * (window[1] - window[0])), np.diff(motifscan.indptr))[(motifscan.indices == motif_ix_oi)]
# position_chosen = np.zeros(fragments.n_genes * (window[1] - window[0]), dtype = bool)
# position_chosen[position_indices] = True

# %%
# nuc = torch.where(onehot_promoters[:-1])[1]

# %%
# plt.plot(torch.bincount(nuc[1:] + (nuc[:-1]*4)))

# %%
peakscores_oi = peakscores

# cluster_oi = "leiden_0"
# cluster_oi = "Monocytes"
# cluster_oi = "B"
cluster_ixs = [cluster_info.loc[cluster_oi, "dimension"]]

cluster_ixs = cluster_info["dimension"]

# %%
peakscores_oi = peakscores_oi.assign(
    differential_positions_group=lambda x: (x["differential_positions"] * 5) // 1
)

# %%
groupenrichments = []
for differential_positions_group, peakscores_group in peakscores_oi.groupby(
    "differential_positions_group"
):
    print(differential_positions_group, peakscores_group.shape)
    motifscores_region = chd.models.diff.enrichment.enrich_windows(
        motifscan,
        peakscores_oi[["start", "end"]].values,
        peakscores_oi["gene_ix"].values,
        oi_slices=(
            peakscores_oi["cluster_ix"].isin(cluster_ixs)
            & (
                peakscores_oi["differential_positions_group"]
                == differential_positions_group
            )
        ).values,
        background_slices=(~peakscores_oi["cluster_ix"].isin([cluster_ix])).values,
        n_genes=fragments.n_genes,
        window=window,
        n_background=1,
    )
    motifscores_region["differential_positions_group"] = differential_positions_group

    groupenrichments.append(motifscores_region)
groupenrichments = pd.concat(groupenrichments).reset_index()
groupenrichments = groupenrichments.reset_index().set_index(
    ["differential_positions_group", "motif"]
)

# %%
groupenrichments["perc_gene_mean"] = [x.mean() for x in groupenrichments["perc_gene"]]
# typeenrichments["perc_gene_mean"] = [x[transcriptome.var["chr"] == "chr6"].mean() for x in typeenrichments["perc_gene"]]

# %%
groupenrichments.sort_values("logodds", ascending=False).head(15)

# %%
groupenrichments["significant"] = groupenrichments["qval"] < 0.05

# %%
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("SPI1")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("HXA13")][0]
motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("monoc")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("Rheumatoid arthritis")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("Whole Blood")][0]

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = groupenrichments.xs(motif_id, level="motif")
ax.bar(np.arange(len(plotdata)), plotdata["logodds"])
# ax.set_xticks(np.arange(len(plotdata)))
# ax.set_xticklabels(plotdata.index)
# chd.slicetypes.label_axis(ax, ax.xaxis)

# %%
sns.histplot(data=groupenrichments, x="logodds", hue="differential_positions_group")

# %% [markdown]
# ## Characteristics

# %% [markdown]
# ### Size

# %%
logbins = np.logspace(np.log10(1), np.log10(10000), 50)
sns.histplot(
    peakresult.get_slicescores()["length"],
    bins=logbins,
    color=methods_info["color"]["peak"],
)
sns.histplot(
    pureregionresult.get_slicescores()["length"],
    bins=logbins,
    color=methods_info["color"]["region"],
)
plt.xscale("log")

# %%
fig, ax = plt.subplots(figsize=(2, 2))
sns.ecdfplot(
    peakresult.get_slicescores()["length"], ax=ax, color=methods_info["color"]["peak"]
)
sns.ecdfplot(
    regionresult.get_slicescores()["length"],
    ax=ax,
    color=methods_info["color"]["region"],
)
sns.ecdfplot(
    pureregionresult.get_slicescores()["length"],
    ax=ax,
    color=methods_info["color"]["region"],
)
ax.set_xscale("log")
ax.set_xticks([10, 100, 1000, 10000])
ax.set_xlim(10, 10000)

# %%
pureregionresult.get_slicescores()[
    "length"
].median(), pureregionresult.get_slicescores()["length"].mean()

# %%
pureregionresult.get_slicescores()["length"].quantile([0.25, 0.75])

# %%
mid_bins = np.linspace(*window, 11)[:-1]

# %%
slicescores_region = pureregionresult.get_slicescores()
slicescores_region["mid_bin"] = np.digitize(slicescores_region["mid"], mid_bins)

# %%
logbins = np.logspace(np.log10(20), np.log10(1000), 50)

# %%
fig, axes = plt.subplots(
    len(mid_bins),
    1,
    figsize=(5, len(mid_bins) / 3),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0},
)
for i, (mid_bin, plotdata) in enumerate(slicescores_region.groupby("mid_bin")):
    ax = axes[i]
    ax.hist(plotdata["length"], density=True, bins=logbins, lw=0)
    ax.axvline(plotdata["length"].median(), color="red")
    ax.set_xscale("log")

# %%
fig, ax = plt.subplots(figsize=(3, 3))
sns.ecdfplot(data=slicescores_region, y="length", hue="mid_bin")
ax.set_yscale("log")
ax.set_ylim(1, 10000)
ax.legend([])

# %%
slicescores_region["loglength"] = np.log1p(slicescores_region["length"])

# %%
midbinscores_region = pd.DataFrame(
    {
        "position": mid_bins,
        "length": np.exp(slicescores_region.groupby("mid_bin")["loglength"].mean()),
        "length_std": np.exp(slicescores_region.groupby("mid_bin")["loglength"].std()),
    }
).set_index("position")

# %%
fig, ax = plt.subplots(figsize=(2, 2))
midbinscores_region["length"].plot(ax=ax)
fig, ax = plt.subplots(figsize=(2, 2))
midbinscores_region["length_std"].plot(ax=ax)

# %%
sns.histplot(
    plotdata["length"], binrange=(0, 500), bins=50
)  # .plot(kind = "hist", bins = 100)

# %% [markdown]
# ### Height

# %%
peakaverages = peakresult.get_sliceaverages(probs_interpolated).join(
    peakresult.get_slicescores().reset_index()
)
regionaverages = regionresult.get_sliceaverages(probs_interpolated).join(
    regionresult.get_slicescores().reset_index()
)
pureregionaverages = pureregionresult.get_sliceaverages(probs_interpolated).join(
    pureregionresult.get_slicescores().reset_index()
)

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(2.5, 1.5), sharey=True, gridspec_kw={"wspace": 0.1}
)

ax = axes[0]
sns.ecdfplot(
    np.exp(peakaverages["max_baseline"]), ax=ax, color=methods_info.loc["peak", "color"]
)
sns.ecdfplot(
    np.exp(regionaverages["max_baseline"]),
    ax=ax,
    color=methods_info.loc["region", "color"],
)
ax.set_xlabel("Maximum\naccessibility")
ax.set_xscale("log")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_xlim(np.exp(regionaverages["max_baseline"]).quantile([0.01, 0.99]))

ax = axes[1]
sns.ecdfplot(
    np.exp(peakaverages["average_baseline"]),
    ax=ax,
    color=methods_info.loc["peak", "color"],
)
sns.ecdfplot(
    np.exp(regionaverages["average_baseline"]),
    ax=ax,
    color=methods_info.loc["region", "color"],
)
ax.set_xscale("log")
ax.set_xlabel("Average\naccessibility")
ax.set_xlim(np.exp(regionaverages["average_baseline"]).quantile([0.01, 0.99]))

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "peak_vs_chd_height")

# %%
upstream_cutoff = -1000 - window[0]
downstream_cutoff = +1000 - window[0]


def classify_position_group(x, upstream_cutoff, downstream_cutoff):
    return np.array(["upstream", "downstream", "promoter"])[
        np.argmax(
            np.vstack(
                [
                    x.end < upstream_cutoff,
                    x.start > downstream_cutoff,
                    np.repeat(True, len(x)),
                ]
            ),
            0,
        )
    ]


position_group_info = pd.DataFrame(
    [
        ["upstream", "Upstream -10kb→-1kb"],
        ["promoter", "-1kb→+1kb"],
        ["downstream", "Downstream +1kb→+10kb"],
    ],
    columns=["position_group", "label"],
).set_index("position_group")

# %%
peakaverages["position_group"] = classify_position_group(
    peakaverages, upstream_cutoff, downstream_cutoff
)
regionaverages["position_group"] = classify_position_group(
    regionaverages, upstream_cutoff, downstream_cutoff
)

# %%
import textwrap

# %%
fig, axes = plt.subplots(1, 3, figsize=(1.5 * 3, 1.5), sharey=True, sharex=True)

score = "max_baseline"
xlim = (0.1, 100)
xlabel = "Maximum accessibility"
# score = "average";xlim = (0.1, 100); xlabel = "Average accessibility"
# score = "average_lfc";xlim = (1., 10); xlabel = "Average fold-change"
# score = "max_lfc";xlim = (1., 10); xlabel = "Max fold-change"
plotdata_all = {
    "peak": np.exp(peakaverages[score]),
    "region": np.exp(regionaverages[score]),
}
for ax, position_group in zip(axes, position_group_info.index):
    plotdata = {
        "peak": np.exp(
            peakaverages.loc[peakaverages["position_group"] == position_group][score]
        ),
        "region": np.exp(
            regionaverages.loc[regionaverages["position_group"] == position_group][
                score
            ]
        ),
    }
    plotdata_mean = {
        method: np.exp(np.log(values).median()) for method, values in plotdata.items()
    }
    print(plotdata_mean)
    sns.ecdfplot(plotdata["peak"], ax=ax, color=methods_info.loc["peak", "color"])
    sns.ecdfplot(plotdata["region"], ax=ax, color=methods_info.loc["region", "color"])

    sns.ecdfplot(
        plotdata_all["peak"],
        ax=ax,
        color=methods_info.loc["peak", "color"],
        zorder=-1,
        alpha=0.1,
    )
    sns.ecdfplot(
        plotdata_all["region"],
        ax=ax,
        color=methods_info.loc["region", "color"],
        zorder=-1,
        alpha=0.1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("")
    ax.set_xlim(*xlim)
    ax.set_title(
        "\n".join(
            textwrap.wrap(position_group_info.loc[position_group]["label"], width=10)
        )
    )
    ax.annotate(
        None,
        (plotdata_mean["peak"], 0.5),
        (plotdata_mean["region"], 0.5),
        xycoords="data",
        textcoords="data",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", ec="black", shrinkA=0, shrinkB=0),
    )
    ax.annotate(
        f'$\\times${plotdata_mean["peak"] / plotdata_mean["region"]:.2f}',
        (max(plotdata_mean["peak"], plotdata_mean["region"]), 0.5),
        va="center",
        ha="left",
    )
ax = axes[0]
ax.set_xlabel(xlabel)
ax.set_yticks([0, 0.5, 1])
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

if peakcaller == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "peak_vs_chd_height_positiongroups")
elif peakcaller == "rolling_50":
    manuscript.save_figure(fig, "4", "peak_vs_chd_height_positiongroups_rolling_50")

# %% [markdown]
# ## Gene-wise overlap

# %%
plotdata = pd.DataFrame(
    {
        "n_region": regionresult.get_slicescores().groupby("gene_ix")["length"].sum(),
        "n_peak": peakresult.get_slicescores().groupby("gene_ix")["length"].sum(),
    }
).fillna(0)
plotdata = plotdata.reindex(np.arange(transcriptome.var.shape[0]), fill_value=0)
plotdata.index = transcriptome.var.index[plotdata.index]

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata["n_region"], plotdata["n_peak"])
ax.set_xscale("symlog", linthresh=100)
ax.set_yscale("symlog", linthresh=100)

# %%
plotdata["diff"] = plotdata["n_region"] - plotdata["n_peak"]

# %%
plotdata.sort_values("diff").assign(
    symbol=lambda x: transcriptome.symbol(x.index).values
)

# %% [markdown]
# ## Classification

# %%
import chromatinhd.slicetypes

chromatinhd.slicetypes.types_info

# %% [markdown]
# ### Classify

# %%
# slicetopologies = pd.concat([
#     regionresult.get_slicetopologies(probs_interpolated),
#     regionresult.get_sliceaverages(probs_interpolated),
#     regionresult.get_slicescores()
# ], axis = 1)
slicetopologies = pd.concat(
    [
        pureregionresult.get_slicetopologies(probs_interpolated),
        pureregionresult.get_sliceaverages(probs_interpolated),
        pureregionresult.get_slicescores(),
    ],
    axis=1,
)
# slicetopologies = pd.concat([
#     peakresult.get_slicetopologies(probs_interpolated),
#     peakresult.get_sliceaverages(probs_interpolated),
#     peakresult.get_slicescores().reset_index()
# ], axis = 1)

# %%
promoters

# %%
sns.histplot(slicetopologies["prominence"], label="prominence")
sns.histplot(slicetopologies["dominance"], label="dominance")
sns.histplot(slicetopologies["shadow"], label="shadow")

# %%
slicetopologies["flank"] = slicetopologies["prominence"] <= 0.5
slicetopologies["hill"] = slicetopologies["dominance"] <= 0.5
slicetopologies["chain"] = (slicetopologies["length"] > 800) & (
    slicetopologies["n_subpeaks"] >= 2
)
slicetopologies["canyon"] = (slicetopologies["balance"] >= 0.2) | (
    slicetopologies["balances_raw"] < np.log(2)
)
slicetopologies["ridge"] = (slicetopologies["length"] > 800) & (
    slicetopologies["shadow"] > 0.5
)
slicetopologies["volcano"] = slicetopologies["max"] < np.log(1.0)

# %%
slicetopologies["type"] = "peak"
slicetopologies.loc[slicetopologies["volcano"], "type"] = "volcano"
slicetopologies.loc[slicetopologies["hill"], "type"] = "hill"
slicetopologies.loc[slicetopologies["canyon"], "type"] = "canyon"
slicetopologies.loc[slicetopologies["flank"], "type"] = "flank"
slicetopologies.loc[slicetopologies["chain"], "type"] = "chain"
slicetopologies.loc[slicetopologies["ridge"], "type"] = "ridge"
slicetopologies["type"] = pd.Categorical(
    slicetopologies["type"], categories=chd.slicetypes.types_info.index
)

# %%
slicetopologies["loglength"] = np.log(slicetopologies["length"])

# %% [markdown]
# ### Store for David

# %%
slicetopologies_mapped = slicetopologies.copy()
slicetopologies_mapped["gene"] = promoters.index[slicetopologies_mapped["gene_ix"]]
slicetopologies_mapped["cluster"] = cluster_info.index[
    slicetopologies_mapped["cluster_ix"]
]

# %%
slicetopologies_mapped["start"] = (
    promoters.loc[slicetopologies_mapped.gene, "tss"].values
    + (slicetopologies["start"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == 1).values
    - (slicetopologies["end"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == -1).values
)

slicetopologies_mapped["end"] = (
    promoters.loc[slicetopologies_mapped.gene, "tss"].values
    + (slicetopologies["end"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == 1).values
    - (slicetopologies["start"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == -1).values
)
slicetopologies_mapped["chr"] = promoters.loc[slicetopologies_mapped.gene, "chr"].values

# %%
scores_dir = prediction.path / "scoring" / "significant_up"
slicetopologies_mapped.to_csv(scores_dir / "slicetopologies.csv")

# %%
slicetopologies_mapped[["chr", "start", "end", "cluster", "type"]]

# %%
# from_file = scores_dir / "slicetopologies.csv"
# to_output = pathlib.Path("/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output")
# to_file = to_output / from_file.relative_to(chd.get_output())

# to_file.parent.mkdir(parents = True, exist_ok = True)

# import shutil
# shutil.copy(from_file, to_file)

# %% [markdown]
# ### 2D visualization of slices

# %%
features = ["prominence", "dominance", "loglength", "n_subpeaks", "shadow"]

# %%
X = slicetopologies[features].values
X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True))

# %%
import sklearn.decomposition

pca = sklearn.decomposition.PCA(2)
X_pca = pca.fit_transform(X)

# %%
fig, ax = plt.subplots()
ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=chd.slicetypes.types_info.loc[slicetopologies["type"], "color"],
    s=1,
)

# %%
import umap

umap = umap.UMAP()

# %%
X_umap = umap.fit_transform(X)

# %%
wrap = polyptich.grid.Wrap()
fig = polyptich.grid.Figure(wrap)

for feature in features:
    ax = wrap.add(polyptich.grid.Ax((3, 3)))
    ax = ax.ax
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=slicetopologies[feature], s=1)
    ax.set_title(feature)
fig.plot()

# %%
fig, ax = plt.subplots()
ax.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=chd.slicetypes.types_info.loc[slicetopologies["type"], "color"],
    s=1,
)

# %% [markdown]
# ### Examples

# %%
# window_oi = np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
# window_oi = np.array([500, 1000])

# %%
types_info = chd.slicetypes.types_info
types_info["ix"] = np.arange(len(types_info))

# %%
import polyptich.grid

main = polyptich.grid.Grid(len(types_info), 2, padding_width=0.1, padding_height=0.1)
fig = polyptich.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005

panel_height = 0.5

total_width_cutoff = 10

for slicetype, slicetype_info in types_info.iterrows():
    slicetype_ix = slicetype_info["ix"]
    ax_row_title = main[slicetype_ix, 0] = polyptich.grid.Ax((panel_height, panel_height))
    ax = ax_row_title.ax
    ax.axis("off")
    ax.text(
        -0.2,
        0.5,
        chd.slicetypes.types_info.loc[slicetype, "label"],
        va="center",
        ha="right",
    )
    chd.slicetypes.plot_type(ax, slicetype)

    wrap = main[slicetype_ix, 1] = polyptich.grid.Wrap(padding_width=0.1, ncol=10)

    width_so_far = 0

    for i in range(10):
        slice_oi = slicetopologies.query("(type == @slicetype) & (length > 100)").iloc[
            i
        ]

        expanded_slice_oi = slice_oi.copy()
        expanded_slice_oi["start"] = np.clip(
            slice_oi["start"] - 800, *(window - window[0])
        )
        expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 800, *(window - window[0]))

        window_oi = (
            np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
        )

        gene_oi = expanded_slice_oi["gene_ix"]
        cluster_info_oi = cluster_info.iloc[[expanded_slice_oi["cluster_ix"]]]

        plotdata_atac = (
            design.query("gene_ix == @gene_oi")
            .copy()
            .rename(columns={"active_latent": "cluster"})
            .set_index(["coord", "cluster"])
            .drop(columns=["batch", "gene_ix"])
        )
        plotdata_atac["prob"] = probs[gene_oi].flatten()
        plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

        plotdata_atac["prob"] = (
            plotdata_atac["prob"]
            - np.log(
                plotdata_atac.reset_index()
                .groupby(["cluster"])
                .apply(
                    lambda x: np.trapz(
                        np.exp(x["prob"]),
                        x["coord"].astype(float) / (window[1] - window[0]),
                    )
                )
            ).mean()
        )
        plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

        resolution = 0.0005
        panel_width = (window_oi[1] - window_oi[0]) * resolution

        width_so_far += panel_width

        if width_so_far >= total_width_cutoff:
            break

        # differential atac
        wrap_differential = chd.models.diff.plot.Differential(
            plotdata_atac,
            plotdata_atac_mean,
            cluster_info_oi,
            window_oi,
            panel_width,
            panel_height,
            padding_height=padding_height,
            title=False,
        )
        wrap.add(wrap_differential)

        ax = wrap_differential.elements[0].ax

        start = slice_oi["start"] + window[0]
        end = slice_oi["end"] + window[0]
        ax.axvspan(start, end, fc="#0003", lw=0)

        # gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
        gene_label = gene_oi
        cluster_label = cluster_info.query("dimension == @slice_oi.cluster_ix")[
            "label"
        ][0]
        position_label = str(
            int(slice_oi["start"] + slice_oi["length"] / 2) + window[0]
        )
        extra = str(slice_oi.name)
        text = ax.annotate(
            f"$\\it{{{gene_label}}}$ $\\bf{{{cluster_label}}}$ {extra}",
            (0, 1),
            (2, 2),
            va="bottom",
            ha="left",
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )

        trans = mpl.transforms.blended_transform_factory(
            y_transform=ax.transAxes, x_transform=ax.transData
        )
        text = ax.annotate(
            f"{start:+}",
            (start, 1),
            (-2, -2),
            va="top",
            ha="right",
            xycoords=trans,
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )
        text = ax.annotate(
            f"{end:+}",
            (end, 1),
            (2, -2),
            va="top",
            ha="left",
            xycoords=trans,
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )

fig.plot()

# %%
wrap = polyptich.grid.WrapAutobreak(padding_width=0.1, max_width=5, padding_height=0.25)

main = wrap
fig = polyptich.grid.Figure(main)

slicetype = "canyon"

slicetype_ix = slicetype_info["ix"]

for i in range(30):
    main.align()
    if main.nrow >= 7:
        break

    slice_oi = slicetopologies.query("(type == @slicetype) & (length > 100)").iloc[i]

    expanded_slice_oi = slice_oi.copy()
    expanded_slice_oi["start"] = np.clip(slice_oi["start"] - 800, *(window - window[0]))
    expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 800, *(window - window[0]))

    window_oi = (
        np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
    )

    gene_oi = expanded_slice_oi["gene_ix"]
    cluster_info_oi = cluster_info.iloc[[expanded_slice_oi["cluster_ix"]]]

    plotdata_atac = (
        design.query("gene_ix == @gene_oi")
        .copy()
        .rename(columns={"active_latent": "cluster"})
        .set_index(["coord", "cluster"])
        .drop(columns=["batch", "gene_ix"])
    )
    plotdata_atac["prob"] = probs[gene_oi].flatten()
    plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

    plotdata_atac["prob"] = (
        plotdata_atac["prob"]
        - np.log(
            plotdata_atac.reset_index()
            .groupby(["cluster"])
            .apply(
                lambda x: np.trapz(
                    np.exp(x["prob"]),
                    x["coord"].astype(float) / (window[1] - window[0]),
                )
            )
        ).mean()
    )
    plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

    resolution = 0.0005
    panel_width = (window_oi[1] - window_oi[0]) * resolution

    # differential atac
    wrap_differential = chd.models.diff.plot.Differential(
        plotdata_atac,
        plotdata_atac_mean,
        cluster_info_oi,
        window_oi,
        panel_width,
        panel_height,
        padding_height=padding_height,
        title=False,
    )
    wrap.add(wrap_differential)

    ax = wrap_differential.elements[0].ax

    start = slice_oi["start"] + window[0]
    end = slice_oi["end"] + window[0]
    ax.axvspan(start, end, fc="#0003", lw=0)

    gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
    cluster_label = cluster_info.query("dimension == @slice_oi.cluster_ix")["label"][0]
    position_label = str(int(slice_oi["start"] + slice_oi["length"] / 2) + window[0])
    extra = str(slice_oi.name)
    text = ax.annotate(
        f"$\\it{{{gene_label}}}$ $\\bf{{{cluster_label}}}$ {extra}",
        (0, 1),
        (2, 2),
        va="bottom",
        ha="left",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=6,
        color="#999",
        zorder=200,
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(foreground="white", linewidth=2),
            mpl.patheffects.Normal(),
        ]
    )

    trans = mpl.transforms.blended_transform_factory(
        y_transform=ax.transAxes, x_transform=ax.transData
    )
    text = ax.annotate(
        f"{start:+}",
        (start, 1),
        (-2, -2),
        va="top",
        ha="right",
        xycoords=trans,
        textcoords="offset points",
        fontsize=6,
        color="#999",
        zorder=200,
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(foreground="white", linewidth=2),
            mpl.patheffects.Normal(),
        ]
    )
    text = ax.annotate(
        f"{end:+}",
        (end, 1),
        (2, -2),
        va="top",
        ha="left",
        xycoords=trans,
        textcoords="offset points",
        fontsize=6,
        color="#999",
        zorder=200,
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(foreground="white", linewidth=2),
            mpl.patheffects.Normal(),
        ]
    )

fig.plot()

# %% [markdown]
# ### Positions

# %%
fig, axes = plt.subplots(
    chd.slicetypes.types_info.shape[0],
    1,
    figsize=(5, chd.slicetypes.types_info.shape[0] * 0.5),
    sharex=True,
    gridspec_kw={"hspace": 0},
)
nbins = 100
density_lim = 1 / ((window[1] - window[0]) / nbins) / 25
for ax, (type, plotdata) in zip(axes, slicetopologies.groupby("type")):
    color = chd.slicetypes.types_info.loc[type, "color"]
    sns.histplot(
        plotdata["mid"],
        bins=nbins,
        stat="density",
        label=type,
        lw=0,
        ax=ax,
        color=color,
    )
    # ax.text(0.02, 0.95, type, color = color, transform = ax.transAxes, va = "top", ha = "left")
    ax.set_yticks([])
    ax.set_xlim(*window)

    w, h = fig.transFigure.inverted().transform([[1, 1]])[0] * 20
    x, y = fig.transFigure.inverted().transform(ax.transAxes.transform([0.1, 0.9]))
    y -= h

    l = ax.yaxis.get_label()
    inset = chd.plot.replace_patch(ax, l, points=25, ha="right")
    l.set_visible(False)
    inset.axis("off")
    chd.slicetypes.plot_type(inset, type)
    ax.set_ylim(0, density_lim)
    ax.axvline(0, dashes=(2, 2), lw=1, color="#333")
axes[-1].set_xlabel("Distance from TSS")
axes[-1].set_xlabel("    ← upstream    TSS    downstream →")

# %% [markdown]
# ### Frequencies

# %%
fig, ax = plt.subplots(figsize=(1.5, 3))

plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby("type").size(),
        "n_positions": slicetopologies.groupby("type")["length"].sum(),
    }
)
plotdata["rel_n_regions"] = plotdata["n_regions"] / plotdata["n_regions"].sum()
plotdata["cum_n_regions"] = (
    np.cumsum(plotdata["rel_n_regions"]) - plotdata["rel_n_regions"]
)
plotdata["rel_n_positions"] = plotdata["n_positions"] / plotdata["n_positions"].sum()
plotdata["cum_n_positions"] = (
    np.cumsum(plotdata["rel_n_positions"]) - plotdata["rel_n_positions"]
)

ax.bar(
    0,
    plotdata["rel_n_regions"],
    bottom=plotdata["cum_n_regions"],
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
    lw=0,
)
ax.bar(
    1,
    plotdata["rel_n_positions"],
    bottom=plotdata["cum_n_positions"],
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
    lw=0,
)

texts = []
for type, plotdata_type in plotdata.iterrows():
    color = chd.slicetypes.types_info.loc[type, "color"]
    text = ax.text(
        -0,
        plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"] / 2,
        f"{plotdata_type['rel_n_regions']:.1%}",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground=color),
            mpl.patheffects.Normal(),
        ]
    )
    # texts.append(text)

    text = ax.text(
        1.0,
        plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"] / 2,
        f"{plotdata_type['rel_n_positions']:.1%}",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground=color),
            mpl.patheffects.Normal(),
        ]
    )
    # texts.append(text)
    # texts_left.append(ax.text(-0.5, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%}", ha = "right"))
    # texts_right.append(ax.text(2.5, plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%}", ha = "left"))
    # texts.append(ax.text(0, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%} {type}", ha = "center", va = "center"))
    # ax.text(1, plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%} {type}", ha = "center", va = "center")

ax.set_xticks([0, 1])
ax.set_xticklabels(["regions", "positions"])
ax.set_ylim(1, 0)
ax.set_xlim(-0.4, 1.4)
ax.set_yticks([0, 0.5, 1])
sns.despine(ax=ax)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.xaxis.tick_top()
# adjustText.adjust_text(texts, autoalign=False, only_move = {"text":"y"}, ha = "center", lim = 3000)
# adjustText.adjust_text(texts_right, autoalign=False, only_move = {"text":"y"}, ha = "left")

# %%
plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby(["type", "cluster_ix"]).size(),
        "n_positions": slicetopologies.groupby(["type", "cluster_ix"])["length"].sum(),
    }
)

# %%
plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby(["cluster_ix", "type"]).size(),
        "n_positions": slicetopologies.groupby(["cluster_ix", "type"])["length"].sum(),
    }
)
plotdata["rel_n_regions"] = (
    plotdata["n_regions"] / plotdata.groupby("cluster_ix")["n_regions"].sum()
)
plotdata["cum_n_regions"] = (
    plotdata.groupby("cluster_ix")["rel_n_regions"].cumsum() - plotdata["rel_n_regions"]
)
plotdata["rel_n_positions"] = (
    plotdata["n_positions"] / plotdata.groupby("cluster_ix")["n_positions"].sum()
)
plotdata["cum_n_positions"] = (
    plotdata.groupby("cluster_ix")["rel_n_positions"].cumsum()
    - plotdata["rel_n_positions"]
)

plotdata_grouped = plotdata.groupby("cluster_ix")

fig, axes = plt.subplots(
    ncols=len(plotdata_grouped), figsize=(1.5 * len(plotdata_grouped), 3), sharey=True
)

for ax, (cluster_ix, plotdata_cluster) in zip(axes, plotdata_grouped):
    ax.bar(
        0,
        plotdata_cluster["rel_n_regions"],
        bottom=plotdata_cluster["cum_n_regions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.bar(
        1,
        plotdata_cluster["rel_n_positions"],
        bottom=plotdata_cluster["cum_n_positions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.set_title(cluster_info.iloc[cluster_ix]["label"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["regions", "positions"])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, 1.4)

    ax.set_yticks([0, 0.5, 1])
    sns.despine(ax=ax)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# %%
slicetopologies["position_group"] = np.array(["upstream", "promoter", "downstream"])[
    np.argmax(
        np.vstack(
            [
                slicetopologies["mid"] < -1000,
                slicetopologies["mid"] < 1000,
                slicetopologies["mid"] <= window[1],
            ]
        ),
        0,
    )
]

position_group_info = pd.DataFrame(
    [
        ["upstream", "Upstream -10kb→-1kb"],
        ["promoter", "Promoter -1kb→+1kb"],
        ["downstream", "Downstream +1kb→+10kb"],
    ],
    columns=["position_group", "label"],
).set_index("position_group")
position_group_info.index = pd.Categorical(
    position_group_info.index, categories=position_group_info.index
)

# %%
grouper = "position_group"
group_info = position_group_info

# %%
import textwrap

# %%
slicetopologies[grouper] = pd.Categorical(
    slicetopologies[grouper], categories=group_info.index
)
plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby([grouper, "type"]).size(),
        "n_positions": slicetopologies.groupby([grouper, "type"])["length"].sum(),
    }
)
plotdata["rel_n_regions"] = (
    plotdata["n_regions"] / plotdata.groupby(grouper)["n_regions"].sum()
)
plotdata["cum_n_regions"] = (
    plotdata.groupby(grouper)["rel_n_regions"].cumsum() - plotdata["rel_n_regions"]
)
plotdata["rel_n_positions"] = (
    plotdata["n_positions"] / plotdata.groupby(grouper)["n_positions"].sum()
)
plotdata["cum_n_positions"] = (
    plotdata.groupby(grouper)["rel_n_positions"].cumsum() - plotdata["rel_n_positions"]
)

plotdata_grouped = plotdata.groupby(grouper)

fig, axes = plt.subplots(
    ncols=len(plotdata_grouped), figsize=(1.5 * len(plotdata_grouped), 3), sharey=True
)

for ax, (group_index, plotdata_cluster) in zip(axes, plotdata_grouped):
    ax.bar(
        0,
        plotdata_cluster["rel_n_regions"],
        bottom=plotdata_cluster["cum_n_regions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.bar(
        1,
        plotdata_cluster["rel_n_positions"],
        bottom=plotdata_cluster["cum_n_positions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.set_title(
        "\n".join(textwrap.wrap(group_info.loc[group_index]["label"], width=10))
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["regions", "positions"])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, 1.4)

    ax.set_yticks([0, 0.5, 1])
    sns.despine(ax=ax)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# %% [markdown]
# ------------

# %% [markdown]
# EQTL

# %%
slicetopologies["gene"] = promoters.iloc[slicetopologies["gene_ix"]].index

# %%
scores = pickle.load((chd.get_git_root() / "code" / "6-eqtl" / "scores.pkl").open("rb"))

# %%
scores["significant"] = scores["bf"] > np.log(10)

# %%
cluster_oi = "B"
cluster_ix = cluster_info.loc[cluster_oi, "dimension"]

# %%

# %%
significant = scores["significant"].unstack().groupby("variant").any()
slicetypes = (
    slicetopologies_oi.groupby(["gene", "type"])
    .size()
    .unstack()
    .reindex(significant.index, fill_value=0.0)
    > 1
)

# %%
contingencies = [
    (~significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
    (~significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
    (significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
    (significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
]
contingencies = np.stack(contingencies)

# %%
import fisher

# %%
slicetype_enrichments = []
for cluster_oi in cluster_info.index:
    cluster_ix = cluster_info.loc[cluster_oi, "dimension"]
    slicetopologies_oi = slicetopologies.query("cluster_ix == @cluster_ix")
    significant = scores["significant"].unstack().groupby("variant").any()
    slicetypes = (
        slicetopologies_oi.groupby(["gene", "type"])
        .size()
        .unstack()
        .reindex(significant.index, fill_value=0.0)
        > 1
    )
    contingencies = [
        (~significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
        (~significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
        (significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
        (significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
    ]
    contingencies = np.stack(contingencies)
    for slicetype, cont in zip(slicetypes.columns, contingencies.T):
        slicetype_enrichments.append(
            {
                "cont": cont,
                # "odds":(cont[0] * cont[3])/(cont[1] * cont[2]),
                "odds": scipy.stats.contingency.odds_ratio(
                    cont.reshape((2, 2))
                ).statistic,
                "p": fisher.pvalue(*cont).right_tail,
                "cluster": cluster_oi,
                "type": slicetype,
            }
        )

# %%
slicetype_enrichments = pd.DataFrame(slicetype_enrichments)

# %%
slicetype_enrichments.groupby("type")["odds"].median()

# %%
slicetype_enrichments = []
for cluster_oi in cluster_info.index:
    cluster_ix = cluster_info.loc[cluster_oi, "dimension"]
    slicetopologies_oi = slicetopologies.query("cluster_ix == @cluster_ix")
    significant = scores["significant"].unstack().groupby("variant").any()
    slicetypes = (
        slicetopologies_oi.groupby(["gene", "type"])
        .size()
        .unstack()
        .reindex(significant.index, fill_value=0.0)
        > 1
    )
    contingencies = [
        (~significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
        (~significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
        (significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
        (significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
    ]
    contingencies = np.stack(contingencies)
    for slicetype, cont in zip(slicetypes.columns, contingencies.T):
        slicetype_enrichments.append(
            {
                "cont": cont,
                "odds": scipy.stats.contingency.odds_ratio(
                    cont.reshape((2, 2))
                ).statistic,
                "p": fisher.pvalue(*cont).right_tail,
                "cluster": cluster_oi,
                "type": slicetype,
            }
        )

# %% [markdown]
# -----

# %% [markdown]
# ### Accessibility summary

# %%
import scipy.stats
import chromatinhd.slicetypes

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(2, 3), sharex=True)
sns.violinplot(
    data=slicetopologies,
    y="average_lfc",
    x="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax1,
    linewidth=0,
)
sns.violinplot(
    data=slicetopologies,
    y="max_lfc",
    x="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax2,
    linewidth=0,
)
sns.stripplot(
    data=slicetopologies,
    y="length",
    x="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax3,
    linewidth=0,
    s=1,
)

ax1.set_ylim(0, 2)
ax2.set_ylim(0, 3)
ax3.set_yscale("log")

chd.slicetypes.label_axis(ax2, ax2.xaxis)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 1.5), sharey=True)
sns.ecdfplot(
    data=slicetopologies,
    x="average_lfc",
    hue="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax1,
    legend=False,
)
ax1.set_xlim(0, 2)
sns.ecdfplot(
    data=slicetopologies,
    x="max_lfc",
    hue="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax2,
    legend=False,
)
ax2.set_xlim(0, 3)
sns.ecdfplot(
    data=slicetopologies,
    x="average",
    hue="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax3,
    legend=False,
)

# chd.slicetypes.label_axis(ax2, ax2.xaxis)

# %%
slice_oi = slicetopologies.query("type == 'ridge'").iloc[3]

expanded_slice_oi = slice_oi.copy()
expanded_slice_oi["start"] = np.clip(slice_oi["start"] - 1000, *(window - window[0]))
expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 1000, *(window - window[0]))

# %%
probs_interpolated_mean = probs_interpolated.mean(1)

# %%
# slicetopologies.query("gene_ix == 13")

# %%
transcriptome.var.iloc[expanded_slice_oi["gene_ix"]]

# %%
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(
    np.arange(expanded_slice_oi["start"], expanded_slice_oi["end"]),
    np.exp(
        probs_interpolated[
            expanded_slice_oi["gene_ix"],
            expanded_slice_oi["cluster_ix"],
            expanded_slice_oi["start"] : expanded_slice_oi["end"],
        ]
    ),
)
ax.plot(
    np.arange(expanded_slice_oi["start"], expanded_slice_oi["end"]),
    np.exp(
        probs_interpolated_mean[
            expanded_slice_oi["gene_ix"],
            expanded_slice_oi["start"] : expanded_slice_oi["end"],
        ]
    ),
)
ax.axvline(slice_oi["start"])
ax.axvline(slice_oi["end"])
ax.set_ylim(0)
# ax.axvline(expanded_slice_oi["start"]
# plt.plot(np.exp(probs_mean[slice_oi["gene_ix"], slice_oi["start"]:slice_oi["end"]]))

# %% [markdown]
# ### Overlap with peak/window

# %%
def position_chosen_type(slices, window, n_clusters, n_genes):
    position_chosen = np.zeros(
        n_clusters * n_genes * (window[1] - window[0]),
        dtype=bool,
    )
    for start, end, gene_ix, cluster_ix in zip(
        slices["start"], slices["end"], slices["gene_ix"], slices["cluster_ix"]
    ):
        position_chosen[
            (start + (window[1] - window[0]) * gene_ix * cluster_ix) : (
                end + (window[1] - window[0]) * gene_ix * cluster_ix
            )
        ] = True
    return position_chosen


# %%
peak_position_chosen = peakresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), window[1] - window[0])
)
ns = []
percs = []
for _, (start, end, gene_ix, cluster_ix) in slicetopologies[
    ["start", "end", "gene_ix", "cluster_ix"]
].iterrows():
    pos = peak_position_chosen[gene_ix, cluster_ix, start:end]
    ns.append(pos.sum())
    percs.append(pos.mean())

# %%
slicetopologies["n_overlap"] = ns
slicetopologies["perc_overlap"] = percs
slicetopologies["high_overlap"] = np.array(percs) > 0.5

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = (
    slicetopologies.groupby("type")["n_overlap"].sum()
    / slicetopologies.groupby("type")["length"].sum()
)
ax.bar(
    np.arange(len(plotdata)),
    plotdata,
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.axhline(
    (slicetopologies["n_overlap"].sum() / slicetopologies["length"].sum()),
    dashes=(2, 2),
    color="#333",
)
ax.set_title(f"Overlap between {peakcaller_label} differential peaks\n and ChromatinHD")
ax.set_ylim(0, 1)
ax.set_xticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.xaxis)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["perc_overlap"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
sns.ecdfplot(slicetopologies["perc_overlap"], color="grey")
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# %% [markdown]
# ### Enrichment

# %%
# motifscan_name = "cutoff_0001"
# motifscan_name = "onek1k_0.2"
motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_cns"
# motifscan_name = "gtex"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)

# %%
slicetopologies.query("cluster == 'B'").query(
    "gene_ix == @transcriptome.gene_ix('POU2AF1')"
)[["start", "end"]]


# %%
def enrich_groups_cluster_vs_clusters(
    regions, clustering_id, grouping_id, inclusive=False
):
    motifscores = []
    for cluster_id in regions[clustering_id].cat.categories:
        oi = regions[clustering_id] == cluster_id
        for group_id in regions[grouping_id].cat.categories:
            print(group_id)
            oi_slices = oi & (regions[grouping_id] == group_id)
            background_slices = (
                (~oi) if inclusive else (~oi & (regions[grouping_id] == group_id))
            )
            motifscores_group = chd.models.diff.enrichment.enrich_windows(
                motifscan,
                regions[["start", "end"]].values,
                regions["gene_ix"].values,
                oi_slices=oi_slices,
                background_slices=background_slices,
                n_genes=fragments.n_genes,
                window=window,
                n_background=None,
            )
            motifscores_group[grouping_id] = group_id
            motifscores_group[clustering_id] = cluster_id

            motifscores.append(motifscores_group)
    motifscores = pd.concat(motifscores).reset_index()
    motifscores = motifscores.reset_index().set_index(
        [clustering_id, grouping_id, "motif"]
    )
    return motifscores


# %%
slicetopologies["cluster"] = pd.Categorical(
    cluster_info.index[slicetopologies["cluster_ix"]], categories=cluster_info.index
)

# %%
typeenrichments = enrich_groups_cluster_vs_clusters(
    slicetopologies, "cluster", "type", inclusive=False
)
type_group_enrichments = chd.models.diff.enrichment.enrich_cluster_vs_clusters(
    motifscan, window, slicetopologies, "type", fragments.n_genes
)

# %%
typeenrichments["perc_gene_mean"] = [x.mean() for x in typeenrichments["perc_gene"]]
# typeenrichments["perc_gene_mean"] = [x[transcriptome.var["chr"] == "chr6"].mean() for x in typeenrichments["perc_gene"]]

# %%
typeenrichments.xs("canyon", level="type").query("`in` > 0").sort_values("odds")

# %%
typeenrichments.loc["CD4 T"].loc["canyon"].loc["Hodgkin's lymphoma"]

# %%
typeenrichments.loc["B"].loc["volcano"].sort_values("odds", ascending=False)

# %%
typeenrichments.loc["B"].loc["volcano"].sort_values("odds", ascending=False)

# %%
typeenrichments.query("qval < 0.05").sort_values("logodds", ascending=False)

# %%
typeenrichments.loc["CD4 T"].loc["canyon"].sort_values("logodds", ascending=False)

# %%
typeenrichments["significant"] = typeenrichments["qval"] < 0.05

# %%
celltype = "pDCs"

scores_flank = (
    typeenrichments.loc[celltype]
    .loc["canyon"]
    .query("qval < 0.05")
    .query("odds > 1")
    .sort_values("odds", ascending=False)
)
scores_flank["odds_peak"] = (
    typeenrichments.loc[celltype].loc[["chain", "peak"]].groupby("motif").mean()["odds"]
)
(scores_flank["odds"] / scores_flank["odds_peak"]).sort_values()

# %%
fig, ax = plt.subplots(figsize=(2, 2))
plotdata = typeenrichments.groupby("type")[["perc", "logodds", "perc_gene_mean"]].mean()
ax.barh(
    np.arange(len(plotdata)),
    plotdata["perc"],
    color=chd.slicetypes.types_info.loc[plotdata.index]["color"],
)
ax.set_yticks(np.arange(len(plotdata)))
ax.set_yticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.yaxis)

# %%
# motifs_oi = typeenrichments.groupby("motif")["significant"].any()
# typeenrichments.loc[(slice(None), motifs_oi.index[motifs_oi]), :].groupby("type")[["perc", "logodds"]].mean().style.bar()

# %%
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("PO2F2")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("TFE2")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("PAX5")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA4")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("SPI1")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF4")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("CEBPB")][0]; cluster = "Monocytes"
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("TCF7")][0]; cluster = "CD4 T"
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("STAT3")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("NFA")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("bin")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("monoc")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("ZBT14")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("INSM1")][0]
motif_id = motifscan.motifs.index[
    motifscan.motifs.index.str.contains("Rheumatoid arthritis")
][0]

# %%
motifs = motifscan.motifs

# %%
# motifclustermapping = pd.DataFrame([
#     [motifs.loc[motifs.index.str.contains("SPI1")].index[0], ["Monocytes", "cDCs"]],
#     [motifs.loc[motifs.index.str.contains("CEBPB")].index[0], ["Monocytes", "cDCs"]],
#     [motifs.loc[motifs.index.str.contains("PEBB")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("IRF8")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("IRF4")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("PAX5")].index[0], ["Lymphoma"]],
#     [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["B", "pDCs"]], # TCF3
#     [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],
#     [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["B"]],
#     [motifs.loc[motifs.index.str.contains("NFKB2")].index[0], ["B"]],
#     [motifs.loc[motifs.index.str.contains("RUNX2")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("RUNX1")].index[0], ["T"]],
#     [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["T"]],
# ], columns = ["motif", "clusters"]).set_index("motif")
# motifclustermapping = motifclustermapping.explode("clusters").rename(columns = {"clusters":"cluster"}).reset_index()[["cluster", "motif"]]

# motifclustermapping = pd.DataFrame([
#     [motifs.loc[motifs.index.str.contains("TCF7")].index[0], ["CD4 T", "CD8 T"]],
#     [motifs.loc[motifs.index.str.contains("IRF8")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("IRF4")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("CEBPB")].index[0], ["Monocytes", "cDCs"]],
#     [motifs.loc[motifs.index.str.contains("GATA4")].index[0], ["CD4 T"]],
#     [motifs.loc[motifs.index.str.contains("HNF6_HUMAN.H11MO.0.B")].index[0], ["CD4 T"]],
#     [motifs.loc[motifs.index.str.contains("RARA")].index[0], ["MAIT"]],
#     [motifs.loc[motifs.index.str.contains("PEBB")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("RUNX2")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("RUNX1")].index[0], ["CD8 T"]],
#     [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["CD8 T"]],
#     [motifs.loc[motifs.index.str.contains("TBX21_HUMAN.H11MO.0.A")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("SPI1")].index[0], ["Monocytes", "B", 'cDCs']],
#     [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["B"]],
#     [motifs.loc[motifs.index.str.contains("NFKB2")].index[0], ["B"]],
#     [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["B", "pDCs"]], # TCF3
#     [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],
#     [motifs.loc[motifs.index.str.contains("FOS")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("RORA")].index[0], ["MAIT"]],
# ], columns = ["motif", "clusters"]).set_index("motif")
# motifclustermapping = motifclustermapping.explode("clusters").rename(columns = {"clusters":"cluster"}).reset_index()[["cluster", "motif"]]

# %%
typeenrichments.query("qval < 0.1").sort_values("odds", ascending=False)["perc_gene"]

# %%
pd.DataFrame(
    {
        "gene": transcriptome.var.index,
        "perc": typeenrichments.query("qval < 0.1").sort_values(
            "odds", ascending=False
        )["perc_gene"][0],
    }
).sort_values("perc")

# %%
gene_id = "ENSG00000090104"
transcriptome.var["ix"] = np.arange(len(transcriptome.var))
gene_ix = transcriptome.var.loc[gene_id, "ix"]

# %%
motifscan.motifs["ix"] = np.arange(len(motifscan.motifs))
motif_id = "Multiple sclerosis"
motif_ix = motifscan.motifs.loc[motif_id, "ix"]

# %%
position_range = (
    gene_ix * (window[1] - window[0]),
    ((gene_ix + 1) * (window[1] - window[0])),
)

# %%
motifscan.indices[
    motifscan.indptr[position_range[0]] : motifscan.indptr[position_range[1]]
]

# %%
motif_ix

# %%
motifscan.indptr

# %%

# %%
transcriptome.var.iloc[292]

# %%
transcriptome.var.iloc[768]

# %%
# typeenrichments.loc["CD8 T"].groupby("motif").mean().sort_values("logodds", ascending = False).query("significant > 0").head(20)

# %%
sc.pl.umap(transcriptome.adata, color="cluster")

# %%
cors = pd.DataFrame(
    np.corrcoef(typeenrichments["logodds"].unstack().T),
    index=motifscan.motifs.index,
    columns=motifscan.motifs.index,
)

# %%
cors["SNAI2_HUMAN.H11MO.0.A"].sort_values(ascending=False).head(10)

# %%
# sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["NFIL3", "CEBPB", "CEBPD", "NFIL3"])) # also includes ATF4
sc.pl.umap(transcriptome.adata, color=transcriptome.gene_id(["TBX21"]))

# %%
typeenrichments_oi = motifclustermapping.set_index(["cluster", "motif"]).join(
    typeenrichments
)
typeenrichments_oi["logodds"].unstack(level="type").style.bar(axis=1, vmin=0)

# %%
plotdata

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata_mean = typeenrichments_oi.groupby("type")[["logodds"]].mean()
plotdata_mean = plotdata_mean.sort_values("logodds")
plotdata_mean["y"] = np.arange(len(plotdata_mean))
plotdata = typeenrichments_oi[["logodds"]].reset_index()

# scatter of individual
ax.scatter(
    np.exp(plotdata_mean["logodds"]),
    plotdata_mean["y"],
    color=chd.slicetypes.types_info.loc[plotdata_mean.index]["color"],
    marker="o",
)
# lines between individual
for motif, plotdata_motif in plotdata.groupby(["motif", "cluster"]):
    plotdata_motif["y"] = plotdata_mean["y"][plotdata_motif.type].values
    plotdata_motif = plotdata_motif.sort_values("y")
    ax.plot(
        np.exp(plotdata_motif["logodds"]), plotdata_motif["y"], color="#3331", zorder=-1
    )
# scatter of mean
ax.scatter(
    x=np.exp(plotdata["logodds"]).values,
    y=plotdata_mean["y"][plotdata["type"]].values,
    color=chd.slicetypes.types_info.loc[plotdata["type"]]["color"].values,
    s=1,
)
ax.axvline(1, dashes=(2, 2), color="grey")
ax.set_yticks(np.arange(len(plotdata_mean)))
ax.set_yticklabels(plotdata_mean.index)
chd.slicetypes.label_axis(ax, ax.yaxis)
ax.set_xscale("log")
ax.set_xlabel("Motif odds-ratio")
ax.set_xticks([1, 3 / 2, 2, 3])
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
sns.despine(ax=ax)

# %%
typeenrichments["logodds"].unstack().loc["B"].loc["ridge"].sort_values()

# %%
fig, ax = plt.subplots()
ax.scatter(
    typeenrichments["logodds"].unstack().loc["Monocytes"].loc["ridge"],
    typeenrichments["logodds"].unstack().loc["Monocytes"].loc["volcano"],
)

# %%
sns.heatmap(np.corrcoef(typeenrichments["logodds"].unstack().loc["Monocytes"]))

# %%
plt.hist(
    np.diag(
        typeenrichment_cors.xs("volcano", level="type").xs("peak", level="type", axis=1)
    )
)
plt.hist(
    np.diag(
        typeenrichment_cors.xs("flank", level="type").xs("peak", level="type", axis=1)
    )
)

# %%
typeenrichment_cors = pd.DataFrame(
    np.corrcoef(typeenrichments["logodds"].unstack()),
    typeenrichments["logodds"].unstack().index,
    typeenrichments["logodds"].unstack().index,
)
sns.heatmap(typeenrichment_cors)

# %% [markdown]
# ### Characteristics

# %% [markdown]
# #### Conservation

# %%
import chromatinhd.conservation

folder_cons = chd.get_output() / "data" / "cons" / "hs" / "gerp"
conservation = chd.conservation.Conservation(folder_cons / "hg38.phastCons100way.bw")

# %%
promoters["gene_ix"] = np.arange(len(promoters))

# %%
slicelocations = pureregionresult.get_slicelocations(promoters)
# slicelocations = pureregionresult.get_randomslicelocations(promoters)

# %%
conservations = []
for slice in slicelocations.itertuples():
    cons = conservation.get_values(slice.chr, slice.start_genome, slice.end_genome)
    cons[np.isnan(cons)] = 0.0
    conservations.append(cons.mean())
conservations = np.array(conservations)

# %%
slicetopologies["conservation"] = conservations
slicetopologies.groupby("type")["conservation"].mean().plot()

# %%
n_random = 1

slicelocations_random = pureregionresult.get_randomslicelocations(
    promoters, n_random=n_random
)

conservations = []
for slice in slicelocations_random.itertuples():
    cons = conservation.get_values(slice.chr, slice.start_genome, slice.end_genome)
    cons[np.isnan(cons)] = 0.0
    conservations.append(cons.mean())
conservations = np.array(conservations)

# %%
slicetopologies_random = slicetopologies.iloc[
    np.repeat(np.arange(len(slicetopologies)), n_random)
].copy()
slicetopologies_random["conservation"] = conservations
slicetopologies_random.groupby("type")["conservation"].mean().plot()

# %%
(
    slicetopologies.groupby("type")["conservation"].mean()
    / slicetopologies_random.groupby("type")["conservation"].mean()
).plot()

# %%
slicetopologies["conservation_random"] = (
    slicetopologies_random["conservation"]
    .values.reshape((len(slicetopologies), n_random))
    .mean(1)
)

# %%
slicetopologies["conservation_norm"] = np.log(
    slicetopologies["conservation"] / slicetopologies["conservation_random"]
)

# %%
slicetopologies.groupby("type")["conservation_norm"].mean()

# %%
sns.boxplot(x="type", y="conservation_norm", data=slicetopologies)

# %% [markdown]
# #### QTLs

# %%
motifscan_name = "gwas_immune"
# motifscan_name = "gtex_immune"
# motifscan_name = "onek1k_0.2"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)

# %%
slicetopologies["cluster_type"] = pd.Categorical(
    slicetopologies["cluster_ix"].astype(str)
    + "_"
    + slicetopologies["type"].astype(str)
)

# %%
qtl_enrichments = chd.models.diff.enrichment.enrich_cluster_vs_all(
    motifscan,
    window,
    slicetopologies,
    "cluster_type",
    fragments.n_genes,
    fragments.var.index,
)
# qtl_enrichments = chd.models.diff.enrichment.enrich_cluster_vs_clusters(motifscan, window, slicetopologies, "cluster_type", fragments.n_genes)

# %%
qtl_enrichments["type"] = (
    qtl_enrichments.index.get_level_values("cluster_type").str.split("_").str[1]
)
qtl_enrichments["cluster"] = cluster_info.index[
    qtl_enrichments.index.get_level_values("cluster_type")
    .str.split("_")
    .str[0]
    .astype(int)
]

# %%
qtl_enrichments.groupby("type")["n_found"].sum()

# %%
qtl_enrichments

# %%
qtl_enrichments.loc[
    qtl_enrichments.index.get_level_values("cluster_type") == "0_canyon"
].query("n_found > 0")

# %%
motif_id = "Multiple sclerosis"
cluster = "B"
# motif_id = "Hodgkin's lymphoma"; cluster = "CD4 T"
motif_id = "Crohn's disease"
cluster = "cDCs"

# %%
qtl_enrichments["label"] = transcriptome.symbol(
    qtl_enrichments.index.get_level_values("gene")
).values

# %%
qtl_enrichments.query("`disease/trait` == @motif_id").query(
    "cluster == @cluster"
).query("type == 'canyon'").sort_values("n_found")

# %%
qtl_enrichments.query("cluster_type == '0_ridge'").query("n_found > 0")

# %%
# qtl_enrichments.query("qval < 0.1").sort_values("odds", ascending = False)["perc_gene"]

# %%
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
).plot(kind="bar")

# %%
(
    qtl_enrichments.groupby("type")["n"].sum()
    / (fragments.n_genes * (window[1] - window[0]))
)

# %%
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
) / (
    qtl_enrichments.groupby("type")["n"].sum()
    / (fragments.n_genes * (window[1] - window[0]))
)

# %%
fig, ax = plt.subplots()
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
    * qtl_enrichments.groupby("type")["n"].sum()
).plot()
ax.set_ylim(0)

# %%
fig, ax = plt.subplots()
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
).plot()
ax.set_ylim(0)

# %% [markdown]
# ### Get interesting genes

# %%
slicetopologies["gene_symbol"] = transcriptome.var.iloc[slicetopologies["gene_ix"]][
    "symbol"
].values
slicetopologies["cluster"] = cluster_info.index[slicetopologies["cluster_ix"]]

# %%
slicetopologies.query("type == 'canyon'").query("cluster == 'B'").query("length > 100")[
    ["expression_lfc", "cluster", "start", "length", "gene_symbol"]
].head(20)

# %%
slicetopologies.query("type == 'canyon'").query("expression_lfc < log(2)").sort_values(
    "average_lfc"
).query("cluster == 'Monocytes'").query("length > 100")[
    ["expression_lfc", "cluster", "start", "length", "gene_symbol"]
].head(
    10
)

# %%
slicetopologies.query("type == 'canyon'").query("expression_lfc <= log(2)")[
    ["cluster_ix", "gene_ix"]
]

# %% [markdown]
# ## Expression

# %%
y_cells = np.array(transcriptome.X.to_scipy_csr().todense())
y_clusters = (
    pd.DataFrame(y_cells, index=pd.from_dummies(latent)).groupby(level=0).mean().values
)

# %%
transcriptome.var["ix"] = np.arange(len(transcriptome.var))

# %%
sc.tl.rank_genes_groups(transcriptome.adata, "cluster")

cluster_diffexp = []
for cluster in cluster_info.index:
    diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, cluster)
    diffexp["gene"] = diffexp["names"]
    diffexp = diffexp.set_index("gene")
    diffexp["lfc"] = diffexp["logfoldchanges"]
    diffexp["cluster"] = cluster
    diffexp["gene_ix"] = transcriptome.var.loc[diffexp.index, "ix"]
    diffexp["cluster_ix"] = cluster_info.loc[cluster, "dimension"]
    cluster_diffexp.append(diffexp)
cluster_diffexp = pd.concat(cluster_diffexp)
cluster_diffexp = cluster_diffexp.reset_index().set_index(["cluster", "gene"])

# %%
lfc_clusters = (
    cluster_diffexp.set_index(["cluster_ix", "gene_ix"])["lfc"].unstack().values
)

# %% [markdown]
# ### Correlation with differential expression and slice type

# %%
fimp_full = np.zeros((fragments.n_genes, (window[1] - window[0])))

# %%
mixture_interpolated = probs_interpolated - probs_interpolated.mean(-1, keepdims=True)

# %%
tasks = []
for gene_ix in tqdm.tqdm(range(0, 5000)):
    # X = basepair_ranking[gene_ix]
    X = probs_interpolated[gene_ix]
    # X = mixture_interpolated[gene_ix]
    X[X == -np.inf] = 0
    y = y_clusters[:, gene_ix]
    tasks.append((X, y))


# %%
def calculate_importance(task):
    X, y = task
    rf = sklearn.ensemble.RandomForestRegressor(
        n_estimators=1000, max_depth=1, max_features=0.05
    )
    rf.fit(X, y)
    # plt.plot(rf.feature_importances_)
    fimp = scipy.ndimage.gaussian_filter(rf.feature_importances_, 20.0)
    return fimp


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    cor = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
    cor[np.isnan(cor)] = 0
    return cor


def calculate_importance(task):
    X, y = task
    return corr2_coeff(X.T, y[None, :])[:, 0]


# %% [markdown]
# $${\displaystyle r={\frac {\sum _{i}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})/(n-1)}{s(x)s(y)}}}$$

# %%
results = list(tqdm.tqdm(map(calculate_importance, tasks)))
results = np.vstack(list(results))

# pool = ProcessPoolExecutor(10)
# results = pool.map(calculate_importance, tasks)
# results = np.vstack(list(results))

# %%
fimp_full[range(results.shape[0])] = results

# %%
gene_ix = transcriptome.gene_ix("PTGDS")

# %%
fig, ax = plt.subplots()
plt.plot(fimp_full[gene_ix])

transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
for _, slice in slicetopologies.query("gene_ix == @gene_ix").iterrows():
    ax.plot(
        [slice["start"], slice["end"]],
        [0.99, 0.99],
        color=chd.slicetypes.types_info.loc[slice["type"], "color"],
        transform=transform,
        lw=3,
        clip_on=False,
        solid_capstyle="butt",
    )
sns.despine()

# %%
slice_fimps = []
slicetype_fimps = {
    slicetype: [] for slicetype in slicetopologies["type"].cat.categories
}
for ix, slice in slicetopologies.iterrows():
    fimp_slice = fimp_full[slice["gene_ix"], slice["start"] : slice["end"]]
    slicetype_fimps[slice.type].extend(fimp_slice)
    slice_fimps.append(fimp_slice.max())
slicetopologies["fimp"] = slice_fimps

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Positions")
for slicetype, fimp in slicetype_fimps.items():
    fimp = np.array(fimp)
    print(slicetype, (fimp > 0.2).mean(), (fimp < -0.2).mean())
    sns.ecdfplot(fimp, color=chd.slicetypes.types_info.loc[slicetype, "color"])
ax.set_xlabel("Correlation")

# %%
expressionscores = []
cor_cutoff = 0.2
for slicetype, fimp in slicetype_fimps.items():
    fimp = np.array(fimp)
    expressionscores.append(
        {
            "slicetype": slicetype,
            "up": (fimp > cor_cutoff).mean(),
            "down": (fimp < -cor_cutoff).mean(),
        }
    )
expressionscores = pd.DataFrame(expressionscores)
expressionscores["same"] = 1 - expressionscores["up"] - expressionscores["down"]
expressionscores.plot(kind="bar")

# %%
slicetypes_info = chd.slicetypes.types_info

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Regions")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    print(slicetype, ((np.abs(slicetopologies_type["fimp"]) > 0.25).mean()))
    sns.ecdfplot(
        slicetopologies_type["fimp"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
ax.set_xlabel("Mean absolute correlation between expression and accessibility")
ax.set_xlim(0, 1)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Regions outside promoter (end is within -10kb → 0)")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    print(slicetype, (np.abs(slicetopologies_type["fimp"] > 0.25).mean()))
    sns.ecdfplot(
        slicetopologies_type.query("(end + @window[0]) < 0")["fimp"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
ax.set_xlabel("Mean absolute correlation between expression and accessibility")
ax.set_xlim(0, 1)

# %%
slicetopologies.groupby("type")["fimp"].mean().plot()

# %%
yrank_clusters = scipy.stats.rankdata(y_clusters, axis=0, method="min")

# %%
slice_ranks = []
for ix, slice in slicetopologies.iterrows():
    expression_rank = yrank_clusters[slice["cluster_ix"], slice["gene_ix"]]
    slice_ranks.append(expression_rank)
slicetopologies["expression_rank"] = slice_ranks

# %%
fig, ax = plt.subplots()
ax.set_title("Expression rank")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["expression_rank"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )

# %% [markdown]
# ### Correlation with presence of slice type and expression effect

# %%
# slicetopologies["expression_lfc"] = y_clusters[slicetopologies["cluster_ix"], slicetopologies["gene_ix"]] - y_clusters.mean(0)[slicetopologies["gene_ix"]]
# slicetopologies["expression_lfc"] = (lfc_clusters)[slicetopologies["cluster_ix"], slicetopologies["gene_ix"]]
slicetopologies["expression_lfc"] = (lfc_clusters)[
    slicetopologies["cluster_ix"], slicetopologies["gene_ix"]
]
# slicetopologies["expression_lfc"] = (lfc_clusters - lfc_clusters.mean(0, keepdims = True))[slicetopologies["cluster_ix"], slicetopologies["gene_ix"]]
# slicetopologies["expression_lfc"] = cluster_diffexp.loc[cluster_info.index[slicetopologies["cluster_ix"]], transcriptome.var.index[slicetopologies["gene_ix"]], :]["lfc"].values

# %%
slicetype_scores = pd.DataFrame(
    {
        "mean_expression_lfc": np.exp(
            slicetopologies.groupby("type")["expression_lfc"].mean()
        ),
        "median_expression_lfc": np.exp(
            slicetopologies.groupby("type")["expression_lfc"].median()
        ),
    }
)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
# ax.set_title("Regions outside promoter (end is within -10kb → 0)")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        np.exp(slicetopologies_type["expression_lfc"]),
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
ax.set_xlabel("Gene differential expression")
ax.set_xscale("log")
ax.set_xlim(1 / 10, 10)

# %%
expression_changes_info = pd.DataFrame(
    {
        "upper_limit": np.array([1 / 8, 1 / 4, 1 / 2, 2, 4, 8, np.inf]),
        "label": ["⅛", "¼", "½", r"1", "2", "4", "8"],
    }
)
expression_changes_info.index = expression_changes_info["label"]
# expression_changes_info["label"] = expression_changes_info.index
expression_changes_info["ix"] = np.linspace(0, 1, len(expression_changes_info) + 2)[
    1:-1
]

# %%
slicetopologies["expression_change"] = expression_changes_info.index[
    np.digitize(
        slicetopologies["expression_lfc"],
        np.log(expression_changes_info["upper_limit"]),
    )
]

# %%
slicetype_expression_changes = slicetopologies.groupby(
    ["type", "expression_change"]
).size()
slicetype_expression_changes = (
    slicetype_expression_changes / slicetype_expression_changes.groupby("type").sum()
)

# %%
expression_changes_info["color"] = [
    c for c in mpl.cm.RdBu_r(np.linspace(0, 1, len(expression_changes_info)))
]

# %%
plotdata = slicetype_expression_changes.reset_index(name="perc")
plotdata["expression_change"] = pd.Categorical(
    plotdata["expression_change"], categories=expression_changes_info.index
)

plotdata = plotdata.sort_values("expression_change")
plotdata["cumperc"] = plotdata.groupby("type")["perc"].cumsum()

# %%
slicetype_scores = slicetype_scores.sort_values(
    "median_expression_lfc", ascending=False
)
slicetype_scores["ix"] = np.arange(len(slicetype_scores))

# %%
plotdata["ix"] = slicetype_scores.loc[plotdata["type"], "ix"].values

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.barh(
    plotdata["ix"],
    plotdata["perc"],
    left=plotdata["cumperc"] - plotdata["perc"],
    color=expression_changes_info.loc[plotdata["expression_change"], "color"],
    lw=0,
    height=0.9
    # color = chd.slicetypes.types_info.loc[plotdata["type"], "color"]
)
ax.set_yticks(np.arange(len(slicetype_scores)))
# ax.set_title(f"Overlap between peaks ({peaks_name}) and ChromatinHD")
ax.set_xlim(0, 1)
ax.set_ylim(-0.4, len(slicetype_scores) - 0.6)
ax.set_yticklabels(slicetype_scores.index)
chd.slicetypes.label_axis(ax, ax.yaxis)
sns.despine(ax=ax)

plotdata_oi = plotdata.loc[plotdata["ix"] == plotdata["ix"].max()].set_index(
    "expression_change"
)
texts = []
for expression_change_oi, expression_change_info in expression_changes_info.iterrows():
    texts.append(
        ax.text(
            plotdata_oi.loc[expression_change_oi, "cumperc"]
            - plotdata_oi.loc[expression_change_oi, "perc"] / 2,
            1.0,
            expression_change_info["label"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
    )
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = slicetype_scores
ax.bar(
    np.arange(len(plotdata)),
    np.log(plotdata["mean_expression_lfc"]),
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.axhline(0, dashes=(2, 2), color="#333")
ax.set_ylabel("Average fold-change")
ax.set_ylim([np.log(1 / 8), np.log(8)])
ax.set_xticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.xaxis)

# %%
plotdata = slicetopologies.query("type == 'canyon'")
fig, ax = plt.subplots(figsize=(3, 3))
# ax.scatter((plotdata["expression_lfc"]), plotdata["balances_raw"], s = 1)
# sns.regplot(x = plotdata["expression_lfc"], y = plotdata["balances_raw"], marker = "None")

ax.scatter((plotdata["expression_lfc"]), np.log(plotdata["balance"]))
sns.regplot(x=plotdata["expression_lfc"], y=np.log(plotdata["balance"]))

# %% [markdown]
# ### Association with high expression and slice type

# %%
yrank = scipy.stats.rankdata(y_clusters.max(0), method="min")

# %%
slice_ranks = []
for ix, slice in slicetopologies.iterrows():
    expression_rank = yrank[slice["gene_ix"]]
    slice_ranks.append(expression_rank)
slicetopologies["expression_rank"] = slice_ranks

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Expression rank")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["expression_rank"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Expression rank")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type.query("(end + @window[0]) > -1000")["expression_rank"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )

# %%
slicetopologies.groupby("type")["expression_rank"].mean().plot()

# %%
slice_fimps = []
for ix, slice in slicetopologies.iterrows():
    slice_fimps.append(
        np.abs(fimp_full[slice["gene_ix"], slice["start"] : slice["end"]]).max()
    )
slicetopologies["fimp"] = slice_fimps

# %%
slicetopologies.groupby("type")["fimp"].mean().plot()

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = (
    slicetopologies.groupby("type")["n_overlap"].sum()
    / slicetopologies.groupby("type")["length"].sum()
)
ax.bar(
    np.arange(len(plotdata)),
    plotdata,
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.axhline(
    (slicetopologies["n_overlap"].sum() / slicetopologies["length"].sum()),
    dashes=(2, 2),
    color="#333",
)
ax.set_title(f"Overlap between peaks ({peaks_name}) and ChromatinHD")
ax.set_ylim(0, 1)
ax.set_xticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.xaxis)

# %% [markdown]
# ## Conservation

# %%
import chromatinhd.conservation

folder_cons = chd.get_output() / "data" / "cons" / "hs" / "gerp"
conservation = chd.conservation.Conservation(folder_cons / "hg38.phastCons100way.bw")

# %%
promoters["gene_ix"] = np.arange(len(promoters))

# %%
# sliceinfo = regionresult.get_slicelocations(promoters)
sliceinfo = peakresult.get_slicelocations(promoters)
# sliceinfo = pureregionresult.get_slicelocations(promoters)

# %%
conservations = []
for slice in sliceinfo.itertuples():
    conservations.extend(
        conservation.get_values(slice.chr, slice.start_genome, slice.end_genome)
    )
conservations = np.array(conservations)

# %%
conservations[np.isnan(conservations)] = 0.0

# %%
np.mean(conservations)
