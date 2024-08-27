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
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import xarray as xr

import seaborn as sns

sns.set_style("ticks")

import scanpy as sc

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_name = "pbmc10k"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")
splitter = "5x1"
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)

model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / splitter / "magic" / "v33"
# model_folder = chd.get_output() / "pred" / dataset_name / "500k500k" / splitter / "magic" / "v34"
# models = chd.models.pred.model.better.Models(model_folder)

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    model_folder / "scoring" / "regionmultiwindow",
)

# %%
design = regionmultiwindow.design.query("window_size == 200.")
windows_oi = design.index

# %%
genedeltacor = regionmultiwindow.scores["deltacor"].sel_xr().sel(phase = "test", window = windows_oi).mean("fold")

# %%
genelost = regionmultiwindow.scores["lost"].sel_xr().sel(phase = "test", window = windows_oi).mean("fold")

# %%
geneeffect = regionmultiwindow.scores["effect"].sel_xr().sel(phase = "test", window = windows_oi).mean("fold")

# %%
deltacor_window_cutoff = -0.001
deltacor_mask = genedeltacor < deltacor_window_cutoff

# %%
metric = "deltacor"
# metric = "deltamse"

overallmetric = "cor_diff"
# overallmetric = "mse_diff"

# %%
deltacor = genedeltacor.mean("gene")

# %%
performance = chd.models.pred.interpret.Performance(model_folder / "scoring" / "performance")

# %%
genescores_overall = performance.scores.sel_xr().mean("fold").sel(phase = "test").to_pandas()

# %%
genedeltar2 = (genescores_overall["cor"].values[:, None] + genedeltacor)**2 - (genescores_overall["cor"].values[:, None] ** 2)

# %% [markdown]
# ### Global view

# %%
deltacor.to_pandas().plot()

# %% [markdown]
# ## Comparing peaks and windows

# %% [markdown]
# ### Linking peaks to windows

# %% [markdown]
# Create a `peak_window_matches` dataframe that contains peak - window - gene in long format

# %%
promoters = fragments.regions.coordinates

# %%
# peaks_name = "cellranger"
# peaks_name = "macs2_improved"
# peaks_name = "macs2_summits"
peaks_name = "macs2_leiden_0.1_merged"
# peaks_name = "rolling_500"

# %%
peaks_folder = chd.get_output() / "peaks" / dataset_name / peaks_name
peaks = pd.read_table(
    peaks_folder / "peaks.bed", names=["chrom", "start", "end"], usecols=[0, 1, 2]
)

# %%
import pybedtools

promoters_bed = pybedtools.BedTool.from_dataframe(
    promoters.reset_index()[["chrom", "start", "end", "gene"]].assign(start = lambda x:np.clip(x.start, 15000, np.inf))
)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)

# %%
intersect = promoters_bed.intersect(peaks_bed)
intersect = intersect.to_dataframe()

peaks = intersect
peaks.columns = ["chrom", "start", "end", "gene"]
peaks = peaks.loc[peaks["start"] != -1]
peaks.index = pd.Index(
    peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str),
    name="peak",
)


# %%
def center_peaks(peaks, promoters):
    promoter = promoters.loc[peaks["gene"]]

    peaks2 = peaks.copy()

    peaks2["start"] = np.where(
        promoter["strand"].values == 1,
        (peaks["start"] - promoter["tss"].values) * promoter["strand"].values,
        (peaks["end"] - promoter["tss"].values) * promoter["strand"].values,
    )
    peaks2["end"] = np.where(
        promoter["strand"].values == 1,
        (peaks["end"] - promoter["tss"].values) * promoter["strand"].values,
        (peaks["start"] - promoter["tss"].values) * promoter["strand"].values,
    )
    return peaks2


# %%
localpeaks = center_peaks(peaks, promoters)

# %%
# match all localpeaks with the windows
matched_peaks, matched_windows = np.where(
    (
        (localpeaks["start"].values[:, None] < np.array(design)[:, 0][None, :])
        & (localpeaks["end"].values[:, None] > np.array(design)[:, 1][None, :])
    )
)
# matched_peaks, matched_windows = np.where(
#     ~(
#         (localpeaks["start"].values[:, None] > np.array(design)[:, 1][None, :])
#         | (localpeaks["end"].values[:, None] < np.array(design)[:, 0][None, :])
#     )
# )

# %%
peak_window_matches = (
    pd.DataFrame(
        {
            "peak": localpeaks.index[matched_peaks],
            "window": genedeltacor.window.values[matched_windows],
            "gene": localpeaks["gene"].iloc[matched_peaks],
        }
    )
    .set_index("peak")
    .reset_index()
)

# %% [markdown]
# ### Is the most predictive window inside a peak?

# %%
gene_best_windows = genedeltacor.idxmin("window").to_pandas()
gene_best_windows = pd.DataFrame({"gene": genedeltacor.coords["gene"].values, "window": gene_best_windows.values}).set_index(["gene", "window"])

# %%
gene_best_windows = gene_best_windows.join(
    peak_window_matches.set_index(["gene", "window"])
).reset_index(level="window")
gene_best_windows = gene_best_windows.groupby("gene").first()

# %%
gene_best_windows["matched"] = ~pd.isnull(gene_best_windows["peak"])

# %%
gene_best_windows["cor_overall"] = genescores_overall["cor"]

# %%
gene_best_windows = gene_best_windows.sort_values("cor_overall", ascending=False)
gene_best_windows["ix"] = np.arange(1, gene_best_windows.shape[0] + 1)
gene_best_windows["cum_matched"] = (
    np.cumsum(gene_best_windows["matched"]) / gene_best_windows["ix"]
)
gene_best_windows["perc"] = gene_best_windows["ix"] / gene_best_windows.shape[0]

# %% [markdown]
# Of the top 5% most predictive genes, how many are inside a peak?

# %%
top_cutoff = 1.0
perc_within_a_peak = gene_best_windows["cum_matched"].iloc[
    int(gene_best_windows.shape[0] * top_cutoff) - 1
]
print(perc_within_a_peak)
print(
    f"Perhaps the most predictive window in the promoter is not inside of a peak?\nIndeed, for {1-perc_within_a_peak:.2%} of the {top_cutoff:.0%} best predicted genes, the most predictive window does not lie within a peak."
)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(gene_best_windows["perc"], 1 - gene_best_windows["cum_matched"])
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
ax.plot([top_cutoff, top_cutoff], [0, 1 - perc_within_a_peak], color="red")
annot = f"{1-perc_within_a_peak:.2%}"
ax.annotate(
    annot,
    xy=(top_cutoff, 1 - perc_within_a_peak),
    xycoords=("data", "data"),
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    color="red",
    bbox=dict(fc="#FFFFFF88"),
)
# ax.annotate([top_cutoff, top_cutoff], [0, 1-perc_within_a_peak], dashes = (2, 2), color = "red")
ax.set_xlabel("Top genes (acording to cor)")
ax.set_title(
    "% genes where most predictive locus is not contained in a peak",
    rotation=0,
    loc="left",
)

# %% [markdown]
# ### Are all predictive windows within a peak?

# %%
genescores_matched = (
    pd.concat([genedeltar2.to_dataframe("deltacor"), genelost.to_dataframe("lost").drop(columns = ["phase"]), geneeffect.to_dataframe("effect").drop(columns = ["phase"])], axis = 1)
    .join(peak_window_matches.set_index(["gene", "window"]))
    .groupby(["gene", "window"])
    .first()
    .reset_index(level="window")
)
genescores_matched["matched"] = ~pd.isnull(genescores_matched["peak"])
genescores_matched = genescores_matched.sort_values("deltacor", ascending=True)

# %%
genescores_matched["ix"] = np.arange(1, genescores_matched.shape[0] + 1)
genescores_matched["cum_matched"] = (
    np.cumsum(genescores_matched["matched"]) / genescores_matched["ix"]
)
genescores_matched["perc"] = genescores_matched["ix"] / genescores_matched.shape[0]

# %% [markdown]
# Of the top 5% most predictive sites, how many are inside a peak?

# %%
top_cutoff = 0.05
perc_within_a_peak = genescores_matched["cum_matched"].iloc[
    int(genescores_matched.shape[0] * top_cutoff)
]
print(perc_within_a_peak)
print(
    f"Perhaps there are many windows that are predictive, but are not contained in any peak?\nIndeed, {1-perc_within_a_peak:.2%} of the top {top_cutoff:.0%} predictive windows does not lie within a peak."
)

# %%
genescores_matched_oi = genescores_matched.iloc[
    : int(top_cutoff * genescores_matched.shape[0] * 2)
]

# %%
fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.plot(
    genescores_matched_oi["perc"], genescores_matched_oi["cum_matched"], color="#333"
)
annot = f"{perc_within_a_peak:.1%}"
ax.annotate(
    annot,
    xy=(top_cutoff, perc_within_a_peak),
    xycoords=("data", "data"),
    xytext=(0, -15),
    textcoords="offset points",
    va="top",
    ha="center",
    color="red",
    # bbox=dict(fc="#FFFFFF88"),
    # arrow with no pad
    arrowprops=dict(arrowstyle="->", color="red", shrinkA=0, shrinkB=0, lw=1),
)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
ax.set_xlabel("Top windows\n(ordered by $\Delta$ cor)")
ax.set_ylabel("% of windows in peak")
# ax.set_title(
#     "% of most predictive windows\nnot contained in a peak", rotation=0, loc="left"
# )

if dataset_name == "pbmc10k":
    manuscript.save_figure(fig, "2", f"predictive_windows_not_in_peak")

# %% [markdown]
# ### How much information do the non-peak regions contain?

# %%
matched_scores = (
    genescores_matched.groupby(["gene", "matched"])["deltacor"].sum().unstack()
)
matched_scores
print(
    f"Perhaps there is information outside of peaks?\nIndeed, {matched_scores.mean(0)[False] / matched_scores.mean(0).sum():.2%} of the cor is gained outside of peaks."
)

# %%
plotdata = genescores_matched.groupby(["gene", "matched"]).sum().reset_index()
sns.boxplot(x="matched", y="deltacor", data=plotdata)

# %%
genescores_matched_loci = (
    genescores_matched.groupby(["gene", "matched"])["deltacor"].sum().unstack()
)

# %%
gene_order = genescores_matched_loci.sum(1).sort_values(ascending=True).index[:5000]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

panel, ax = fig.main.add_right(polyptich.grid.Panel((4, 2)))
inside_peaks = (
    genescores_matched_loci.iloc[:, 1].sum() / genescores_matched_loci.sum().sum()
)
outside_peaks = (
    genescores_matched_loci.iloc[:, 0].sum() / genescores_matched_loci.sum().sum()
)
ax.bar(
    np.arange(len(gene_order)),
    -genescores_matched_loci.loc[gene_order, True],
    width=1,
    lw=0,
    label="In peaks ({}%)".format(round(inside_peaks * 100)),
    color="#0074D9",
)
ax.bar(
    np.arange(len(gene_order)),
    -genescores_matched_loci.loc[gene_order, False],
    bottom=-genescores_matched_loci.loc[gene_order, True],
    width=1,
    lw=0,
    label="Outside peaks ({}%)".format(round(outside_peaks * 100)),
    color="#FF851B",
)
ax.set_xlim(0, len(gene_order) + 1)
ax.set_xlabel("Genes (sorted by cor)")
ax.set_ylabel("$\\Delta$ cor")
sns.despine()
ax.legend(loc="upper left", ncol=2, frameon=False)
ax.set_ylim(0)

fig.plot()

# %%
manuscript.save_figure(fig, "2", "information_beyond_peaks")

# %%
genescores_matched_loci = (
    genescores_matched.groupby(["gene", "matched"])["deltacor"].sum().unstack() / (genescores_matched.groupby(["gene", "matched"])["lost"].sum().unstack()+1e-6)
)
genescores_matched_loci = genescores_matched_loci / (genescores_matched_loci.values.sum(1, keepdims = True)+1e-6)

# %%
gene_order = (genescores_matched_loci.sum(1)[gene_order] + (genescores_matched_loci.loc[:, False])).sort_values(ascending=False).index[:5000]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

panel, ax = fig.main.add_right(polyptich.grid.Panel((4, 2)))
inside_peaks = (
    genescores_matched_loci.iloc[:, 1].sum() / genescores_matched_loci.sum().sum()
)
outside_peaks = (
    genescores_matched_loci.iloc[:, 0].sum() / genescores_matched_loci.sum().sum()
)
ax.bar(
    np.arange(len(gene_order)),
    genescores_matched_loci.loc[gene_order, True],
    width=1,
    lw=0,
    label="In peaks ({}%)".format(int(inside_peaks * 100)),
    color="#0074D9",
)
ax.bar(
    np.arange(len(gene_order)),
    genescores_matched_loci.loc[gene_order, False],
    bottom=genescores_matched_loci.loc[gene_order, True],
    width=1,
    lw=0,
    label="Outside peaks ({}%)".format(int(outside_peaks * 100)),
    color="#FF851B",
)
ax.set_xlim(0, len(gene_order) + 1)
ax.set_xlabel("Genes")
ax.set_ylabel("")
sns.despine()
ax.legend(loc="upper left", ncol=2, frameon=True)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylabel("Fraction of $\\Delta$ cor,\nnormalized by # of fragments")

fig.plot()

# %%
manuscript.save_figure(fig, "2", "information_beyond_peaks_normalized")

# %% [markdown]
# ### Are opposing effects put into the same peak?

# %%
gene_peak_scores = pd.DataFrame(
    {
        "effect_min": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["effect"]
        .min(),
        "effect_max": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["effect"]
        .max(),
        "deltacor_min": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .min(),
        "deltacor_sum": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .sum(),
    }
)

gene_peak_scores["label"] = transcriptome.symbol(
    gene_peak_scores.index.get_level_values("gene")
).values

# %%
gene_peak_scores["effect_highest"] = np.maximum(
    np.abs(gene_peak_scores["effect_min"]), np.abs(gene_peak_scores["effect_max"])
)
gene_peak_scores["effect_highest_cutoff"] = (
    gene_peak_scores["effect_highest"] / 20
)  # we put the cutoff at 1/8 of the highest effect

# %%
gene_peak_scores["significant"] = gene_peak_scores["deltacor_min"] < -0.0005

# %%
gene_peak_scores["up"] = (
    (gene_peak_scores["effect_max"] > 0) & gene_peak_scores["significant"]
)
gene_peak_scores["down"] = (
    (gene_peak_scores["effect_min"] < 0) & gene_peak_scores["significant"]
)
gene_peak_scores["updown"] = gene_peak_scores["up"] & gene_peak_scores["down"]

# %%
gene_peak_scores = gene_peak_scores.sort_values("deltacor_min", ascending=True)

# %%
gene_peak_scores["ix"] = np.arange(1, gene_peak_scores.shape[0] + 1)
gene_peak_scores["cum_updown"] = (
    np.cumsum(gene_peak_scores["updown"]) / gene_peak_scores["ix"]
)
gene_peak_scores["perc"] = gene_peak_scores["ix"] / gene_peak_scores.shape[0]

# %%
contingency = gene_peak_scores.groupby(["up", "down"]).size().unstack()

# %%
fig, ax = plt.subplots(figsize = (2, 2))
# for x in [True, False]:
#     for y in [True, False]:
#         ax.annotate(
#             contingency.loc[x, y],
#             xy=(x, y),
#             xycoords=("data", "data"),
#             xytext=(0, 0),
#             textcoords="offset points",
#             va="center",
#             ha="center",
#             color="white",
#         )
# ax.set_xlabel("Up")
ax.set_ylabel("Positively predictive")
ax.set_xlabel("Negatively predictive")
ax.matshow(contingency)
for i in range(2):
    for j in range(2):
        ax.text(j, i, contingency.iloc[i, j], ha = "center", va = "center", color = "white" if contingency.iloc[i, j] <20000 else "black")
ax.set_xticks([0, 1])
ax.set_xticklabels(["False", "True"])
ax.xaxis.tick_bottom()
ax.set_yticks([0, 1])
ax.set_yticklabels(["False", "True"])

manuscript.save_figure(fig, "2", "updown_contingency")

# %%
contingency.loc[True, True] / (contingency.loc[False, True] + contingency.loc[True, False] + contingency.loc[True, True])


# %% [markdown]
# ### Is the most informative locus in a peak also its summit?

# %%
def match_deltacor_retained(df, deltacor_quantile=1.0, retained_quantile=1.0):
    # return (
    #     (df["lost"] >= df["lost"].quantile(retained_quantile))
    #     & (df["deltacor"] <= df["deltacor"].quantile(1 - deltacor_quantile))
    # ).any()
    return np.argmax(df["lost"]) == np.argmin(df["deltacor"])


# %%
peak_max_matches = (
    genescores_matched.query("matched")
    .groupby(["gene", "peak"])
    .apply(match_deltacor_retained, deltacor_quantile=1.0, retained_quantile=1.0)
)
# peak_max_matches = genescores_matched.query("matched").groupby(["gene", "peak"]).apply(match_deltacor_retained, deltacor_quantile = 0.8, retained_quantile = 0.9)

# %%
genescores_matched["position"] = design["window_mid"].loc[genescores_matched["window"]].values

# %%
peak_max_scores = pd.DataFrame(
    {
        "match": peak_max_matches,
        "deltacor_sum": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .sum(),
        "position": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["position"]
        .mean(),
    }
)
peak_max_scores = peak_max_scores.sort_values("deltacor_sum", ascending=True)

# %%
peak_max_scores["ix"] = np.arange(1, peak_max_scores.shape[0] + 1)
peak_max_scores["cum_nonmatched"] = (
    np.cumsum(~peak_max_scores["match"]) / peak_max_scores["ix"]
)
peak_max_scores["perc"] = peak_max_scores["ix"] / peak_max_scores.shape[0]

# %% [markdown]
# Of the top 20% most predictive peaks, how many have a match between # of fragments and most predictive window

# %%
top_cutoff = 0.1
perc_notmatched = peak_max_scores["cum_nonmatched"].iloc[
    int(peak_max_scores.shape[0] * top_cutoff)
]
print(perc_notmatched)
print(
    f"Perhaps within a peak the peak maximum is not really the most predictive window?\nIndeed, {perc_notmatched:.2%} of the top {top_cutoff:.0%} predictive peaks does not have a match between the top predictive locus and the max of the peak."
)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(peak_max_scores["perc"], peak_max_scores["cum_nonmatched"])

ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
annot = f"{perc_notmatched:.2%}"
ax.annotate(
    annot,
    xy=(top_cutoff, perc_notmatched),
    xycoords=("data", "data"),
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    color="red",
    bbox=dict(fc="#FFFFFF88"),
    arrowprops={"arrowstyle": "-", "ec": "red"},
)
ax.set_xlabel("Top peaks (acording to delta cor)")
ax.set_title(
    "% peaks where summit does not match top $\Delta$ cor", rotation=0, loc="left"
)

# %% [markdown]
# ### What is the distance between the peak maximum and the most predictive window within a peak?

# %%
peak_max_scores["distance"] = (
    genescores_matched.reset_index()
    .set_index("position")
    .groupby(["gene", "peak"])["deltacor"]
    .idxmin()
    - genescores_matched.reset_index()
    .set_index("position")
    .groupby(["gene", "peak"])["lost"]
    .idxmax()
)

# %%
fig, ax = plt.subplots()
ax.hist(
    peak_max_scores.query("perc < @top_cutoff")["distance"], range=(-500, 500), bins=11
)

# %%
top_cutoff = 0.1

# %%
fig, ax = plt.subplots(figsize=(2, 2))
sns.ecdfplot(
    np.abs(peak_max_scores.query("perc < @top_cutoff").query("distance > 0")["distance"]), color="black"
)
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_xlabel("Distance between summit\nand most predictive position")
ax.set_xlim(-10, 1000)

manuscript.save_figure(fig, "4", "peak_summit_distance")
