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

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_name = "pbmc10k"
# dataset_name = "hspc"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
if dataset_name == "pbmc10k/subsets/top250":
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")

regions_name = "100k100k"
# regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

# %%
probs_mean_bins = pd.DataFrame(
    {"cut_exp":[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., 4, 8, np.inf]}
)
probs_mean_bins["cut"] = np.log(probs_mean_bins["cut_exp"])
probs_mean_bins["label"] = ["<" + str(probs_mean_bins["cut_exp"][0])] + ["≥" + str(x) for x in probs_mean_bins["cut_exp"].astype(str)[:-1]]

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": list(np.log(np.round(np.logspace(np.log(1.25), np.log(2.5), 7, base = np.e), 5))) + [np.inf]}
)
clusterprobs_diff_bins["cut_exp"] = np.exp(clusterprobs_diff_bins["cut"])
clusterprobs_diff_bins["label"] = ["<" + str(np.round(clusterprobs_diff_bins["cut_exp"][0], 1))] + ["≥" + str(x) for x in np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1]]
clusterprobs_diff_bins["do_label"] = True
clusterprobs_diff_bins

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
step = 10
desired_x = np.arange(*fragments.regions.window, step=step) - fragments.regions.window[0]

# %%
clustering.var

# %%
# clusters_oi = ["CD14+ Monocytes", "FCGR3A+ Monocytes"]
# clusters_oi = ["pDCs", "NK"]
# clusters_oi = ["CD14+ Monocytes", "NK"]
# clusters_oi = ["CD4 naive T", "CD4 memory T"]
clusters_oi = ["Erythroblast", "HSPC"]

# %%
scores = []

# for region_id in tqdm.tqdm(fragments.var.index[:500]):
for region_id in tqdm.tqdm(fragments.var.index):
    region = fragments.regions.coordinates.loc[region_id]

    # calculate chd metrics
    probs = regionpositional.probs[region_id].sel(cluster = clusters_oi)
    region_ix = fragments.regions.coordinates.index.get_indexer([region_id])[0]

    x_raw = probs.coords["coord"].values - fragments.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
    ).numpy()

    ymean = y.mean(0)
    ymax = y.max(0)
    ymed = np.median(y, 0)
    z = y - ymean

    zstd = z.std(0)
    zmax = z.max(0)

    max_ix = np.abs(z).argmax(0)
    zcentermax = z[max_ix, np.arange(z.shape[1])]

    data_binned = pd.DataFrame(
        {"coord": desired_x, "probs_mean": ymean, "probs_max":ymax, "probs_med":ymed, "diff_std": zstd, "diff_max": z.max(0), "diff_centermax":zcentermax}
    )

    scores.append(data_binned.assign(region_id=region_id))
scores = pd.concat(scores)

# %%
scores["probs_mean_bin"] = np.digitize(scores["probs_mean"], bins=probs_mean_bins["cut"])
scores["probs_med_bin"] = np.digitize(scores["probs_med"], bins=probs_mean_bins["cut"])
scores["diff_centermax_bin"] = np.digitize(scores["diff_centermax"], bins=clusterprobs_diff_bins["cut"])
scores["diff_max_bin"] = np.digitize(scores["diff_max"], bins=clusterprobs_diff_bins["cut"])
scores["diff_std_bin"] = np.digitize(scores["diff_std"], bins=clusterprobs_diff_bins["cut"])
scores["probs_max_bin"] = np.digitize(scores["probs_max"], bins=probs_mean_bins["cut"])

# %%
plotdata = scores.groupby(["diff_max_bin", "probs_mean_bin"]).size().unstack().T 
# plotdata = scores.groupby(["diff_std_bin", "probs_mean_bin"]).size().unstack().T
# plotdata = scores.groupby(["diff_std_bin", "probs_max_bin"]).size().unstack().T
plotdata = plotdata.fillna(0.0)
# plotdata = scores.groupby(["diff_centermax_bin", "probs_mean_bin"]).size().unstack().T
# plotdata = np.log(scores.groupby(["diff_centermax_bin", "probs_med_bin"]).size().unstack().T + 1)
plotdata = (plotdata / plotdata.values.sum(1, keepdims=True))

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
norm = mpl.colors.SymLogNorm(linthresh=1e-9, linscale=0.1, vmin=0, vmax=1)
cmap = mpl.colormaps["viridis"]
ax.matshow(plotdata, norm = norm, cmap = cmap)

ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"], rotation=90)
fig.plot()

fig.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5)

# %%
# plotdata = scores.groupby(["diff_max_bin", "probs_mean_bin"]).size().unstack().T 
plotdata = scores.groupby(["diff_std_bin", "probs_mean_bin"]).size().unstack().T
# plotdata = scores.groupby(["diff_std_bin", "probs_max_bin"]).size().unstack().T
plotdata = plotdata.fillna(0.0)
# plotdata = scores.groupby(["diff_centermax_bin", "probs_mean_bin"]).size().unstack().T
# plotdata = np.log(scores.groupby(["diff_centermax_bin", "probs_med_bin"]).size().unstack().T + 1)
plotdata = (plotdata / plotdata.values.sum(1, keepdims=True))

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
norm = mpl.colors.SymLogNorm(linthresh=1e-9, linscale=0.1, vmin=0, vmax=1)
cmap = mpl.colormaps["viridis"]
ax.matshow(plotdata, norm = norm, cmap = cmap)

ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"], rotation=90)
fig.plot()

fig.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5)

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
norm = mpl.colors.SymLogNorm(linthresh=1e-9, linscale=0.1, vmin=0, vmax=1)
cmap = mpl.colormaps["viridis"]
ax.matshow(plotdata, norm = norm, cmap = cmap)

ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"], rotation=90)
fig.plot()

fig.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5)

# %% [markdown]
# ## Difference with peaks

# %%
# peakcaller = "rolling_500"
peakcaller = "macs2_leiden_0.1_merged"

# diffexp = "t-test"
# diffexp = "snap"
# diffexp = "t-test-foldchange"
diffexp = "wilcoxon"

scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / diffexp / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "logreg" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_summits" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "cellranger" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "cellranger" / "t-test" / "scoring" / "regionpositional"
differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
differential_slices_peak.window = fragments.regions.window

# %%
n_desired_positions = 24638218

# %%
slicescores_peak = differential_slices_peak.get_slice_scores(regions = fragments.regions).set_index(["region_ix", "start", "end", "cluster_ix"])["score"].unstack().max(1).to_frame("score").reset_index()

slicescores_peak = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores_peak, fragments.regions.coordinates)
slicescores_peak["slice"] = pd.Categorical(slicescores_peak["chrom"].astype(str) + ":" + slicescores_peak["start_genome"].astype(str) + "-" + slicescores_peak["end_genome"].astype(str))
slicescores_peak = slicescores_peak.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome", "score"]].first()

slicescores_peak = slicescores_peak.sort_values("score", ascending=False)
slicescores_peak["length"] = slicescores_peak["end"] - slicescores_peak["start"]
slicescores_peak["cum_length"] = slicescores_peak["length"].cumsum()

slices_peak = slicescores_peak[slicescores_peak["cum_length"] <= n_desired_positions].reset_index(drop=True)
# slices_peak = slicescores_peak[slicescores_peak["score"] >= 5].reset_index(drop=True)

# %%
slices_peak["score"].hist(bins = 100)

# %%
slices_peak["start_bin"] = (slices_peak["start"] - fragments.regions.window[0]) // step
slices_peak["end_bin"] = (slices_peak["end"] - fragments.regions.window[0]) // step

# %%
scores = []

desired_x = np.arange(*regionpositional.regions.window, step=step) - regionpositional.regions.window[0]

found = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
tot = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
for region_ix, subregions in tqdm.tqdm(slices_peak.groupby("region_ix")):
    region_id = fragments.regions.coordinates.index[region_ix]
    region = fragments.regions.coordinates.loc[region_id]

    probs = regionpositional.probs[region_id]

    x_raw = probs.coords["coord"].values - regionpositional.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
    ).numpy()
    ymean = y.mean(0)

    z = y - ymean
    zmax = np.abs(z.max(0))
    # zmax = z.max(0)

    ybin = np.searchsorted(probs_mean_bins["cut"].values, ymean)
    zbin = np.searchsorted(clusterprobs_diff_bins["cut"].values, zmax)

    # get association
    ixs = np.concatenate([np.arange(int(subregion["start_bin"]), int(subregion["end_bin"])) for _, subregion in subregions.iterrows()])

    # add bin counts
    tot += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin, minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
    found += np.bincount(ybin[ixs] * len(clusterprobs_diff_bins) + zbin[ixs], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))

tot = tot.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins))
found = found.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins))

# %%
contingencies = np.stack([
    tot,
    found.sum() - found,
    tot - found,
    found
]).T.reshape(-1, 2, 2)

# %%
odds = (((found)/(tot)) / (found.sum()/tot.sum()))
# odds[odds == 0.] = 1.

# %%
sns.heatmap(odds)

# %%
perc = found / tot
plotdata = perc
plotdata[tot < 50] = np.nan

# %%
sns.heatmap(perc)

# %%
probs_mean_bins

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.BuPu
odds_max = 8
norm = mpl.colors.Normalize(0, 1)

ax.matshow(perc, cmap=cmap, norm=norm)

ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(1, len(probs_mean_bins)) - 0.5)
# ax.set_yticklabels([])
ax.set_yticklabels(probs_mean_bins["cut_exp"][:-1])

ax.xaxis.set_ticks_position("bottom")
ax.set_xticks(np.arange(1, len(clusterprobs_diff_bins)) - 0.5, minor=True)
ax.set_xticklabels(np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1], minor=True, rotation=90)
ax.set_xticks([])
ax.set_xticklabels([])

# ax.set_xlabel("Accessibility\nfold change")

color = "#0074D9"

##
x0 = 5
x1 = 8
y0 = 8
y1 = 10
y = len(probs_mean_bins) - h - yadjust
rect = mpl.patches.Rectangle(
    (x0 - 0.5, y0 - 0.5), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor="none", linestyle="--", clip_on=False, zorder=20
)
ax.add_patch(rect)

perc_found = 1-found[y0:y1, x0:x1].sum() / tot[y0:y1, x0:x1].sum()

text = ax.annotate(
    f"{perc_found:.1%}",
    (x - 0.5 + w, y0 + (y1-y0)/2 - 0.5),
    (5, 0),
    textcoords="offset points",
    color=color,
    fontsize=10,
    ha="left",
    va="center",
    zorder=30,
    fontweight="bold",
)
text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])

##
x0 = 5
x1 = 8
y0 = 6
y1 = 8
y = len(probs_mean_bins) - h - yadjust
rect = mpl.patches.Rectangle(
    (x0 - 0.5, y0 - 0.5), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor="none", linestyle="--", clip_on=False, zorder=20
)
ax.add_patch(rect)

perc_found = 1-found[y0:y1, x0:x1].sum() / tot[y0:y1, x0:x1].sum()

text = ax.annotate(
    f"{perc_found:.1%}",
    (x - 0.5 + w, y0 + (y1-y0)/2 - 0.5),
    (5, 0),
    textcoords="offset points",
    color=color,
    fontsize=10,
    ha="left",
    va="center",
    zorder=30,
    fontweight="bold",
)
text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])

##
x0 = 5
x1 = 8
y0 = 4
y1 = 6
y = len(probs_mean_bins) - h - yadjust
rect = mpl.patches.Rectangle(
    (x0 - 0.5, y0 - 0.5), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor="none", linestyle="--", clip_on=False, zorder=20
)
ax.add_patch(rect)

perc_found = 1-found[y0:y1, x0:x1].sum() / tot[y0:y1, x0:x1].sum()

text = ax.annotate(
    f"{perc_found:.1%}",
    (x - 0.5 + w, y0 + (y1-y0)/2 - 0.5),
    (5, 0),
    textcoords="offset points",
    color=color,
    fontsize=10,
    ha="left",
    va="center",
    zorder=30,
    fontweight="bold",
)
text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])

##
x0 = 5
x1 = 8
y0 = 4
y1 = 10
y = len(probs_mean_bins) - h - yadjust
rect = mpl.patches.Rectangle(
    (x0 - 0.5, y0 - 0.5), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor="none", linestyle="--", clip_on=False, zorder=20
)
# ax.add_patch(rect)

perc_found = 1-found[y0:y1, x0:x1].sum() / tot[y0:y1, x0:x1].sum()
text = ax.annotate(
    f"{perc_found:.1%}",
    (x0+(x1-x0)/2-0.5, y0-0.5),
    (x0+(x1-x0)/2-0.5, -2.0),
    # (1.05, y - 2),
    textcoords=mpl.transforms.blended_transform_factory(ax.transData, ax.transData),
    color=color,
    fontsize=10,
    ha="center",
    va="top",
    zorder=30,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", pad=0),
    arrowprops=dict(edgecolor=color, arrowstyle="->", shrinkB=0),
)
text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])

##
color = "#FF4136"
x0 = 0
x1 = 3
y0 = 0
y1 = 10
y = len(probs_mean_bins) - h - yadjust
rect = mpl.patches.Rectangle(
    (x0 - 0.5, y0 - 0.5), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor="none", linestyle="--", clip_on=False, zorder=20
)
ax.add_patch(rect)

perc_found = found[y0:y1, x0:x1].sum() / found.sum()
text = ax.annotate(
    f"{perc_found:.1%}",
    (x0+(x1-x0)/2-0.5, y0-0.5),
    (x0+(x1-x0)/2-0.5, -2.0),
    # (1.05, y - 2),
    textcoords=mpl.transforms.blended_transform_factory(ax.transData, ax.transData),
    color=color,
    fontsize=10,
    ha="center",
    va="top",
    zorder=30,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", pad=0),
    arrowprops=dict(edgecolor=color, arrowstyle="->", shrinkB=0),
)
text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])

sns.despine(fig, ax)

manuscript.save_figure(fig, "4", f"mean_vs_variance_peak_captured_{peakcaller}_{diffexp}")

# %%
sns.heatmap(np.log(found))

# %%
found.sum()

# %%
fig_colorbar = plt.figure(figsize=(2.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=norm, cmap=cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal", ticks=[0, 0.5, 1])
ax_colorbar.set_xticklabels(["0%", "50%", "100%"])
ax_colorbar.xaxis.set_label_position('bottom')
colorbar.set_label("Positions captured\nin differential peak")
manuscript.save_figure(fig_colorbar, "4", "colorbar_captured")

# %%

# %%

# %% [markdown]
# ## Locality

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
genes_oi = transcriptome.var.index[:10]

# %%
regionmultiwindow.interpolate(genes_oi, force = True)

# %%
scores = []
for gene_oi in tqdm.tqdm(genes_oi):
    print(gene_oi)
    position_scores = regionmultiwindow.get_plotdata(gene_oi)
    deltacors = position_scores["deltacor"].values
    losts = position_scores["lost"].values

    window_sizes = [10000, 5000, 2000, 1000, 500, 200, 100, 50, 25]
    for window_size in window_sizes:
        i = 0
        step_size = 100
        for i in np.arange(0, len(position_scores), step_size):
            l = losts[i:i+window_size]
            d = deltacors[i:i+window_size]
            if sum(l) > 0.01:
                scores.append({"i":i, "window_size":window_size, "cor":np.corrcoef(-d, l)[0, 1], "mindeltacor":d.mean(), "gene_oi":gene_oi})
            else:
                scores.append({"i":i, "window_size":window_size, "cor":0, "gene_oi":gene_oi})

# %%
fig, ax = plt.subplots()

plotdata = pd.DataFrame(scores).fillna(0.)
for i, (window_size, plotdata) in enumerate(plotdata.query("mindeltacor < -0.001").groupby("window_size")):
    ax.boxplot(plotdata["cor"], positions = [i], showfliers = False, widths = 0.5)
ax.set_xticks(np.arange(len(window_sizes)))
ax.set_xticklabels(window_sizes)

# %%
meanscores = pd.DataFrame(scores).fillna(0.).groupby(["gene_oi", "window_size"]).mean()
# meanscores.style.bar()

fig, ax = plt.subplots()
for gene, plotdata in meanscores.reset_index().groupby("gene_oi"):
    ax.plot(plotdata["window_size"], plotdata["cor"].values, label = transcriptome.symbol(gene))
# ax.plot(meanscores.index, meanscores["cor"].values)
ax.set_xscale("log")
