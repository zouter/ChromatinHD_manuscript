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
# dataset_name = "lymphoma"
# dataset_name = "liver"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
if dataset_name == "pbmc10k/subsets/top250":
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")

# regions_name = "100k100k"
regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
import chromatinhd.data.peakcounts

# %%
# slices = regionpositional.calculate_slices(-1., step = 1)
# top_slices = regionpositional.calculate_top_slices(slices, 1.5)

scoring_folder = regionpositional.path / "top" / "-1-1.5"
top_slices = pickle.load(open(scoring_folder / "top_slices.pkl", "rb"))

slicescores = top_slices.get_slice_scores(regions = fragments.regions)
slices = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores, fragments.regions.coordinates)
slicescores["slice"] = pd.Categorical(slicescores["chrom"].astype(str) + ":" + slicescores["start_genome"].astype(str) + "-" + slicescores["end_genome"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome"]].first()

# %%
# create bed file using original region_ix and relative coordinates to region start
import pybedtools
slices["start0"] = (slices["start"] - fragments.regions.window[0]).astype(int)
slices["end0"] = (slices["end"] - fragments.regions.window[0]).astype(int)
bed = pybedtools.BedTool.from_dataframe(slices[['region_ix', 'start0', 'end0']]).sort().merge()

# %%
n_desired_positions = (bed.to_dataframe()["end"] - bed.to_dataframe()["start"]).sum()
n_desired_positions

# %%
from chromatinhd_manuscript.designs_diff import (
    dataset_latent_peakcaller_diffexp_combinations as cre_design,
)
cre_design = cre_design.loc[
    (cre_design["dataset"] == dataset_name) &
    (cre_design["latent"] == latent) &
    (cre_design["regions"] == regions_name) &
    True
].copy()
cre_design["method"] = cre_design["peakcaller"] + "/" + cre_design["diffexp"]
cre_design = cre_design.loc[cre_design.diffexp == 't-test']
cre_design = cre_design.loc[~(cre_design.peakcaller.str.contains('rolling') & (cre_design['diffexp'] != 't-test'))]
cre_design = cre_design.loc[~(cre_design.peakcaller.str.contains('gene_body'))]

design = pd.concat([
    cre_design,
    pd.DataFrame({
        "dataset":[dataset_name],
        "latent":[latent],
        "regions":[regions_name],
        "method":["chd"],
    })
], ignore_index = True)
design.index = np.arange(len(design))
design["ix"] = np.arange(len(design))

# %%
prsets = {}
slicesets = {}

# %%
slicesets[design.index[-1]] = slices
prsets[design.index[-1]] = bed

# %%
import pybedtools

# %%
for design_ix, row in tqdm.tqdm(design.iterrows()):
    if pd.isnull(row["peakcaller"]):
        continue

    if design_ix in prsets:
        continue
    scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / row["peakcaller"] / row["diffexp"] / "scoring" / "regionpositional"

    try:
        differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))
    except FileNotFoundError:
        print(row)
        continue

    # fix
    differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
    differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
    differential_slices_peak.window = fragments.regions.window 

    # get top slices
    slicescores_peak = differential_slices_peak.get_slice_scores(regions = fragments.regions).set_index(["region_ix", "start", "end", "cluster_ix"]).drop_duplicates()["score"].unstack().max(1).to_frame("score").reset_index()

    slicescores_peak = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores_peak, fragments.regions.coordinates)
    slicescores_peak["slice"] = pd.Categorical(slicescores_peak["chrom"].astype(str) + ":" + slicescores_peak["start_genome"].astype(str) + "-" + slicescores_peak["end_genome"].astype(str))
    slicescores_peak = slicescores_peak.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome", "score"]].first()

    slicescores_peak = slicescores_peak.sort_values("score", ascending=False)
    slicescores_peak["length"] = slicescores_peak["end"] - slicescores_peak["start"]
    slicescores_peak["cum_length"] = slicescores_peak["length"].cumsum()
    slices_peak = slicescores_peak[slicescores_peak["cum_length"] <= n_desired_positions].reset_index(drop=True)

    # create pyranges
    import pybedtools
    slices_peak["start0"] = (slices_peak["start"] - fragments.regions.window[0]).astype(int)
    slices_peak["end0"] = (slices_peak["end"] - fragments.regions.window[0]).astype(int)
    bed_peak = pybedtools.BedTool.from_dataframe(slices_peak[['region_ix', 'start0', 'end0']]).sort().merge()

    prsets[design_ix] = bed_peak
    slicesets[design_ix] = slices_peak

# %%
design = design.loc[prsets.keys()].copy()

# %% [markdown]
# ## Lengths

# %%
length_bins = pd.DataFrame([
    # [20, "<20bp"],
    [50, "≥20bp"],
    [100, "≥50bp"],
    [200, "≥100bp"],
    [500, "≥200bp"],
    [1000, "≥500bb"],
    [2000, "≥500bb"],
    # [np.inf, "≥1kb"],
], columns = ["cut", "label"])
length_bins["ix"] = np.arange(len(length_bins))
length_bins.index = pd.Index(length_bins["ix"], name="bin")

# %%
fig, ax = plt.subplots()

ax.set_xscale("log")
ax.set_xlim(50, 5000)

for design_ix, bed in prsets.items():
    bed2 = bed.slop(b = 20, genome = "hg38").merge()
    x, y = chd.utils.ecdf.weighted_ecdf(
        bed2.to_dataframe()["end"] - bed2.to_dataframe()["start"],
        (bed2.to_dataframe()["end"] - bed2.to_dataframe()["start"])
    )

    color = "#333"
    if design.loc[design_ix]["peakcaller"] == 'encode_screen':
        color = "red"
    elif design.loc[design_ix]["peakcaller"] == 'macs2_summit':
        color = "green"
    elif design.loc[design_ix]["peakcaller"] == 'rolling_500':
        color = "orange"
    elif design.loc[design_ix]["peakcaller"] == 'cellranger':
        color = "purple"
    elif design.loc[design_ix]["peakcaller"] == 'macs2_leiden_0.1_merged':
        color = "pink"
    elif design.loc[design_ix]["peakcaller"] == 'macs2_summits':
        color = "lightgreen"
    elif design.loc[design_ix]["peakcaller"] == 'rolling_50':
        color = "yellow"
    elif pd.isnull(design.loc[design_ix]["peakcaller"]):
        color = "blue"

    ax.plot(x, y,color = color)

# %%
ecdfs = {}
for design_ix, bed in prsets.items():
    if design.loc[design_ix]["peakcaller"] in ["gene_body"]:
        continue
    # peaks = bed.merge().to_dataframe()
    peaks = bed.slop(b = 20, genome = "hg38").merge().to_dataframe()
    peaks["length"] = peaks["end"] - peaks["start"]
    peaks = peaks.loc[peaks["length"] > 25]
    peaks["length"] = peaks["length"] - 20
    x, y = chd.utils.ecdf.weighted_ecdf(
        peaks["length"],
        peaks["length"]
    )
    ecdfs[design_ix] = (x, y)

# %%
import textwrap

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap(padding_width = 0.05, padding_height = 0.))

for ix, focus in enumerate(["encode_screen/t-test", "macs2_summits/t-test", "macs2_leiden_0.1_merged/t-test", "rolling_50/t-test", "rolling_500/t-test"]):
    panel, ax = fig.main.add(polyptich.grid.Panel((0.8, 0.8)))

    ax.set_xscale("log")
    ax.set_xlim(50, 5000)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_xticks(length_bins["cut"])

    if ix == 0:
        ax.set_xticklabels([chd.plot.tickers.distance_ticker(x) for x in length_bins["cut"]], rotation=90, fontsize=8)
        ax.set_xlabel("Region length")
        ax.set_ylabel("ECDF")
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0%", "50%", "100%"])
    else:
        ax.set_xticklabels([])

    for design_ix, (x, y) in ecdfs.items():
        color = "#CCCCCC33"
        zorder = 0
        if (design.loc[design_ix]["method"] == focus):
            focus_color = color = chdm.methods.differential_methods.loc[design.loc[design_ix]["method"]]["color"]
            zorder = 10
        elif (design.loc[design_ix]["method"] == "chd"):
            color = chdm.methods.differential_methods.loc[design.loc[design_ix]["method"]]["color"]
            zorder = 5
        ax.plot(x, y, color = color, zorder = zorder)
    ax.set_title("\n".join(textwrap.wrap(chdm.methods.differential_methods.loc[focus]["label"], 15)), fontsize = 8, color = focus_color)
    fig.plot()
manuscript.save_figure(fig, "4", "length_ecdf")

# %% [markdown]
# ## Overlap

# %%
import itertools

# %%
scores = itertools.combinations(design.index, 2)
scores = list(scores)
scores = pd.DataFrame(scores, columns=["design_ix1", "design_ix2"]).set_index(["design_ix1", "design_ix2"])


# %%
def intersect_intervals(start1, start2, end1, end2):
    return np.maximum(
        0, np.minimum(end1, end2[:, None]) - np.maximum(start1, start2[:, None])
    )


def union_intervals(intersect, start1, start2, end1, end2):
    return (end1 - start1) + (end2[:, None] - start2[:, None]) - intersect


def jaccard_intervals(start1, start2, end1, end2):
    intersect = intersect_intervals(start1, start2, end1, end2)
    union = union_intervals(intersect, start1, start2, end1, end2)
    return intersect / union


start1 = np.array([300, 500, 1000])
end1 = np.array([500, 1000, 2000])

start2 = np.array([100, 1500])
end2 = np.array([200, 1800])

intersect = intersect_intervals(start1, start2, end1, end2)
union = union_intervals(intersect, start1, start2, end1, end2)
jaccard = jaccard_intervals(start1, start2, end1, end2)


# %%
def calculate_slicescore_F1(slicescores1, slicescores2, cluster_info, region_info, window):
    slicescores1["regionxcluster"] = (
        slicescores1["region_ix"].astype(int) * len(cluster_info)
        + slicescores1["cluster_ix"].astype(int)
    ).values
    slicescores1 = slicescores1.sort_values("regionxcluster")

    slicescores2["regionxcluster"] = (
        slicescores2["region_ix"].astype(int) * len(cluster_info)
        + slicescores2["cluster_ix"].astype(int)
    ).values
    slicescores2 = slicescores2.sort_values("regionxcluster")

    n_regionxcluster = len(region_info) * len(cluster_info)
    regionxcluster_indptr1 = chd.utils.indices_to_indptr(
        slicescores1["regionxcluster"].values, n_regionxcluster
    )

    start1 = slicescores1["start"].values
    end1 = slicescores1["end"].values
    start2 = slicescores2["start"].values
    end2 = slicescores2["end"].values

    regionxcluster_indptr2 = chd.utils.indices_to_indptr(
        slicescores2["regionxcluster"].values, n_regionxcluster
    )

    return calculate_F1(
        regionxcluster_indptr1,
        regionxcluster_indptr2,
        n_regionxcluster,
        start1,
        end1,
        start2,
        end2,
        window,
    )


# import numba
# @numba.jit
def calculate_F1(
    regionxcluster_indptr1,
    regionxcluster_indptr2,
    n_regionxcluster,
    start1,
    end1,
    start2,
    end2,
    window,
):
    recoveries = []
    relevances = []
    for regionxcluster in np.arange(n_regionxcluster):
        i1 = regionxcluster_indptr1[regionxcluster]
        j1 = regionxcluster_indptr1[regionxcluster + 1]
        i2 = regionxcluster_indptr2[regionxcluster]
        j2 = regionxcluster_indptr2[regionxcluster + 1]

        if j1 - i1 == 0:
            if j2 - i2 == 0:
                continue
            else:
                relevances.append([0] * (j2 - i2))
        elif j2 - i2 == 0:
            recoveries.append([0] * (j1 - i1))
        else:
            jac = jaccard_intervals(
                start1[i1:j1], start2[i2:j2], end1[i1:j1], end2[i2:j2]
            )

            relevances.append(jac.max(1))
            recoveries.append(jac.max(0))
    recoveries = np.hstack(recoveries)
    relevances = np.hstack(relevances)
    assert len(recoveries) == len(start1)
    assert len(relevances) == len(start2)
    return 2 / (1 / recoveries.mean() + 1 / relevances.mean())

def calculate_jaccard(
    regionxcluster_indptr1,
    regionxcluster_indptr2,
    n_regionxcluster,
    start1,
    end1,
    start2,
    end2,
    window,
):
    recoveries = []
    relevances = []
    for regionxcluster in np.arange(n_regionxcluster):
        i1 = regionxcluster_indptr1[regionxcluster]
        j1 = regionxcluster_indptr1[regionxcluster + 1]
        i2 = regionxcluster_indptr2[regionxcluster]
        j2 = regionxcluster_indptr2[regionxcluster + 1]

        if j1 - i1 == 0:
            if j2 - i2 == 0:
                continue
            else:
                relevances.append([0] * (j2 - i2))
        elif j2 - i2 == 0:
            recoveries.append([0] * (j1 - i1))
        else:
            jac = jaccard_intervals(
                start1[i1:j1], start2[i2:j2], end1[i1:j1], end2[i2:j2]
            )

            relevances.append(jac.max(1))
            recoveries.append(jac.max(0))
    recoveries = np.hstack(recoveries)
    relevances = np.hstack(relevances)
    assert len(recoveries) == len(start1)
    assert len(relevances) == len(start2)
    return 2 / (1 / recoveries.mean() + 1 / relevances.mean())


# %%
jaccards = []

for design_ix1, design_ix2 in tqdm.tqdm(itertools.combinations(design.index, 2)):
    if design_ix1 not in slicesets or design_ix2 not in slicesets:
        continue
    slices1 = slicesets[design_ix1]
    slices2 = slicesets[design_ix2]

    slices1["cluster_ix"] = 0
    slices2["cluster_ix"] = 0

    f1 = calculate_slicescore_F1(slices1, slices2, cluster_info = clustering.cluster_info, region_info = fragments.var, window = fragments.regions.window)

    scores.loc[(design_ix1, design_ix2), "f1"] = f1

    # jaccard
    bed1 = prsets[design_ix1]
    bed2 = prsets[design_ix2]

    intersect = bed1.intersect(bed2)
    union = bed1.cat(bed2).sort().merge()

    if len(intersect) == 0:
        jaccard = 0.
    else:
        jaccard = (intersect.to_dataframe()["end"] - intersect.to_dataframe()["start"]).sum() / (union.to_dataframe()["end"] - union.to_dataframe()["start"]).sum()

    scores.loc[(design_ix1, design_ix2), "jaccard"] = jaccard

# %%
f1 = scores["f1"].unstack().reindex(index = design.index, columns = design.index)
f1 = f1.fillna(0) + f1.fillna(0).T

jaccard = scores["jaccard"].unstack().reindex(index = design.index, columns = design.index)
jaccard = jaccard.fillna(0) + jaccard.fillna(0).T

# %%
# define distance
distance = (1 - f1.values)
distance = (distance + distance.T) / 2
np.fill_diagonal(distance, 0)

# cluster
import scipy.spatial
dist = scipy.spatial.distance.squareform(distance)
cluster_f1 = scipy.cluster.hierarchy.linkage(dist, method="centroid")

# define distance
distance = 1 - jaccard.values
distance = (distance + distance.T) / 2
np.fill_diagonal(distance, 0)

# cluster
import scipy.spatial
dist = scipy.spatial.distance.squareform(distance)
cluster_jaccard = scipy.cluster.hierarchy.linkage(dist, method="centroid")

# reorder
design["ix"] = pd.Series(
    np.arange(len(f1)), f1.index[scipy.cluster.hierarchy.leaves_list(cluster_f1)]
).reindex(design.index)

# %%
design["label"] = chdm.methods.differential_methods.loc[design["method"]]["label"].values
# design["label"] = design['method']

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

resolution = 0.12
dim = (len(design) * resolution, len(design) * resolution)

# vertical dendrogram
panel, ax_dendrogram = fig.main[0, 1] = polyptich.grid.Panel((dim[0], 0.2))
ax_dendrogram.set_axis_off()
dendrogram = scipy.cluster.hierarchy.dendrogram(
    cluster_f1, ax=ax_dendrogram, orientation="top", no_labels=True, link_color_func=lambda x: "black"
)

# horizontal dendrogram
panel, ax_dendrogram = fig.main[1, 0] = polyptich.grid.Panel((0.2, dim[1]))
ax_dendrogram.set_axis_off()
dendrogram = scipy.cluster.hierarchy.dendrogram(
    cluster_f1, ax=ax_dendrogram, orientation="left", no_labels=True, link_color_func=lambda x: "black"
)
ax_dendrogram.invert_yaxis()

panel, ax = fig.main.add(polyptich.grid.Panel(dim), padding_height = 0., padding_width = 0., column = 1, row = 1)

norm_f1 = mpl.colors.Normalize(vmin=0, vmax=.5)
cmap_f1 = mpl.cm.get_cmap("viridis")

norm_jaccard = mpl.colors.Normalize(vmin=0, vmax=.5)
cmap_jaccard = mpl.cm.get_cmap("magma")

plotdata = f1.values[np.argsort(design["ix"]), :][:, np.argsort(design["ix"])]
plotdata[np.triu_indices(plotdata.shape[0])] = np.nan
ax.matshow(plotdata, norm = norm_f1, cmap = cmap_f1)

plotdata = jaccard.values[np.argsort(design["ix"]), :][:, np.argsort(design["ix"])]
plotdata[np.tril_indices(plotdata.shape[0])] = np.nan
ax.matshow(plotdata, norm = norm_jaccard, cmap = cmap_jaccard)

ax.set_xticks([])
ax.set_xticklabels([])
# ax.set_yticks([])
# ax.set_yticklabels([])
# ax.set_xticks(design["ix"])
# ax.set_xticklabels(design["label"], rotation=90)
ax.set_yticks(design["ix"])
ax.set_yticklabels(design["label"], fontsize = 8)
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
ax.tick_params(axis='y', which='major', length=0, pad = 2.0)

for method, ticklabel in zip(design["method"], ax.get_yticklabels()):
    ticklabel.set_color(chdm.methods.differential_methods.loc[method]["color"])

# panel, ax = fig.main.add_right(polyptich.grid.Panel((0.1, dim[1])), padding = 0.1, row = 1)
# ax.matshow(jaccard.values[np.argsort(design["ix"]), :][:, np.argsort(design["ix"])].mean(0, keepdims = True).T, cmap = cmap_jaccard, norm = norm_jaccard)
# ax.set_xticks([0.])
# ax.set_xticklabels(["Mean"], rotation=90)
# ax.set_yticks([])

panel, ax = fig.main.add_under(polyptich.grid.Panel([panel.dim[1], 0.1]), padding = 0.5, column = 1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm_jaccard, cmap = cmap_jaccard), cax = ax, orientation = "horizontal", label = "Positional overlap (Jaccard)", extend = "max")
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(width = 0.5, pad = 0.)

# panel, ax = fig.main.add_under(polyptich.grid.Panel((dim[0], 0.1)), padding = 0.1, column = 1)
# ax.matshow(f1.values[np.argsort(design["ix"]), :][:, np.argsort(design["ix"])].mean(0, keepdims = True), cmap = cmap_f1, norm = norm_f1)
# ax.set_xticks([])
# ax.set_yticks([0])
# ax.set_yticklabels(["Mean"])

panel, ax = fig.main.add_under(polyptich.grid.Panel([dim[0], 0.1]), padding = 0.0, column = 1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm_f1, cmap = cmap_f1), cax = ax, orientation = "horizontal", label = "Region overlap (F1)", extend = "max")
cbar.ax.tick_params(width = 0.5, pad = 2.)

fig.plot()

manuscript.save_figure(fig, "3", "overlap")

# %%
