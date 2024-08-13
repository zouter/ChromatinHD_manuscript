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
# %load_ext autoreload
# %autoreload 2

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

regions_name = "100k100k"
# regions_name = "10k10k"
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
# ## Positional

# %%
design_ix1 = design.index[(
    # (design["peakcaller"] == "macs2_summits") &
    # (design["peakcaller"] == "encode_screen") &
    (design["peakcaller"] == "macs2_leiden_0.1_merged") &
    # (design["peakcaller"] == "cellranger") &
    (design["diffexp"] == "t-test")
)
][0]

design_ix2 = design.index[(
    (design["method"] == "chd")
)
][0]

# %%
bed1 = prsets[design_ix1]
bed2 = prsets[design_ix2]

# %%
intersect = bed1.intersect(bed2)

# %%
overlaps = np.zeros(fragments.regions.window[1] - fragments.regions.window[0])
unique1 = np.zeros(fragments.regions.window[1] - fragments.regions.window[0])
unique2 = np.zeros(fragments.regions.window[1] - fragments.regions.window[0])

for start, end in intersect.to_dataframe()[["start", "end"]].values:
    overlaps[start:end] += 1
    unique1[start:end] -= 1
    unique2[start:end] -= 1

for start, end in bed1.to_dataframe()[["start", "end"]].values:
    unique1[start:end] += 1

for start, end in bed2.to_dataframe()[["start", "end"]].values:
    unique2[start:end] += 1

tot = overlaps + unique1 + unique2

# %%
plotdata = pd.DataFrame({
    "overlaps": overlaps,
    "unique1": unique1,
    "unique2": unique2,
    # "tot": tot,
})
# plotdata = plotdata / plotdata.values.sum(1, keepdims = True)

# %%
fig, ax = plt.subplots()
ax.fill_between(
    np.arange(len(plotdata)),
    plotdata["unique1"],
    color = "black",
    label = "Unique1"
)
ax.fill_between(
    np.arange(len(plotdata)),
    plotdata["unique1"],
    plotdata["overlaps"] + plotdata["unique1"],
    color = "red",
    label = "Unique 1"
)
ax.fill_between(
    np.arange(len(plotdata)),
    plotdata["overlaps"] + plotdata["unique1"],
    plotdata["overlaps"] + plotdata["unique1"] + plotdata["unique2"],
    color = "blue",
    label = "Unique 2"
)
ax.set_xlim(-100000 - fragments.regions.window[0], 100000 - fragments.regions.window[0])

# %% [markdown]
# ## Overlap between "summits"

# %%
import chromatinhd.data.peakcounts
peakcounts = chd.data.peakcounts.PeakCounts(chd.get_output() / "datasets" / dataset_name / "peakcounts" / "macs2_leiden_0.1_merged" / "100k100k")
# peakcounts = chd.data.peakcounts.PeakCounts(chd.get_output() / "datasets" / dataset_name / "peakcounts" / "macs2_summits" / "100k100k")

# %%
window = [-5000, 5000]

# %%
peakcounts.peaks["region_ix"] = peakcounts.peaks["gene_ix"]
peaks = peakcounts.peaks.copy()

# %%
peaks["mid"] = (peaks["relative_start"] + peaks["relative_end"]) // 2  - fragments.regions.window[0]

# %%
peakcounts.peaks["region_ix"] = peakcounts.peaks["gene_ix"]
peaks = peakcounts.peaks.copy()
peaks["mid"] = (peaks["relative_start"] + peaks["relative_end"]) // 2  - fragments.regions.window[0]
peaks["start"] = np.clip(peaks["mid"] + window[0], 0, None)
peaks["end"] = np.clip(peaks["mid"] + window[1], 0, None)
peaks["name"] = peaks["region_ix"].astype(str) + ":" + peaks["start"].astype(str) + "-" + peaks["end"].astype(str)
peaks.index = peaks["name"]
import pybedtools
peaks_bed = pybedtools.BedTool.from_dataframe(peaks[["region_ix", "start", "end", "name"]]).sort()

# %%
slices = slicesets[design_ix1][["region_ix", "start", "end"]].copy()

slices["region_ix"] = slices["region_ix"].astype(int)
slices["start"] = slices["start"].astype(int) - fragments.regions.window[0]
slices["end"] = slices["end"].astype(int) - fragments.regions.window[0]
diffpeaks_bed = pybedtools.BedTool.from_dataframe(slices[["region_ix", "start", "end"]]).sort()
linkage1 = diffpeaks_bed.intersect(peaks_bed, wb = True)

linkage1 = linkage1.to_dataframe().copy()
linkage1["peak_ix"] = peaks.index.get_indexer(linkage1["thickStart"])
linkage1["actual_start"] = linkage1["start"] - peaks["mid"].loc[linkage1["thickStart"]].values-window[0]
linkage1["actual_end"] = linkage1["end"] - peaks["mid"].loc[linkage1["thickStart"]].values-window[0]

# %%
slices = slicesets[design_ix2][["region_ix", "start", "end"]].copy()

slices["region_ix"] = slices["region_ix"].astype(int)
slices["start"] = slices["start"].astype(int) - fragments.regions.window[0]
slices["end"] = slices["end"].astype(int) - fragments.regions.window[0]
diffpeaks_bed = pybedtools.BedTool.from_dataframe(slices[["region_ix", "start", "end"]]).sort()
linkage2 = diffpeaks_bed.intersect(peaks_bed, wb = True)

linkage2 = linkage2.to_dataframe().copy()
linkage2["peak_ix"] = peaks.index.get_indexer(linkage2["thickStart"])
linkage2["actual_start"] = linkage2["start"] - peaks["mid"].loc[linkage2["thickStart"]].values-window[0]
linkage2["actual_end"] = linkage2["end"] - peaks["mid"].loc[linkage2["thickStart"]].values-window[0]

# %%
bed1 = pybedtools.BedTool.from_dataframe(linkage1[["peak_ix", "actual_start", "actual_end"]].rename(columns = {"peak_ix":"Chromosome", "actual_start":"Start", "actual_end":"End"}))
bed2 = pybedtools.BedTool.from_dataframe(linkage2[["peak_ix", "actual_start", "actual_end"]].rename(columns = {"peak_ix":"Chromosome", "actual_start":"Start", "actual_end":"End"}))
bed_intersect = bed1.intersect(bed2)

# %%
overlaps = np.zeros(window[1] - window[0])
unique1 = np.zeros(window[1] - window[0])
unique2 = np.zeros(window[1] - window[0])

for start, end in bed_intersect.to_dataframe()[["start", "end"]].values:
    overlaps[start:end] += 1
    unique1[start:end] -= 1
    unique2[start:end] -= 1

for start, end in bed1.to_dataframe()[["start", "end"]].values:
    unique1[start:end] += 1

for start, end in bed2.to_dataframe()[["start", "end"]].values:
    unique2[start:end] += 1

# %%
overlaps.sum() / (overlaps.sum() + unique1.sum() + unique2.sum())

# %%
plotdata = pd.DataFrame({
    "overlaps": overlaps,
    "unique1": unique1,
    "unique2": unique2,
})
# plotdata = plotdata / plotdata.values.sum(1, keepdims = True)

# %%
fig, ax = plt.subplots()
plt.plot(np.arange(*window), unique1)
plt.plot(np.arange(*window), unique2)
plt.plot(np.arange(*window), overlaps)
ax.axvline(0, color = "black", dashes = (2, 2))

# %%
fig, ax = plt.subplots()
ax.fill_between(
    np.arange(len(plotdata)),
    plotdata["unique1"],
    color = "black",
    label = "Unique1"
)
ax.fill_between(
    np.arange(len(plotdata)),
    plotdata["unique1"],
    plotdata["overlaps"] + plotdata["unique1"],
    color = "red",
    label = "Unique 1"
)
ax.fill_between(
    np.arange(len(plotdata)),
    plotdata["overlaps"] + plotdata["unique1"],
    plotdata["overlaps"] + plotdata["unique1"] + plotdata["unique2"],
    color = "blue",
    label = "Unique 2"
)

# %%
