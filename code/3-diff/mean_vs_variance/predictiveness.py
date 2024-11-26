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
import tempfile

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

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
dataset_name = "pbmc10k"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")
splitter = "5x1"
model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / splitter / "magic" / "v33"
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    model_folder / "scoring" / "regionmultiwindow",
)

# %%
scored = regionmultiwindow.scores["scored"].sel_xr().all("fold")
regions_oi = scored.coords["gene"][scored].values

# %%
window_size = 50
windows_oi = regionmultiwindow.design.loc[regionmultiwindow.design["window_size"] == window_size].index
design = regionmultiwindow.design.loc[windows_oi]
assert len(design) > 0

joined_all = []
for gene_oi in tqdm.tqdm(regions_oi):
    joined = (
        regionmultiwindow.scores.sel_xr(
            (gene_oi, slice(None), "test"), variables=["deltacor"]
        )
        .sel(window=design.index)
        .mean("fold")
    ).to_pandas()
    joined["gene"] = gene_oi
    joined_all.append(joined)

# %%
probs_mean_bins = pd.DataFrame(
    {"cut_exp":[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., np.inf]}
)
probs_mean_bins["cut"] = np.log(probs_mean_bins["cut_exp"])
probs_mean_bins["label"] = ["<" + str(probs_mean_bins["cut_exp"][0])] + ["≥" + str(x) for x in probs_mean_bins["cut_exp"].astype(str)[:-1]]

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": np.log(np.array([1.5, 2, 2.5, 3, 3.5, 4, 8, np.inf]))}
)
clusterprobs_diff_bins["label"] = ["<1.5"]+["","≥2","","≥3","","≥4","≥8"]
clusterprobs_diff_bins = pd.DataFrame(
    {"cut": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.inf]}
)
clusterprobs_diff_bins["label"] = clusterprobs_diff_bins["cut"]

# %%
regions = fragments.regions

# %%
scores = []

for region_id in tqdm.tqdm(regions_oi[:1000]):
    # calculate predictivity metrics
    joined = (
            regionmultiwindow.scores.sel_xr(
                (region_id, slice(None), "test"), variables=["deltacor", "lost"]
            )
            .sel(window=design.index)
            .mean("fold")
        ).to_pandas()

    # calculate diff metrics
    y = regionpositional.get_interpolated(region_id, desired_x = design["window_mid"].values)

    ymean = y.mean(0)
    z = y - ymean
    zmax = z.std(0)

    # add chd diff metrics to data binned
    joined["clusterprobs_diff"] = zmax
    joined["clusterprobs_diff_bin"] = np.digitize(joined["clusterprobs_diff"], clusterprobs_diff_bins["cut"])
    joined["probs_mean"] = ymean
    joined["probs_mean_bin"] = np.digitize(joined["probs_mean"], probs_mean_bins["cut"])

    joined["window_mid"] = design["window_mid"].values

    scores.append(joined.assign(region_id=region_id))
scores = pd.concat(scores)

# %%
fig, ax = plt.subplots()
ax.plot(joined["window_mid"], joined["clusterprobs_diff"])
ax2 = ax.twinx()
ax2.plot(joined["window_mid"], -joined["deltacor"], color = 'red')

# %%
scores["significant"] = scores["deltacor"] < -1e-5
plotdata = scores.dropna().groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["significant"].mean().unstack().T
plotdata = scores.dropna().groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["deltacor"].mean().unstack().T
sns.heatmap(
    plotdata,
    cmap="RdBu_r",
    center=0,
)

# %%
sns.heatmap(np.log(scores.groupby(["clusterprobs_diff_bin", "probs_mean_bin"]).size().unstack().T))
