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
# dataset_name = "pbmc10kx"
# dataset_name = "pbmc10k_gran"
dataset_name = "pbmc10k"
# dataset_name = "hspc"
# dataset_name = "lymphoma"
# dataset_name = "liver"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

regions_name = "100k100k"
# regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot

# %%
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_immune_main"
# motifscan_name = "onek1k_gwas_specific"
# motifscan_name = "causaldb_immune"
# motifscan_name = "gtex_immune"
motifscan_name = "gtex_caviar_immune"
# motifscan_name = "gtex_caviar_immune_differential"
# motifscan_name = "gtex_caveman_immune_differential"

# motifscan_name = "gwas_hema_main"

# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_lymphoma_main"

# motifscan_name = "gwas_liver"
# motifscan_name = "gwas_liver_main"
# motifscan_name = "causaldb_liver"
# motifscan_name = "gtex_liver"
# motifscan_name = "gtex_caviar_liver"
# motifscan_name = "gtex_caviar_liver_differential"

associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

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


# %% [markdown]
# ## Enrichment

# %%
def determine_enrichment(clusterprobs_diff_bins, probs_mean_bins, fragments, regionpositional, step = 25):
    desired_x = np.arange(*regionpositional.regions.window, step=step) - regionpositional.regions.window[0]

    found = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
    tot = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
    for region_ix, region_id in tqdm.tqdm(zip(np.arange(len(fragments.var)), fragments.var.index)):
        # calculate differential accessibility landscape
        probs = regionpositional.probs[region_id]
        region_ix = regionpositional.regions.coordinates.index.get_indexer([region_id])[0]

        x_raw = probs.coords["coord"].values - regionpositional.regions.window[0]
        y_raw = probs.values

        y = chd.utils.interpolate_1d(
            torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
        ).numpy()
        ymean = y.mean(0)

        z = y - ymean
        zmax = np.abs(z.max(0))
        zmax = z.std(0)

        ybin = np.searchsorted(probs_mean_bins["cut"].values, ymean)
        zbin = np.searchsorted(clusterprobs_diff_bins["cut"].values, zmax)

        # get association
        positions, indices = associations.get_slice(region_ix, return_scores = False, return_strands = False)
        positions = positions - regionpositional.regions.window[0]
        ixs = positions // step

        # add bin counts
        tot += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin, minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
        found += np.bincount(ybin[ixs] * len(clusterprobs_diff_bins) + zbin[ixs], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))

    tot_reshaped = tot.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins)).T
    found_reshaped = found.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins)).T
    return found_reshaped, tot_reshaped


# %%
found, tot = determine_enrichment(clusterprobs_diff_bins, probs_mean_bins, fragments, regionpositional, step = 25)

# %%
contingencies = np.stack([
    tot,
    found.sum() - found,
    tot - found,
    found
]).T.reshape(-1, 2, 2)

# %%
odds = (((found)/(tot)) / (found.sum()/tot.sum()))
odds[odds == 0.] = 1.

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG
odds_max = 8
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(np.log(odds).T, cmap=cmap, norm=norm)
ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Odds ratio", extend="both", ticks=[np.log(1/odds_max), 0, np.log(odds_max)])
cbar.ax.set_yticklabels([f"1/{odds_max}", "1", f"{odds_max}"])

# fig.save_figure()

# manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment")

# %%
perc_contained = found / found.sum()

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))
plotdata = pd.DataFrame(np.log(odds), columns = pd.Index(probs_mean_bins["cut"], name = "mean"), index = pd.Index(clusterprobs_diff_bins["cut"], name = "diff")).stack(dropna = False).reset_index(name = "odds")
plotdata["perc_contained"] = perc_contained.flatten()
plotdata["x"] = pd.Index(clusterprobs_diff_bins["cut"]).get_indexer(plotdata["diff"])
plotdata["y"] = pd.Index(probs_mean_bins["cut"]).get_indexer(plotdata["mean"])

norm_size = mpl.colors.Normalize(vmin=0, vmax=0.01, clip=True)

for _, plotdata_row in plotdata.iterrows():
    size = np.sqrt(norm_size(plotdata_row["perc_contained"]))
    rect = mpl.patches.Rectangle((plotdata_row.x-size/2, plotdata_row.y-size/2), size, size, linewidth=0, facecolor=cmap(norm(plotdata_row["odds"])))
    ax.add_patch(rect)
    # ax.scatter(plotdata_row.x, plotdata_row.y, color = cmap(norm(plotdata_row["odds"])))
ax.invert_yaxis()
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])
ax.set_xlim(-0.5, len(clusterprobs_diff_bins)-0.5)
ax.set_ylim(len(probs_mean_bins)-0.5, -0.5)
ax.set_aspect(1)

# %%
sns.heatmap(np.log(tot+1).T)

# %% [markdown]
# ## Differential enrichment

# %%
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "logreg" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test" / "scoring" / "regionpositional"
scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_500" / "t-test" / "scoring" / "regionpositional"
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
step = 25

# %%
slices_peak["start_bin"] = (slices_peak["start"] - fragments.regions.window[0]) // step
slices_peak["end_bin"] = (slices_peak["end"] - fragments.regions.window[0]) // step

# %%
scores = []

desired_x = np.arange(*regionpositional.regions.window, step=step) - regionpositional.regions.window[0]

found_peak = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
found_nonpeak = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
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
    positions, indices = associations.get_slice(region_ix, return_scores = False, return_strands = False)
    positions = positions - regionpositional.regions.window[0]
    ixs = positions // step

    # add bin counts
    tot += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin, minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
    found += np.bincount(ybin[ixs] * len(clusterprobs_diff_bins) + zbin[ixs], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))

    # get association
    ixs_peak = np.concatenate([np.arange(subregion["start_bin"], subregion["end_bin"]) for _, subregion in subregions.iterrows()])
    ixs_found = ixs[(ixs[None, :] == ixs_peak[:, None]).any(0)]
    ixs_notfound = np.setdiff1d(ixs, ixs_found)

    # add bin counts
    tot += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin, minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
    found_peak += np.bincount(ybin[ixs_found] * len(clusterprobs_diff_bins) + zbin[ixs_found], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
    found_nonpeak += np.bincount(ybin[ixs_notfound] * len(clusterprobs_diff_bins) + zbin[ixs_notfound], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))

tot = tot.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins))
found_peak = found_peak.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins))
found_nonpeak = found_nonpeak.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins))

# %%
odds_nonpeak = (((found_peak)/(tot)) / (found_peak.sum()/tot.sum()))
odds_peak = (((found_peak+found_nonpeak)/(tot)) / ((found_peak+found_nonpeak).sum()/tot.sum()))
# odds[odds == 0.] = 1.

# %%
sns.heatmap(odds_nonpeak/odds_peak, vmax = 4)

# %%
sns.heatmap(odds_nonpeak, vmax = 10)

# %%
sns.heatmap(odds_peak, vmax = 10)


# %%
def determine_enrichment(clusterprobs_diff_bins, probs_mean_bins, fragments, regionpositional, step = 25):
    desired_x = np.arange(*regionpositional.regions.window, step=step) - regionpositional.regions.window[0]

    found = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
    tot = np.zeros(len(clusterprobs_diff_bins) * len(probs_mean_bins), dtype=int)
    for region_ix, region_id in tqdm.tqdm(zip(np.arange(len(fragments.var)), fragments.var.index)):
        # calculate differential accessibility landscape
        probs = regionpositional.probs[region_id]
        region_ix = regionpositional.regions.coordinates.index.get_indexer([region_id])[0]

        x_raw = probs.coords["coord"].values - regionpositional.regions.window[0]
        y_raw = probs.values

        y = chd.utils.interpolate_1d(
            torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
        ).numpy()
        ymean = y.mean(0)

        z = y - ymean
        zmax = np.abs(z.max(0))
        zmax = z.std(0)

        ybin = np.searchsorted(probs_mean_bins["cut"].values, ymean)
        zbin = np.searchsorted(clusterprobs_diff_bins["cut"].values, zmax)

        # get association
        positions, indices = associations.get_slice(region_ix, return_scores = False, return_strands = False)
        positions = positions - regionpositional.regions.window[0]
        ixs = positions // step

        # add bin counts
        tot += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin, minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
        found += np.bincount(ybin[ixs] * len(clusterprobs_diff_bins) + zbin[ixs], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))

    tot_reshaped = tot.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins)).T
    found_reshaped = found.reshape(len(probs_mean_bins), len(clusterprobs_diff_bins)).T
    return found_reshaped, tot_reshaped


# %% [markdown]
# ## Enrichment multiple

# %%
from chromatinhd_manuscript.designs_qtl import design
design = design.query("regions == '100k100k'").query("dataset in ['pbmc10k_gran', 'pbmc10k', 'hspc', 'lymphoma', 'liver']")
output_folder = pathlib.Path("./natural_variation/")
output_folder.mkdir(exist_ok=True, parents=True)
# # !rm -r {output_folder}

# %%
for design_ix, row in design.iterrows():
    dataset_name = row["dataset"]
    motifscan_name = row["motifscan"]

    latent = "leiden_0.1"
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

    regions_name = "100k100k"
    # regions_name = "10k10k"
    fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
    regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

    associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

    file = output_folder / f"{dataset_name}_{regions_name}_{motifscan_name}.pkl"

    if not file.exists():
        print(file)
        found, tot = determine_enrichment(clusterprobs_diff_bins, probs_mean_bins, fragments, regionpositional, step = 25)
        with open(file, "wb") as f:
            pickle.dump((found, tot), f)

# %%
log_oddss = []

founds = []
tots =[]

name = "gwas"
design_oi = design.loc[(design["motifscan"].str.contains("gwas") & design["motifscan"].str.contains("main")) | (design["motifscan"].str.contains("causaldb"))]

# name = "gwas_pbmc10k"
# name = "gwas_liver"
# name = "gwas_lymphoma"
# design_oi = design.loc[(design["motifscan"].str.contains("gwas") & design["motifscan"].str.contains("main")) | (design["motifscan"].str.contains("causaldb")) | (design["motifscan"].str.contains("gtex") & design["motifscan"].str.contains("differential"))
# ].query("dataset == 'lymphoma'")

name = "eqtl_all"
design_oi = design.loc[design["motifscan"].str.contains("gtex") & ~design["motifscan"].str.contains("differential")]

# name = "eqtl_differential"
# design_oi = design.loc[design["motifscan"].str.contains("gtex") & design["motifscan"].str.contains("differential")]

name = "eqtl_finemapped"
design_oi = design.loc[design["motifscan"].str.contains("gtex") & ~design["motifscan"].str.contains("differential") & (design["motifscan"].str.contains("caviar") | design["motifscan"].str.contains("caveman"))]


for design_ix, row in design_oi.iterrows():
    dataset_name = row["dataset"]
    motifscan_name = row["motifscan"]
    file = output_folder / f"{dataset_name}_{regions_name}_{motifscan_name}.pkl"

    if file.exists():
        print(motifscan_name)
        with open(file, "rb") as f:
            found, tot = pickle.load(f)
        founds.append(found)
        tots.append(tot)
        # pseudocount = 0.
        odds = (((found)/(tot)) / (found.sum()/tot.sum()))
        odds[odds == 0.] = np.nan
        log_oddss.append(np.log(odds).T)

# %%
founds_stacked = np.stack(founds, -1)
tots_stacked = np.stack(tots, -1)

weights = 1.
# weights = 1/founds.sum(-1).sum(-1)
# weights = weights/weights.sum() * len(weights)

found = (founds_stacked*weights).sum(-1)
tot = (tots_stacked*weights).sum(-1)

log_odds = odds = np.log(((found/tot) / (found.sum()/tot.sum()))).T
log_odds[np.isinf(log_odds)] = 0.
sns.heatmap(log_odds)

# %%
plotdata = np.nanmean(np.stack(log_oddss), axis = 0)
# plotdata = log_odds

fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG
odds_max = 8.
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(plotdata, cmap=cmap, norm=norm)

ax.set_aspect(1)

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Odds ratio", extend="both", ticks=[np.log(1/odds_max), 0, np.log(odds_max)], orientation = "horizontal")
# cbar.ax.set_xticklabels([f"1/{odds_max}", "1", f"{odds_max}"])

ax.set_ylabel("")
ax.set_yticks(np.arange(1, len(probs_mean_bins))-0.5)
ax.set_yticklabels([])

ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, len(clusterprobs_diff_bins)) - 0.5, minor=True)
ax.set_xticklabels(np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1], minor=True, rotation = 90)
ax.set_xticks([])
ax.set_xticklabels([])

sns.despine(fig, ax)

manuscript.save_figure(fig, "4", f"mean_vs_variance_{name}")

# %%
fig_colorbar = plt.figure(figsize=(2.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=norm, cmap=cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal", ticks=[np.log(1/odds_max), 0, np.log(odds_max)], extend = "both")
ax_colorbar.set_xticklabels([f"1/{odds_max:.0f}", "1", f"{odds_max:.0f}"])
colorbar.set_label("Odds ratio")
manuscript.save_figure(fig_colorbar, "4", "colorbar_odds_qtl")

# %% [markdown]
# ## Cell-type specific in GWAS enrichment

# %%
dataset_name = "pbmc10k"

# %%
motifscan_name = "gtex_caviar_immune"
associations_gtex_general = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
snps_gtex_general = associations_gtex_general.association["chr"] + "_" + associations_gtex_general.association["pos"].astype(str)

motifscan_name = "gtex_caviar_immune_differential"
associations_gtex_differential = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
snps_gtex_differential = associations_gtex_differential.association["chr"] + "_" + associations_gtex_differential.association["pos"].astype(str)

motifscan_name = "gwas_immune"
associations_gwas = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
snps_gwas = associations_gwas.association["chr"] + "_" + associations_gwas.association["pos"].astype(str)

# %%
(snps_gtex_general.isin(snps_gwas).sum()/len(snps_gtex_general))/(snps_gtex_differential.isin(snps_gwas).sum()/len(snps_gtex_differential))

# %% [markdown]
# ## Diff vs distance to max

# %%
bandwidth = 4000

dist_bins = pd.DataFrame(
    {"cut":[*np.linspace(0, bandwidth//2, 5)[1:-1], np.inf]}
)
dist_bins["label"] = dist_bins["cut"].astype(str)

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": np.log(np.array([1.5, 2, 2.5, 3, 3.5, 4, 8, np.inf]))}
)
clusterprobs_diff_bins["label"] = ["<1.5"]+["","≥2","","≥3","","≥4","≥8"]
clusterprobs_diff_bins = pd.DataFrame(
    {"cut": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.inf]}
)
clusterprobs_diff_bins["label"] = clusterprobs_diff_bins["cut"]


# %%
def rolling_argmax(x, bandwidth=25):
    left = bandwidth // 2
    right = bandwidth - left
    idxs = []
    for i in range(0, len(x)):
        l, r = max(i-left, 0),min(i+right, len(x))
        idxs.append(l + np.argmax(x[l:r]))
    return np.array(idxs)


# %%
found = np.zeros(len(clusterprobs_diff_bins) * len(dist_bins), dtype=int)
tot = np.zeros(len(clusterprobs_diff_bins) * len(dist_bins), dtype=int)
for region_ix, region_id in tqdm.tqdm(zip(np.arange(len(fragments.var)), fragments.var.index)):
    # calculate differential accessibility landscape
    probs = regionpositional.probs[region_id]
    region_ix = self.regions.coordinates.index.get_indexer([region_id])[0]

    x_raw = probs.coords["coord"].values - self.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
    ).numpy()
    ymean = y.mean(0)
    ydist = np.abs(np.arange(len(ymean)) - rolling_argmax(ymean, bandwidth = bandwidth // step)) * step

    z = y - ymean
    zmax = np.abs(z.max(0))
    zmax = z.std(0)

    ybin = np.searchsorted(dist_bins["cut"].values, ydist)
    zbin = np.searchsorted(clusterprobs_diff_bins["cut"].values, zmax)

    # get association
    positions, indices = associations.get_slice(region_ix, return_scores = False, return_strands = False)
    positions = positions - self.regions.window[0]
    ixs = positions // step

    # add bin counts
    tot += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin, minlength=len(clusterprobs_diff_bins) * len(dist_bins))
    found += np.bincount(ybin[ixs] * len(clusterprobs_diff_bins) + zbin[ixs], minlength=len(clusterprobs_diff_bins) * len(dist_bins))

# %%
tot_reshaped = tot.reshape(len(dist_bins), len(clusterprobs_diff_bins)).T
found_reshaped = found.reshape(len(dist_bins), len(clusterprobs_diff_bins)).T

# %%
contingencies = np.stack([
    tot_reshaped,
    found.sum() - found_reshaped,
    tot_reshaped - found_reshaped,
    found_reshaped
]).T.reshape(-1, 2, 2)

# %%
odds = (((found_reshaped)/(tot_reshaped)) / (found.sum()/tot.sum()))
# odds = (found_reshaped/found.sum() + 1e-5) * (tot.sum() / tot_reshaped)
# odds = ((found_reshaped/tot_reshaped+1e-2) / (found.sum()/tot.sum()+1e-2))
odds[odds == 0.] = 1.

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG
odds_max = 8
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(np.log(odds).T, cmap=cmap, norm=norm)
ax.set_ylabel("Distance to summit")
ax.set_yticks(np.arange(len(dist_bins)))
ax.set_yticklabels(dist_bins["label"])

ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Odds ratio", extend="both", ticks=[np.log(1/odds_max), 0, np.log(odds_max)])
cbar.ax.set_yticklabels([f"1/{odds_max}", "1", f"{odds_max}"])

# fig.save_figure()

# manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment")

# %%
perc_contained = found_reshaped / found_reshaped.sum()

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))
plotdata = pd.DataFrame(np.log(odds), columns = pd.Index(dist_bins["cut"], name = "mean"), index = pd.Index(clusterprobs_diff_bins["cut"], name = "diff")).stack(dropna = False).reset_index(name = "odds")
plotdata["perc_contained"] = perc_contained.flatten()
plotdata["x"] = pd.Index(clusterprobs_diff_bins["cut"]).get_indexer(plotdata["diff"])
plotdata["y"] = pd.Index(dist_bins["cut"]).get_indexer(plotdata["mean"])

norm_size = mpl.colors.Normalize(vmin=0, vmax=0.01, clip=True)

for _, plotdata_row in plotdata.iterrows():
    size = norm_size(plotdata_row["perc_contained"])
    rect = mpl.patches.Rectangle((plotdata_row.x-size/2, plotdata_row.y-size/2), size, size, linewidth=0, facecolor=cmap(norm(plotdata_row["odds"])))
    ax.add_patch(rect)
    # ax.scatter(plotdata_row.x, plotdata_row.y, color = cmap(norm(plotdata_row["odds"])))
ax.invert_yaxis()
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])
ax.set_yticks(np.arange(len(dist_bins)))
ax.set_yticklabels(dist_bins["label"])
ax.set_xlim(-0.5, len(clusterprobs_diff_bins)-0.5)
ax.set_ylim(len(dist_bins)-0.5, -0.5)
ax.set_aspect(1)

# %%
sns.heatmap(np.log(tot_reshaped+1).T)

# %% [markdown]
# ## Diff vs distance to max 2

# %%
bandwidth = 5000

dist_bins = pd.DataFrame(
    {"cut":[*np.linspace(0, bandwidth//2, 10)[1:-1], np.inf]}
)
dist_bins["label"] = dist_bins["cut"].astype(str)

max_bins = pd.DataFrame(
    {"cut_exp":[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., np.inf]}
)
max_bins["cut"] = np.log(max_bins["cut_exp"])
max_bins["label"] = ["<" + str(max_bins["cut_exp"][0])] + ["≥" + str(x) for x in max_bins["cut_exp"].astype(str)[:-1]]


# %%
def rolling_argmax(x, bandwidth=25):
    left = bandwidth // 2
    right = bandwidth - left
    idxs = []
    maxs = []
    for i in range(0, len(x)):
        l, r = max(i-left, 0),min(i+right, len(x))
        idxs.append(l + np.argmax(x[l:r]))
    return np.array(idxs), x[idxs]


# %%
found = np.zeros(len(max_bins) * len(dist_bins), dtype=int)
tot = np.zeros(len(max_bins) * len(dist_bins), dtype=int)
for region_ix, region_id in tqdm.tqdm(zip(np.arange(len(fragments.var)), fragments.var.index)):
    # calculate differential accessibility landscape
    probs = regionpositional.probs[region_id]
    region_ix = fragments.regions.coordinates.index.get_indexer([region_id])[0]

    x_raw = probs.coords["coord"].values - fragments.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
    ).numpy()
    ymean = y.mean(0)

    z = y - ymean
    zmax = z.std(0)

    # selected = zmax > 1.0
    # selected = ymean > -4

    selected = (zmax > 0.5) & (ymean > -3)

    ydist, ymax = rolling_argmax(ymean, bandwidth = bandwidth // step)
    ydist = np.abs(np.arange(len(ymean)) - ydist) * step

    ybin = np.searchsorted(dist_bins["cut"].values, ydist)
    zbin = np.searchsorted(max_bins["cut"].values, ymax)

    # get association
    positions, indices = associations.get_slice(region_ix, return_scores = False, return_strands = False)
    positions = positions - fragments.regions.window[0]
    ixs = positions // step
    ixs_selected = ixs[selected[ixs]]

    # add bin counts
    tot += np.bincount(ybin * len(max_bins) + zbin, minlength=len(max_bins) * len(dist_bins))
    # tot += np.bincount(ybin[selected] * len(max_bins) + zbin[selected], minlength=len(max_bins) * len(dist_bins))
    found += np.bincount(ybin[ixs_selected] * len(max_bins) + zbin[ixs_selected], minlength=len(max_bins) * len(dist_bins))

    # break

# %%
plt.plot(ymean[:5000])

# %%
plt.plot(ydist[:1000])

# %%
tot_reshaped = tot.reshape(len(dist_bins), len(max_bins)).T
found_reshaped = found.reshape(len(dist_bins), len(max_bins)).T

# %%
contingencies = np.stack([
    tot_reshaped,
    found.sum() - found_reshaped,
    tot_reshaped - found_reshaped,
    found_reshaped
]).T.reshape(-1, 2, 2)

# %%
odds = (((found_reshaped)/(tot_reshaped)) / (found.sum()/tot.sum()))
# odds = (found_reshaped/found.sum() + 1e-5) * (tot.sum() / tot_reshaped)
# odds = ((found_reshaped/tot_reshaped+1e-2) / (found.sum()/tot.sum()+1e-2))
odds[odds == 0.] = 1.

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG
odds_max = 8
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(np.log(odds).T, cmap=cmap, norm=norm)
ax.set_ylabel("Distance to summit")
ax.set_yticks(np.arange(len(dist_bins)))
ax.set_yticklabels(dist_bins["label"])

ax.set_xlabel("Height of nearby summit")
ax.set_xticks(np.arange(len(max_bins)))
ax.set_xticklabels(max_bins["label"])

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Odds ratio", extend="both", ticks=[np.log(1/odds_max), 0, np.log(odds_max)])
cbar.ax.set_yticklabels([f"1/{odds_max}", "1", f"{odds_max}"])

# fig.save_figure()

# manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment")

# %%
sns.heatmap(np.log(tot_reshaped+1).T)

# %%
