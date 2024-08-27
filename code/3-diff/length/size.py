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
import tempfile

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %% [markdown]
# ## Choose bin sizes

# %%
length_bins = pd.DataFrame([
    [20, "<20bp"],
    [50, "≥20bp"],
    [100, "≥50bp"],
    [200, "≥100bp"],
    [500, "≥200bp"],
    [1000, "≥500bb"],
    [np.inf, "≥1kb"],
], columns = ["cut", "label"])
length_bins["ix"] = np.arange(len(length_bins))
length_bins.index = pd.Index(length_bins["ix"], name="bin")

# %% [markdown]
# ## Select slices for QTL and CRISPR

# %%
# dataset_name = "pbmc10k"
dataset_name = "hspc"
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
def uncenter_peaks(slices, coordinates):
    if "region_ix" not in slices.columns:
        slices["region_ix"] = coordinates.index.get_indexer(slices["region"])
    coordinates_oi = coordinates.iloc[slices["region_ix"]].copy()

    slices["chrom"] = coordinates_oi["chrom"].values

    slices["start_genome"] = np.where(
        coordinates_oi["strand"] == 1,
        (slices["start"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
        (slices["end"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
    )
    slices["end_genome"] = np.where(
        coordinates_oi["strand"] == 1,
        (slices["end"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
        (slices["start"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
    )
    return slices

coordinates = fragments.regions.coordinates

# %%
slices = regionpositional.calculate_slices(-1., step = 5)
top_slices = regionpositional.calculate_top_slices(slices, 1.5)

slicescores = top_slices.get_slice_scores(regions = fragments.regions)
coordinates = fragments.regions.coordinates
slices = uncenter_peaks(slicescores, fragments.regions.coordinates)
slicescores["slice"] = pd.Categorical(slicescores["chrom"].astype(str) + ":" + slicescores["start_genome"].astype(str) + "-" + slicescores["end_genome"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome"]].first()

# %%
slices["length"] = slices["end"] - slices["start"]
slices["length_bin"] = np.searchsorted(length_bins["cut"], slices["length"])

# %%
fig, ax = plt.subplots()
slices["length"].plot.hist(bins = np.linspace(0, 1000, 50))

# %%
length_scores = pd.DataFrame({
    "n_positions":slices.groupby("length_bin")["length"].sum(),
    "n_slices":slices.groupby("length_bin")["length"].count(),
})

# %%
length_scores.style.bar()

# %% [markdown]
# ## SNP enrichment

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot

# %%
motifscan_names = [
    "gwas_immune_main",
    # "causaldb_immune",
    "gtex_caviar_immune",
    # "gtex_caveman_immune_differential",
]

# %%
scores = []

from scipy.stats import fisher_exact

for motifscan_name in motifscan_names:
    associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
    association = associations.association
    association["start"] = (association["pos"]).astype(int)
    association["end"] = (association["pos"] + 1).astype(int)

    for length_bin, slices_length in slices.groupby("length_bin"):
        import pyranges
        pr = pyranges.PyRanges(
            slices_length[["chrom", "start_genome", "end_genome"]].rename(
                columns={"chrom": "Chromosome", "start_genome": "Start", "end_genome": "End"}
            )
        ).merge()
        pr = pr.sort()

        pr_snps = pyranges.PyRanges(association.reset_index()[["chr", "start", "end", "index"]].rename(columns = {"chr":"Chromosome", "start":"Start", "end":"End"}))
        overlap = pr_snps.intersect(pr)

        haplotype_scores = association[["snp", "disease/trait"]].copy()
        if len(overlap) > 0:
            haplotype_scores["n_matched"] = (haplotype_scores.index.isin(overlap.as_df()["index"])).astype(int)
        else:
            haplotype_scores["n_matched"] = 0
        haplotype_scores["n_total"] = 1

        matched = haplotype_scores["n_matched"].sum()
        total_snps = haplotype_scores["n_total"].sum()
        total_diff = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
        total_positions = fragments.regions.width * fragments.n_regions

        contingency = pd.DataFrame([
            [matched, total_snps - matched],
            [total_diff - matched, total_positions - total_snps - total_diff + matched]
        ], index = ["SNP", "Not SNP"], columns = ["In slice", "Not in slice"])
        contingency

        fisher = (fisher_exact(contingency))

        scores.append({
            "motifscan": motifscan_name,
            "length_bin": length_bin,
            "matched": matched,
            "total_snps": total_snps,
            "total_diff": total_diff,
            "total_positions": total_positions,
            "fisher": fisher,
            "odds": fisher[0],
        })
scores = pd.DataFrame(scores)
scores["motifscan"] = pd.Categorical(scores["motifscan"], motifscan_names)

# %%
scores.style.bar()

# %% [markdown]
# ## CRISPRi score

# %%
folder = chd.get_output() / "data" / "crispri" / "fulco_2019"
data = pd.read_table(folder / "data.tsv", sep="\t")

data["region_ix"] = pd.Index(transcriptome.var["symbol"]).get_indexer(data["Gene"])
data = data[data["region_ix"] >= 0]

# %%
slicescores = slices.copy()
slicescores["score"] = np.nan
for region_ix, data_region in tqdm.tqdm(data.groupby("region_ix")):
    slices_oi = slices[slices["region_ix"] == region_ix].copy()

    for _, slice in slices_oi.iterrows():
        data_oi = data_region[
            (data_region["start"] >= slice["start_genome"]) &
            (data_region["end"] <= slice["end_genome"])
        ]
        slicescores.loc[_, "score"] = data_oi["HS_LS_logratio"].mean()

# %%
crispr_scores = slicescores.dropna().groupby("length_bin")[["score"]].mean()
crispr_scores = crispr_scores.reindex(length_bins.index)
crispr_scores

# %%
slicescores.groupby("length_bin")["score"].mean().plot()

# %% [markdown]
# ## Motifs

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
# slices = regionpositional.calculate_slices(0., step = 25)
slices = regionpositional.calculate_slices(-1, step = 5)
# slices = regionpositional.calculate_slices(-2, step = 25)
differential_slices = regionpositional.calculate_differential_slices(slices, fc_cutoff = 1.5)

slicescores = differential_slices.get_slice_scores(regions = fragments.regions, clustering = clustering)

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

motifscan.motifs["label"] = motifscan.motifs["HUMAN_gene_symbol"]
clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
slicecounts = motifscan.count_slices(slices)

# %%
slices["length"] = slices["end"] - slices["start"]
slices["length_bin"] = np.searchsorted(length_bins["cut"], slices["length"])

# %%
motifs = motifscan.motifs
motifclustermapping = chdm.motifclustermapping.get_motifclustermapping(dataset_name, motifscan, clustering)

# %% [markdown]
# ### Motifs across clusters

# %%
motif_scores = []
motif_scores_all = []
for length_bin, slices_length in slices.groupby("length_bin"):
    print(len(slices_length))
    slicescores_oi = slicescores.loc[slicescores["slice"].isin(slices_length.index)]
    enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores_oi, slicecounts[motifclustermapping["motif"].unique()])
    enrichment = enrichment.loc[enrichment["contingency"].str[1].str[1] != 0]
    enrichment["log_odds"] = np.log(enrichment["odds"])

    motif_scores.append({
        "log_odds":enrichment.reindex(pd.MultiIndex.from_frame(motifclustermapping))["log_odds"].mean(),
        "length_bin":length_bin,
    })
    motif_scores_all.append(enrichment.assign(length_bin = length_bin))
motif_scores = pd.DataFrame(motif_scores)

# %%
motif_scores_all = pd.concat(motif_scores_all)

# %%
plotdata = motif_scores_all.loc[pd.MultiIndex.from_frame(motifclustermapping[["cluster", "motif"]])].reset_index().set_index(["cluster", "motif", "length_bin"])["log_odds"].unstack()

# %%
# cluster plotdata and extract the leaf ordering
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

clusterdata = plotdata.copy()
schlinkage = sch.linkage(1-np.corrcoef(clusterdata), method="ward")
dendrogram = sch.dendrogram(schlinkage, no_plot=True)

# %%
plotdata = plotdata / plotdata.values.max(1, keepdims=True)
plotdata = plotdata.iloc[np.argsort(plotdata.idxmax(1))]

cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=-1, vmax=1.)

motif_resolution = 0.1
fig, ax = plt.subplots(figsize = (2, len(plotdata) * motif_resolution))
ax.matshow(plotdata, cmap = cmap, norm = norm, aspect = "auto")
# ax.set_yticks()
# ax.set_yticklabels(plotdata.index)

ax.set_xticks(length_bins["ix"])
ax.set_xticklabels(length_bins["label"], rotation = 90, fontsize = 10, ha = "center", va = "bottom")

# %%
plotdata = motif_scores_all.loc[pd.MultiIndex.from_frame(motifclustermapping[["cluster", "motif"]])].reset_index().set_index(["cluster", "motif", "length_bin"])["log_odds"].unstack()

cmap = mpl.cm.PiYG
odds_max = 4.
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max))

sns.heatmap(plotdata.iloc[np.argsort(plotdata.idxmax(1))], cmap = cmap, norm = norm)

# %% [markdown]
# ### Motifs within sizes

# %%
slicescores["length_bin"] = np.searchsorted(length_bins["cut"], slicescores["length"])

# %%
enrichments = []
for cluster_oi in tqdm.tqdm(clustering.var.index):
    slicescores_oi = slicescores.query("cluster == @cluster_oi").copy()
    slicescores_oi["cluster"] = pd.Categorical(slicescores_oi["length_bin"])
    motifclustermapping_oi = motifclustermapping.query("cluster == @cluster_oi")
    enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores_oi, slicecounts[motifclustermapping_oi["motif"].unique()])
    enrichment["cluster_oi"] = cluster_oi
    enrichments.append(enrichment)
enrichments = pd.concat(enrichments)

# %%
plotdata = np.log(enrichments.reset_index().set_index(["cluster_oi", "motif", "cluster"])["odds"].unstack())
plotdata = plotdata.iloc[np.argsort(plotdata.idxmax(1))]

# %%
cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=-1, vmax=1.)

motif_resolution = 0.1
fig, ax = plt.subplots(figsize = (2, len(plotdata) * motif_resolution))
ax.matshow(plotdata, cmap = cmap, norm = norm, aspect = "auto")
# ax.set_yticks()
# ax.set_yticklabels(plotdata.index)

ax.set_xticks(length_bins["ix"])
ax.set_xticklabels(length_bins["label"], rotation = 90, fontsize = 10, ha = "center", va = "bottom")
""

# %%
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores_oi, slicecounts[motifclustermapping["motif"].unique()])

# %% [markdown]
# ### Motifs general for plot

# %%
motif_scores = []
motif_scores_all = []
for length_bin, slices_length in slices.groupby("length_bin"):
    print(len(slices_length))
    slicescores_oi = slicescores.loc[slicescores["slice"].isin(slices_length.index)]
    enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores_oi, slicecounts[motifclustermapping["motif"].unique()])
    enrichment = enrichment.loc[enrichment["contingency"].str[1].str[1] != 0]
    enrichment["log_odds"] = np.log(enrichment["odds"])

    motif_scores.append({
        "log_odds":enrichment.reindex(pd.MultiIndex.from_frame(motifclustermapping))["log_odds"].mean(),
        "length_bin":length_bin,
    })
    motif_scores_all.append(enrichment.assign(length_bin = length_bin))
motif_scores = pd.DataFrame(motif_scores)

# %% [markdown]
# ## Plot

# %%
import chromatinhd_manuscript as chdm

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height = 0.1, padding_width = 1.5))

panel_width = 1
panel_height = 0.5

## # positions
panel, ax = fig.main[0, 0] = fig.main.add_under(polyptich.grid.Panel((panel_width, panel_height)))
plotdata = length_scores
plotdata["ix"] = length_bins.loc[plotdata.index, "ix"]
ax.bar(
    plotdata["ix"],
    plotdata["n_positions"],
    color = "black",
    width = 1,
    lw = 0.5,
    edgecolor = "#FFFFFF99",
)
ax.set_ylabel("# positions", rotation = 0, ha = "right", va = "center")
ax.yaxis.set_major_formatter(chd.plot.tickers.distance_ticker)

## # slices
panel, ax = fig.main[1, 0] = polyptich.grid.Panel((panel_width, panel_height))
plotdata = length_scores
plotdata["ix"] = length_bins.loc[plotdata.index, "ix"]
ax.bar(
    plotdata["ix"],
    plotdata["n_slices"],
    color = "black",
    width = 1,
    lw = 0.5,
    edgecolor = "#FFFFFF99",
)
ax.set_ylabel("# regions", rotation = 0, ha = "right", va = "center")


## CRISPR
panel, ax = fig.main[2, 0] = (polyptich.grid.Panel((panel_width, panel_height)))
plotdata = crispr_scores
plotdata["ix"] = length_bins.loc[plotdata.index, "ix"]
ax.bar(
    plotdata["ix"],
    plotdata["score"],
    width = 1,
    color = "black",
    lw = 0.5,
    edgecolor = "#FFFFFF99",
    # marker = "o"
)
ax.set_ylim(0, np.log(0.75))
ax.set_yticks([0, np.log(0.8)])
ax.set_yticklabels(["1", "0.8"])
ax.axhline(data["HS_LS_logratio"].mean(), color = "tomato", dashes = (2, 2), lw = 1.5)
for length_bin, plotdata_length in plotdata.loc[pd.isnull(plotdata["score"])].iterrows():
    ax.axvspan(
        plotdata_length["ix"]-0.5,
        plotdata_length["ix"]+0.5,
        color = "#33333322",
        zorder = -1,
        lw = 0.5,
        edgecolor = "#FFFFFF99",
    )
ax.set_ylabel("\n\nCRISPRi low vs high fold", rotation = 0, ha = "right", va = "center")

text = ax.annotate(
    "random",
    xy = (1, data["HS_LS_logratio"].mean() - 0.02),
    xycoords = mpl.transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    ),
    ha = "right",
    color = "tomato",
    fontsize = 8,
    fontweight = "bold",
)
# add white border
text.set_path_effects(
    [
        mpl.patheffects.Stroke(linewidth = 1., foreground = "black"),
        mpl.patheffects.Normal(),
    ]
)

## Odds
max_odds = 8.

for ix, (motifscan_name, plotdata) in enumerate(scores.groupby("motifscan")):
    panel, ax = fig.main[ix, 1] = polyptich.grid.Panel((panel_width, panel_height))

    plotdata["length"] = length_bins["label"].iloc[plotdata["length_bin"]].values
    ax.bar(
        np.arange(len(plotdata)),
        plotdata["odds"],
        color = "black",
        bottom = 1,
        width = 1,
        lw = 0.5,
        edgecolor = "#FFFFFF99",
    )
    ax.set_yscale("log")
    ax.set_ylim(1, max_odds)
    # label = chdm.qtl_motifscans.motifscans.loc[motifscan_name, "label"]
    label = "GWAS odds" if motifscan_name == "gwas_immune_main" else "eQTL odds"
    ax.set_ylabel(f"\n\n{label}", rotation = 0, ha = "right", va = "center")

    plotdata["ix"] = np.arange(len(plotdata))
    for length_bin, plotdata_length in plotdata.query("odds > @max_odds").iterrows():
        ax.text(
            plotdata_length["ix"]+0.1,
            0.98,
            f"{plotdata_length['odds']:0.1f}",
            transform = mpl.transforms.blended_transform_factory(
                ax.transData, ax.transAxes
            ),
            ha = "center",
            va = "top",
            color = "white",
            rotation = 90,
            fontsize = 8,
        )

    ax.set_yticklabels([], minor = True)
    ax.set_yticks([1, 4])
    ax.set_yticklabels(["1", "4"])

    ax.axhline(1.1, color = "tomato", dashes = (2, 2), lw = 1.5)

## Motif
panel, ax = fig.main[ix+1, 1] = polyptich.grid.Panel((panel_width, panel_height))
plotdata = motif_scores
plotdata["ix"] = length_bins.loc[plotdata.index, "ix"]
ax.bar(
    plotdata["ix"],
    plotdata["log_odds"],
    width = 1,
    color = "black",
    lw = 0.5,
    edgecolor = "#FFFFFF99",
    # marker = "o"
)
ax.set_ylabel("\n\nTFBS odds", rotation = 0, ha = "right", va = "center")
ax.axhline(0.01, color = "tomato", dashes = (2, 2), lw = 1.5)
ax.set_yticks([np.log(1), np.log(1.5)])
ax.set_yticklabels(["1", "1.5"])

# set common ticks
for panel, ax in fig.main:
    ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlim(-0.5, len(length_bins) - 0.5)
# panel, ax = fig.main.get_bottom_left_corner()

for panel, ax in [fig.main[2, 0], fig.main[2, 1]]:
    ax.set_xticks(length_bins["ix"])
    ax.set_xticklabels(length_bins["label"], rotation = 90, fontsize = 8, ha = "center", va = "top")
    ax.tick_params(axis = "x", which = "major", length = 0, pad = 2)
    ax.set_xlabel("Region length")

fig.plot()

manuscript.save_figure(fig, "4", "size_enrichments")

# %%
import scipy.stats
resolution = 0.005

# %%
window = [-15, 15]
plotdata = pd.DataFrame({
    "position":np.arange(*window),
    "prob_mean":np.log(scipy.stats.norm.pdf(np.arange(*window), 0, 5) * 100),
    "prob_diff":scipy.stats.norm.pdf(np.arange(*window), 0, 10) * 40
})
plotdata_mean = plotdata[["prob_mean"]].rename(columns = {"prob_mean":"prob"})
plotdata_cluster = plotdata.copy()
plotdata_cluster["prob"] = plotdata_cluster["prob_mean"] + plotdata_cluster["prob_diff"]

cmap_atac_diff, norm_atac_diff = chd.models.diff.plot.differential.get_cmap_atac_diff(), chd.models.diff.plot.differential.get_norm_atac_diff()

fig, ax = plt.subplots(figsize = ((window[1] - window[0])*resolution, 0.4))
chd.models.diff.plot.differential._draw_differential(ax, plotdata_cluster, plotdata_mean, cmap_atac_diff, norm_atac_diff)
ax.axis("off")

# %%

window = [-50, 50]
plotdata = pd.DataFrame({
    "position":np.arange(*window),
    "prob_mean":np.log(scipy.stats.norm.pdf(np.arange(*window), 0, 20) * 10),
    "prob_diff":scipy.stats.norm.pdf(np.arange(*window), 0, 20) * 75
})
plotdata_mean = plotdata[["prob_mean"]].rename(columns = {"prob_mean":"prob"})
plotdata_cluster = plotdata.copy()
plotdata_cluster["prob"] = plotdata_cluster["prob_mean"] + plotdata_cluster["prob_diff"]

cmap_atac_diff, norm_atac_diff = chd.models.diff.plot.differential.get_cmap_atac_diff(), chd.models.diff.plot.differential.get_norm_atac_diff()

fig, ax = plt.subplots(figsize = ((window[1] - window[0])*resolution, 0.4))
chd.models.diff.plot.differential._draw_differential(ax, plotdata_cluster, plotdata_mean, cmap_atac_diff, norm_atac_diff)
ax.axis("off")

# %%
window = [-100, 100]
plotdata = pd.DataFrame({
    "position":np.arange(*window),
    "prob_mean":np.log(scipy.stats.norm.pdf(np.arange(*window), 0, 100) * 5),
    "prob_diff":scipy.stats.norm.pdf(np.arange(*window), 0, 50) * 150
})
plotdata_mean = plotdata[["prob_mean"]].rename(columns = {"prob_mean":"prob"})
plotdata_cluster = plotdata.copy()
plotdata_cluster["prob"] = plotdata_cluster["prob_mean"] + plotdata_cluster["prob_diff"]

cmap_atac_diff, norm_atac_diff = chd.models.diff.plot.differential.get_cmap_atac_diff(), chd.models.diff.plot.differential.get_norm_atac_diff()

fig, ax = plt.subplots(figsize = ((window[1] - window[0])*resolution, 0.4))
chd.models.diff.plot.differential._draw_differential(ax, plotdata_cluster, plotdata_mean, cmap_atac_diff, norm_atac_diff)
ax.axis("off")

# %%
