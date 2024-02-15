# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import tqdm.auto as tqdm
import itertools
import textwrap


# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")


# %%
from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations as design,
)

# %%
import scipy.stats

# %% [markdown]
# ## > Expression logodds correlation

# %%
# design_rows = design.query(
#     "(dataset == 'e18brain') and (promoter == '10k10k') and (latent == 'leiden_0.1') and (diffexp == 'scanpy') and (enricher == 'cluster_vs_clusters')"
# )
# cluster_oi = "leiden_0"

# design_rows = design.query(
#     "(dataset == 'lymphoma') and (promoter == '10k10k') and (latent == 'celltype') and (diffexp == 'scanpy') and (enricher == 'cluster_vs_clusters')"
# )
# cluster_oi = "T"

design_rows = design.query(
    "(dataset == 'pbmc10k') and (promoter == '10k10k') and (latent == 'leiden_0.1') and (diffexp == 'scanpy') and (enricher == 'cluster_vs_clusters')"
)
cluster_oi = "cDCs"


# %%
# motifscores_oi.query("design_ix == 0")[["logodds_peak", "logodds_region", "logodds_diff", "expression_lfc"]].sort_values("logodds_diff").style.bar()

# %%
from example import Example


# %%
example = Example(design_rows, cluster_oi)
example.fig.plot()
example.fig.show()
manuscript.save_figure(
    example.fig, "2", "likelihood_expression_correlation_" + cluster_oi
)

# %%
np.corrcoef(motifscores_oi["expression_lfc"], motifscores_oi["logodds_region"])

# %%
np.corrcoef(motifscores_oi["expression_lfc"], motifscores_oi["logodds_peak"])

# %%
# motifscores_oi.sort_values("logodds_peak", ascending = False)[["logodds_peak", "logodds_region"]].style.bar()

# %% [markdown]
# ## > Focus on a single motif

# %%
# focus only on clusters with enough cells, e.g. 50
# all the rest will be noise anyway
clusters_oi = transcriptome.adata.obs.groupby("cluster").size() > 50
clusters_oi = clusters_oi.index[clusters_oi]

# %%
# scores2 = []
# for motif_oi in motifs_oi.index:
#     gene_oi = motifs_oi.loc[motif_oi]["gene"]
#     plotdata_expression = (
#         transcriptome.adata.obs.assign(
#             expression=sc.get.obs_df(transcriptome.adata, keys=gene_oi)
#         )
#         .groupby("cluster")["expression"]
#         .mean()
#         .loc[clusters_oi]
#     )

#     # design_ixs_oi = design_rows.query("peakcaller in 'macs2_leiden_0.1_merged'").index[0]
#     design_ixs_oi = design_rows.index
#     plotdata_logodds_peaks = (
#         (
#             motifscores.query("(motif == @motif_oi)")
#             .query("cluster in @clusters_oi")
#             .reset_index()
#             .set_index(["cluster", "design_ix"])["logodds_peak"]
#         )
#         .unstack()
#         .loc[clusters_oi]
#     )
#     plotdata_logodds_region = (
#         (
#             motifscores.query("(motif == @motif_oi)")
#             .reset_index()
#             .set_index(["cluster", "design_ix"])["logodds_region"]
#         )
#         .unstack()
#         .loc[clusters_oi]
#     )

#     scores2.append(
#         pd.DataFrame(
#             {
#                 "motif": motif_oi,
#                 "gene": gene_oi,
#                 "cor_peak": np.corrcoef(plotdata_expression, plotdata_logodds_peaks.T)[
#                     0
#                 ],
#                 "cor_region": np.corrcoef(
#                     plotdata_expression, plotdata_logodds_region.T
#                 )[0],
#                 "design_ix": design_ixs_oi,
#             }
#         )
#     )


# scores2 = pd.concat(scores2)
# scores2["cor_diff"] = scores2["cor_peak"] - scores2["cor_region"]

# %%

# motif_oi = motifs_oi.index[motifs_oi.index.str.startswith("TCF7")][0]
# motif_oi = motifs_oi.index[motifs_oi.index.str.startswith("CEBPD")][0]
# motif_oi = motifs_oi.index[motifs_oi.index.str.startswith("COE1")][0]
# motif_oi = motifs_oi.index[motifs_oi.index.str.startswith("RUNX")][0]
motif_oi = motifs_oi.index[motifs_oi.index.str.startswith("IRF1")][0]
# motif_oi = motifs_oi.index[motifs_oi.index.str.startswith("STAT2")][0]

gene_oi = motifs_oi.loc[motif_oi]["gene"]

# %%

plotdata_expression = (
    transcriptome.adata.obs.assign(
        expression=sc.get.obs_df(transcriptome.adata, keys=gene_oi)
    )
    .groupby("cluster")["expression"]
    .mean()
    .loc[clusters_oi]
)

# design_ixs_oi = design_rows.query("peakcaller in 'macs2_leiden_0.1_merged'").index[0]
design_ixs_oi = design_rows.index
plotdata_logodds_peaks = (
    motifscores.query("(motif == @motif_oi)")
    .query("cluster in @clusters_oi")
    .reset_index()
    .set_index(["cluster", "design_ix"])["logodds_peak"]
).unstack()
ref_design_ix = design_rows.query("peakcaller == 'cellranger'").index[0]
plotdata_logodds_region = (
    motifscores.query("(motif == @motif_oi) & (design_ix == @ref_design_ix)")
    .query("cluster in @clusters_oi")
    .reset_index()
    .set_index(["cluster", "design_ix"])["logodds_region"]
).droplevel("design_ix")

logodds_norm = mpl.colors.Normalize(vmin=-np.log(2), vmax=np.log(2))
logodds_cmap = mpl.cm.RdBu_r

expression_cmap = mpl.cm.magma
expression_norm = mpl.colors.Normalize(vmin=0)

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.1))
panel = fig.main[0, 0] = chd.grid.Panel((2, len(plotdata_logodds_peaks.columns) * 0.15))
panel.ax.imshow(
    plotdata_logodds_peaks.T, cmap=logodds_cmap, norm=logodds_norm, aspect="auto"
)
panel.ax.set_yticks(np.arange(len(plotdata_logodds_peaks.columns)))
panel.ax.set_yticklabels(
    chdm.peakcallers.reindex(
        design_rows.loc[plotdata_logodds_peaks.columns, "peakcaller"]
    )["label"],
    fontsize=10,
)

panel = fig.main[1, 0] = chd.grid.Panel((2, 1 * 0.15))
panel.ax.imshow(
    plotdata_logodds_region.values[None, :],
    cmap=logodds_cmap,
    norm=logodds_norm,
    aspect="auto",
)
panel.ax.set_yticks([0])
panel.ax.set_yticklabels(["ChromatinHD"], fontsize=10)

panel = fig.main[2, 0] = chd.grid.Panel((2, 1 * 0.15))
panel.ax.imshow(
    plotdata_expression.values[None, :],
    cmap=expression_cmap,
    norm=expression_norm,
    aspect="auto",
)
panel.ax.set_xticks(np.arange(len(plotdata_logodds_peaks.index)))
panel.ax.set_xticklabels(plotdata_logodds_peaks.index, rotation=90, fontsize=10)

panel.ax.set_yticks([])

fig.plot()

from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")
manuscript.save_figure(fig, "2", "cross_celltype_correlation" + "_" + motif_oi)

# sns.heatmap(plotdata)

# np.corrcoef(plotdata.T.values)

# %%
fig_colorbar = plt.figure(figsize=(1.5, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(norm=logodds_norm, cmap=logodds_cmap)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal")
colorbar.set_ticks(np.log([1 / 2, 1, 2]))
colorbar.set_ticklabels(["1/2", "1", "2"])
colorbar.set_label("Log odds")
manuscript.save_figure(fig_colorbar, "2", "colorbar_logodds")

# %%
fig_colorbar = plt.figure(figsize=(1.5, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(norm=expression_norm, cmap=expression_cmap)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal")
colorbar.set_ticks([0, expression_norm.vmax])
colorbar.set_ticklabels(["0", "max"])
colorbar.set_label("RNA expression\n(log1p)")
manuscript.save_figure(fig_colorbar, "2", "colorbar_expression")

# %%
scores = {}
for ix, score_folder in design["score_folder"].items():
    try:
        scores_peaks = pd.read_pickle(score_folder / "scores_peaks.pkl")
        scores_regions = pd.read_pickle(score_folder / "scores_regions.pkl")

        # scores[ix] = scores_peaks
        scores = pd.merge(
            scores_peaks,
            scores_regions,
            on=["cluster", "motif"],
            suffixes=("_peak", "_region"),
        )
        print(scores.groupby("cluster").apply(calculate_overenrichment))
    except BaseException as e:
        print(e)
# scores = pd.concat(scores, names=["dataset", "latent", "method", "predictor", *scores_.index.names])

# %% [markdown]
# -----------------------------------------------------------------------------------------
# ## > Slope logodds
# -----------------------------------------------------------------------------------------

# %%
design_rows = design.query(
    "(dataset == 'pbmc10k') and (promoter == '10k10k') and (latent == 'leiden_0.1') and (enricher == 'cluster_vs_clusters')"
)
design_rows = design.query(
    "(dataset == 'lymphoma') and (promoter == '10k10k') and (latent == 'celltype') and (enricher == 'cluster_vs_background')"
)
# design_rows = design.query("(dataset == 'lymphoma') and (promoter == '10k10k') and (latent == 'celltype') and (diffexp == 'scanpy') and (enricher == 'cluster_vs_background')")
# design_rows = design.query("(promoter == '10k10k') and (diffexp == 'scanpy') and (enricher == 'cluster_vs_background')")

assert len(design_rows) > 0

# cluster_oi = "Monocytes"
# cluster_oi = "CD4 T"
cluster_oi = "T"
# cluster_oi = "cDCs"
# cluster_oi = "NK"
# cluster_oi = "B"

motifscores = []
for design_ix, design_row in design_rows.iterrows():
    score_folder = design_row["score_folder"]
    try:
        scores_peaks = pd.read_pickle(score_folder / "scores_peaks.pkl")
        scores_regions = pd.read_pickle(score_folder / "scores_regions.pkl").assign(
            design_ix="region"
        )

        motifscores.append(
            pd.merge(
                scores_peaks,
                scores_regions,
                on=["cluster", "motif"],
                suffixes=("_peak", "_region"),
                how="outer",
            ).assign(design_ix=design_ix)
        )
    except FileNotFoundError:
        pass

motifscores = pd.concat(motifscores).reset_index().set_index(["cluster", "motif"])

# %%
motifscores["abslogodds_peak"] = np.abs(motifscores["logodds_peak"])
motifscores["abslogodds_region"] = np.abs(motifscores["logodds_region"])

motifscores["improved"] = motifscores["odds_peak"] > motifscores["odds_region"]
motifscores.query(
    "((qval_peak < 0.05) & (odds_peak > 1.5)) | ((qval_region < 0.05) & (odds_region > 1.5))"
).groupby(["design_ix", "cluster"])["improved"].mean().unstack().mean(axis=1).plot()

motifscores.query(
    "((qval_peak < 0.05) & (odds_peak > 1.5)) | ((qval_region < 0.05) & (odds_region > 1.5))"
).groupby(["design_ix", "cluster"])["improved"].mean().unstack().mean(axis=1).mean()

# %%
motifscores_oi = motifscores.loc[[cluster_oi]].sort_values(
    "logodds_region", ascending=False
)


# %%
def plot(ax, motifscores_oi, motif_ids_oi=None):
    ax.set_aspect(1)
    ax.axline([0, 0], slope=1, color="#333333", zorder=0)
    ax.scatter(
        motifscores_oi["logodds_region"],
        motifscores_oi["logodds_peak"],
        s=1,
    )

    lim = 4
    ax.set_yticks(np.log([0.25, 0.5, 1, 2, 4]))
    ax.set_xticks(np.log([0.25, 0.5, 1, 2, 4]))
    ax.set_yticklabels(["¼", "½", "1", "2", "4"])
    ax.set_xticklabels(["¼", "½", "1", "2", "4"])

    ax.set_ylim(np.log(1 / lim), np.log(lim))
    ax.set_xlim(np.log(1 / lim), np.log(lim))

    ax.axvline(0, color="grey")
    ax.axhline(0, color="grey")
    ax.set_xlabel("Odds-ratio differential ChromatinHD")
    ax.set_ylabel(
        "Odds-ratio\ndifferential\npeaks",
        rotation=0,
        va="center",
        ha="right",
    )

    linreg = scipy.stats.linregress(
        motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"]
    )
    slope = linreg.slope
    intercept = linreg.intercept

    ax.axline((0, intercept), (1, slope + intercept), color="orange")
    ax.axline((0, 0), (1, 1 / 2), dashes=(2, 2), color="#33333388")
    ax.axline((0, 0), (1, 1 / 1.5), dashes=(2, 2), color="#33333388")

    if motif_ids_oi is not None:
        texts = []
        for _, row in motifscores_oi.loc[motif_ids_oi].iterrows():
            label = motifs_oi.loc[row.name]["gene_label"]
            text = ax.text(
                row["logodds_region"],
                row["logodds_peak"],
                label,
                fontsize=8,
                ha="center",
            )
            ax.scatter(
                row["logodds_region"],
                row["logodds_peak"],
            )
            texts.append(text)

        # adjustText.adjust_text(texts, ax=ax)


# %%
motif_ids_oi = [
    motifs_oi.index[motifs_oi.index.str.startswith("TCF")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("MEF2C")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("PAX5")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("IRF1")][0],
]

# %%
motifscores_oi.loc[cluster_oi].groupby("motif")[
    "logodds_peak"
].median().sort_values().head(10)

# %%
motifscores_grouped = motifscores_oi.groupby("design_ix")

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for design_ix, motifscores_group in motifscores_grouped:
    panel = fig.main.add(chd.grid.Panel((1.5, 1.5)))
    ax = panel.ax
    motifscores_group = motifscores_group.reset_index().set_index("motif")
    plot(ax, motifscores_group, motif_ids_oi=motif_ids_oi)
    ax.set_title(
        design.loc[design_ix, "diffexp"] + " " + design.loc[design_ix, "peakcaller"]
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
fig.plot()

# %% [markdown]
# ---------------------------------------------------------------------
# ### Specific clusters
# ---------------------------------------------------------------------

# %%
clusters_oi = ["CD4 T", "Monocytes", "B"]

design_ix_oi = design_rows.query(
    "(peakcaller == 'macs2_improved') & (diffexp == 'scanpy')"
).index[0]
motifscores_grouped = (
    motifscores.loc[clusters_oi]
    # .query("(qval_peak < 0.05) | (qval_region < 0.05)")
    .query("design_ix == @design_ix_oi").groupby(["design_ix", "cluster"])
)

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for (design_ix, cluster), motifscores_group in motifscores_grouped:
    panel = fig.main.add(chd.grid.Panel((1.5, 1.5)))
    ax = panel.ax
    motifscores_group = motifscores_group.reset_index().set_index("motif")
    # plot(ax, motifscores_group, motif_ids_oi=motif_ids_oi)
    plot(ax, motifscores_group)
    ax.set_title(
        design.loc[design_ix, "diffexp"] + " " + design.loc[design_ix, "peakcaller"]
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
fig.plot()

# %%
motifs_oi.loc["EHF_HUMAN.H11MO.0.B"]

# %%
motifscores_oi.groupby("motif")["logodds_peak"].mean().sort_values(ascending=False)

# %%
motifscores_oi["abslogodds_peak"] = np.abs(motifscores_oi["logodds_peak"])
motifscores_oi["abslogodds_region"] = np.abs(motifscores_oi["logodds_region"])

# %%
motifscores_oi = motifscores_oi.query("((qval_peak < 0.05) | (qval_region < 0.05))")
# motifscores_oi = motifscores_oi.query("((qval_peak < 0.05) | (qval_region < 0.05)) & ((abslogodds_peak > 0.7) | (abslogodds_region > 0.7))")

# %%
plt.scatter(motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"])

# %%
motifscores_oi["abslogodds_diff"] = (
    motifscores_oi["abslogodds_region"] - motifscores_oi["abslogodds_peak"]
)

# %%
design.index.name = "design_ix"

# %%
(motifscores_oi.xs("SPI1_HUMAN.H11MO.0.A", level="motif")["abslogodds_diff"] > 0)

# %%
motifscores_oi.groupby(["cluster", "motif"])["logodds_peak"].mean().sort_values().loc[
    "CD4 T"
]

# %%
motifscores_oi.groupby(["design_ix", "cluster"])[
    "abslogodds_diff"
].mean().to_frame().join(design).groupby("peakcaller")["abslogodds_diff"].mean()

# %%
motifscores_oi.groupby(["design_ix", "cluster"])["abslogodds_diff"].mean().groupby(
    "design_ix"
).mean().to_frame().join(design)

# %%
motifscores_oi.groupby(["design_ix", "cluster"])["abslogodds_diff"].mean().groupby(
    "design_ix"
).mean().plot(kind="hist")

# %%
for design_ix, motifscores_row in motifscores_oi.groupby("design_ix"):
    for cluster, motifscores_row in motifscores_row.groupby("cluster"):
        # fig, ax = plt.subplots()
        # ax.set_title(design.loc[design_ix, "peakcaller"])
        # sns.ecdfplot(motifscores_row["logodds_peak"], ax = ax)
        # sns.ecdfplot(motifscores_row["logodds_region"], ax = ax, lw = 5)
        print((motifscores_row["abslogodds_diff"]).mean())

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope=1, color="#333333", zorder=0)
ax.scatter(
    np.exp(motifscores_oi["logodds_peak"]),
    np.exp(motifscores_oi["logodds_region"]),
    s=1,
)

ax.set_ylim(1 / 4, 4)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(1 / 4, 4)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["¼", "½", "1", "2", "4"])

for i, label in zip(
    [1 / 2, 1 / np.sqrt(2), np.sqrt(2), 2],
    ["½", r"$\frac{1}{\sqrt{2}}$", "$\sqrt{2}$", "2"],
):
    intercept = 1
    slope = i
    ax.axline((1, slope * 1), (intercept * 2, slope * 2), color="grey", dashes=(1, 1))

    if i > 1:
        x = 4
        y = intercept + slope * i
        ax.text(x, y, label, fontsize=8)
    # ax.text(np.sqrt(1/i), np.sqrt(i), label, fontsize = 8)
ax.axvline(1, color="grey")
ax.axhline(1, color="grey")
ax.set_xlabel("Odds-ratio differential peaks")
ax.set_ylabel(
    "Odds-ratio\ndifferential\nChromatinHD\nregions",
    rotation=0,
    va="center",
    ha="right",
)

linreg = scipy.stats.linregress(
    motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"]
)
slope = linreg.slope
intercept = linreg.intercept
print(1 / slope)

ax.axline(
    (np.exp(0), np.exp(intercept)), (np.exp(1), np.exp(1 / slope)), color="orange"
)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope=1, color="#333333", zorder=0)
ax.scatter(
    motifscores_oi["logodds_peak"],
    motifscores_oi["logodds_region"],
    s=1,
)

ax.set_ylim(np.log(1 / 4), np.log(4))
ax.set_yticks(np.log([0.25, 0.5, 1, 2, 4]))
ax.set_yticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(np.log(1 / 4), np.log(4))
ax.set_xticks(np.log([0.25, 0.5, 1, 2, 4]))
ax.set_xticklabels(["¼", "½", "1", "2", "4"])

# for i, label in zip(
#     [1 / 2, 1 / np.sqrt(2), np.sqrt(2), 2],
#     ["½", r"$\frac{1}{\sqrt{2}}$", "$\sqrt{2}$", "2"],
# ):
#     intercept = 1
#     slope = i
#     ax.axline((1, slope * 1), (intercept * 2, slope * 2), color="grey", dashes=(1, 1))

#     if i > 1:
#         x = 4
#         y = intercept + slope * i
#         ax.text(x, y, label, fontsize=8)
# ax.text(np.sqrt(1/i), np.sqrt(i), label, fontsize = 8)
ax.axvline(0, color="grey")
ax.axhline(0, color="grey")
ax.set_xlabel("Odds-ratio differential peaks")
ax.set_ylabel(
    "Odds-ratio\ndifferential\nChromatinHD\nregions",
    rotation=0,
    va="center",
    ha="right",
)

linreg = scipy.stats.linregress(
    motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"]
)
slope = linreg.slope
intercept = linreg.intercept
print(slope)

ax.axline((0, intercept), (1, 1 / slope - intercept), color="orange")

# %%
motifscores_oi["logodds_diff"] = np.abs(motifscores_oi["logodds_region"]) - np.abs(
    motifscores_oi["logodds_peak"]
)
motifscores_oi["logodds_mean"] = (
    np.abs(motifscores_oi["logodds_region"]) + np.abs(motifscores_oi["logodds_peak"])
) / 2

# %%
motifscores_oi.sort_values("logodds_mean")["logodds_diff"].plot()

# %%
sns.ecdfplot(
    np.abs(motifscores_oi["logodds_region"]) - np.abs(motifscores_oi["logodds_peak"])
)

# %%

# %%
