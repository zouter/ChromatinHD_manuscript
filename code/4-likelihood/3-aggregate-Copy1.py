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
import itertools

# %% [markdown]
# ### Method info

# %%
from designs import dataset_latent_peakcaller_combinations as design

design = chd.utils.crossing(
    design,
    pd.DataFrame({"method": ["v9_128-64-32"]}),
    pd.DataFrame({"diffexp": ["scanpy"]}),
    pd.DataFrame(
        {
            "motifscan": ["cutoff_0001"],
        }
    ),
    pd.DataFrame(
        {
            "enricher": [
                "cluster_vs_clusters",
                "cluster_vs_background",
            ],
        }
    ),
)
design = design.query("dataset != 'alzheimer'")#.query("peakcaller == 'cellranger'").query("enricher == 'cluster_vs_background'")

# %%
promoter_name = "10k10k"


# %%
def get_score_folder(x):
    return chd.get_output() / "prediction_likelihood" / x.dataset / promoter_name / x.latent / x.method / "scoring" / x.peakcaller / x.diffexp / x.motifscan / x.enricher
design["score_folder"] = design.apply(get_score_folder, axis = 1)

# %%
import scipy.stats


# %%
def calculate_motifscore_expression_correlations(motifscores):
    linreg_peak = scipy.stats.linregress(motifscores["expression_lfc"], motifscores["logodds_peak"])
    slope_peak = linreg_peak.slope
    r2_peak = linreg_peak.rvalue ** 2
    
    linreg_region = scipy.stats.linregress(motifscores["expression_lfc"], motifscores["logodds_region"])
    slope_region = linreg_region.slope
    r2_region = linreg_region.rvalue ** 2
    
    if (r2_peak > 0) and (r2_region > 0):
        r2_diff = (r2_region - r2_peak)
    elif (r2_region > 0):
        r2_diff = r2_region
    elif (r2_peak > 0):
        r2_diff = -r2_peak
    else:
        r2_diff = 0.
        
    cor_peak = np.corrcoef(motifscores["expression_lfc"], motifscores["logodds_peak"])[0, 1]
    cor_region = np.corrcoef(motifscores["expression_lfc"], motifscores["logodds_region"])[0, 1]
    cor_diff = cor_region - cor_peak
    
    contingency_peak = pd.crosstab(index=motifscores_oi["expression_lfc"] > 0, columns=motifscores_oi["logodds_peak"] > 0)
    contingency_region = pd.crosstab(index=motifscores_oi["expression_lfc"] > 0, columns=motifscores_oi["logodds_region"] > 0)
    
    odds_peak = scipy.stats.contingency.odds_ratio(contingency_peak).statistic
    odds_region = scipy.stats.contingency.odds_ratio(contingency_region).statistic
    
    return {
        "cor_peak":cor_peak,
        "cor_region":cor_region,
        "cor_diff":cor_diff,
        "r2_region":r2_region,
        "r2_diff":r2_diff,
        "slope_region":slope_region,
        "slope_peak":slope_peak,
        "slope_diff":slope_region - slope_peak,
        "logodds_peak":np.log(odds_peak),
        "logodds_region":np.log(odds_region),
        "logodds_difference":np.log(odds_region) - np.log(odds_peak)
    }


# %%
scores = []
for design_ix, design_row in tqdm.tqdm(design.iterrows(), total = len(design)):
    dataset_name = design_row["dataset"]
    latent_name = design_row["latent"]

    folder_data_preproc = chd.get_output() / "data" / dataset_name
    promoter_name = "10k10k"
    transcriptome = chd.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    latent_folder = folder_data_preproc / "latent"
    latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

    cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
    transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

    motifscan_name = design_row["motifscan"]
    motifscan_folder = (
        chd.get_output()
        / "motifscans"
        / dataset_name
        / promoter_name
        / motifscan_name
    )
    motifscan = chd.data.Motifscan(motifscan_folder)
    
    sc.tl.rank_genes_groups(transcriptome.adata, "cluster", method = "t-test")
    
    motifscores = None
    try:
        score_folder = design_row["score_folder"]
        scores_peaks = pd.read_pickle(
            score_folder / "scores_peaks.pkl"
        )
        scores_regions = pd.read_pickle(
            score_folder / "scores_regions.pkl"
        )

        # scores[ix] = scores_peaks
        motifscores = pd.merge(scores_peaks, scores_regions, on = ["cluster", "motif"], suffixes = ("_peak", "_region"), how = "outer")
    except BaseException as e:
        print(e)

    if motifscores is not None:
        for cluster_oi in cluster_info.index:
            score = {}
            diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, cluster_oi)
            diffexp = diffexp.set_index("names")

            motifs_oi = motifscan.motifs.loc[motifscan.motifs["gene"].isin(diffexp.index)]

            motifscores_oi = motifscores.loc[cluster_oi].loc[motifs_oi.index].sort_values("logodds_peak", ascending = False)
            motifscores_oi["gene"] = motifs_oi.loc[motifscores_oi.index, "gene"]
            motifscores_oi["expression_lfc"] = np.clip(diffexp.loc[motifscores_oi["gene"]]["logfoldchanges"].tolist(), -np.log(4), np.log(4))

            motifscores_significant = motifscores_oi.query("(qval_peak < 0.05) | (qval_region < 0.05)")
            if len(motifscores_significant) == 0:
                motifscores_significant = motifscores_oi.iloc[[0]]

            score.update({k + "_all":v for k, v in calculate_motifscore_expression_correlations(motifscores_oi).items()})
            score.update({k + "_significant":v for k, v in calculate_motifscore_expression_correlations(motifscores_significant).items()})

            # get logodds slope of all
            linreg_peakslope = scipy.stats.linregress(motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"])
            slope_logodds_diffexp = np.clip(1/linreg_peakslope.slope, 0, 3)

            motifscores_all = motifscores.loc[cluster_oi]
            linreg_peakslope = scipy.stats.linregress(motifscores_all["logodds_region"], motifscores_all["logodds_peak"])
            slope_logodds_all = np.clip(1/linreg_peakslope.slope, 0, 3)

            score.update({"slope_logodds_diffexp":slope_logodds_diffexp, "slope_logodds_all":slope_logodds_all})

            score["cluster"] = cluster_oi

            score["n_cells"] = (transcriptome.obs["cluster"] == cluster_oi).sum()
            
            score["design_ix"] = design_ix
            score["cluster"] = cluster_oi

            scores.append(score)
scores = pd.DataFrame(scores)

# %%
scores_joined = scores.set_index("design_ix").join(design)

# %%
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 500").query("peakcaller == 'macs2_improved'")["cor_diff_all"].mean()

# %%
np.exp(scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 500").groupby(["dataset", "peakcaller"])["logodds_difference_significant"].mean()).unstack().plot()

# %%
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 500").groupby(["dataset", "peakcaller"])["slope_diff_all"].mean().unstack().plot()

# %%
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 500").groupby(["dataset", "peakcaller"])["cor_diff_significant"].mean().unstack().plot()

# %%
fig, ax = plt.subplots()
scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > 100").query("slope_logodds_diffexp > 0").groupby(["dataset", "peakcaller"])["slope_logodds_all"].median().unstack().plot(ax = ax)
ax.axhline(1)

# %%
scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > 100").groupby("peakcaller")["slope_logodds_diffexp"].median()

# %%
scores1 = scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 500")
meanscores1 = scores1.groupby("peakcaller").mean()
scores2 = scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > 500")
meanscores2 = scores2.groupby("peakcaller").mean()

# %%
fig, ax = plt.subplots()
np.exp(meanscores2["slope_logodds_diffexp"]).plot(kind = "bar")

# %%
meanscores1["logodds_difference_significant"].plot(kind = "bar")

# %%
meanscores1["cor_diff_all"].plot(kind = "bar")

# %%
scores.sort_values("n_cells").query("n_cells > 100").style.bar()

# %%
scores_.sort_values("n_cells").query("n_cells > 100")[["cor_diff_all", "slope_diff_all","r2_diff_all", "cor_diff_significant", "slope_diff_significant", "r2_diff_significant", "slope_logodds_diffexp", "slope_logodds_all"]].mean()

# %%
scores_.sort_values("n_cells").query("n_cells > 100")[["cor_diff_all", "slope_diff_all","r2_diff_all", "cor_diff_significant", "slope_diff_significant", "r2_diff_significant", "slope_logodds_diffexp", "slope_logodds_all"]].mean()

# %%
# fig, ax = plt.subplots()
# (scores_["slope_logodds_all"]).plot()
# ax.set_yscale("log")

# %%
# cluster_oi = "B"
cluster_oi = "T"
# cluster_oi = "Monocytes"
# cluster_oi = "NK"
# cluster_oi = "Lymphoma"
# cluster_oi = "leiden_0"

# %%
diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, cluster_oi)
diffexp = diffexp.set_index("names")

motifs_oi = motifscan.motifs.loc[motifscan.motifs["gene"].isin(diffexp.index)]

motifscores_oi = scores.loc[cluster_oi].loc[motifs_oi.index].sort_values("logodds_region", ascending = False)
motifscores_oi["gene"] = motifs_oi.loc[motifscores_oi.index, "gene"]
motifscores_oi["expression_lfc"] = np.clip(diffexp.loc[motifscores_oi["gene"]]["logfoldchanges"].tolist(), -np.log(4), np.log(4))

# %%
motifscores_oi["logodds_diff"] = (motifscores_oi["logodds_peak"] - motifscores_oi["logodds_region"])

# %%
motifscores_oi.query("qval_region < 0.05").sort_values("logodds_diff", ascending = True).head(20)[["odds_peak", "odds_region", "expression_lfc"]]

# %%
motifscores_oi

# %%
sc.pl.umap(transcriptome.adata, color = ["cluster"])
sc.pl.umap(transcriptome.adata, color = motifs_oi.loc[motif_ids_oi]["gene"], title = motifs_oi.loc[motif_ids_oi]["gene_label"])

# %%
motif_ids_oi = [
    # motifs_oi.index[motifs_oi.index.str.startswith("SPI1")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("CEBPB")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("RUNX2")][0]
    
    # B
    # motifs_oi.index[motifs_oi.index.str.startswith("PO2F2")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("CEBPB")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("NFKB2")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("FOS")][0]
    
    # leiden 0
    # motifs_oi.index[motifs_oi.index.str.startswith("NDF1")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("CUX2")][0],
    # motifs_oi.index[motifs_oi.index.str.startswith("COT2")][0]
    
    # T lymphoma
    motifs_oi.index[motifs_oi.index.str.startswith("RUNX3")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("JUNB")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("TCF7")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("MEF2C")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("PAX5")][0],
    motifs_oi.index[motifs_oi.index.str.startswith("KLF12")][0],
]

# %%
plotdata = motifscores_oi.query("(qval_peak < 0.05) | (qval_region < 0.05)")
plotdata_scores = calculate_motifscore_expression_correlations(plotdata)
plotdata_oi = motifscores_oi.loc[motif_ids_oi]

# %%
import adjustText

# %%
fig, axes = plt.subplots(1, 2, figsize = (4, 2), sharex = True, sharey = True)

for ax, suffix in zip(axes, ["_peak", "_region"]):
    ax.scatter(plotdata["expression_lfc"], plotdata["logodds" + suffix], s = 1)
    ax.scatter(plotdata_oi["expression_lfc"], plotdata_oi["logodds" + suffix])
    ax.axvline(0, color = "#333", dashes = (2, 2))
    ax.axhline(0, color = "#333", dashes = (2, 2))
    cor = plotdata_scores["cor" + suffix]
    odds = np.exp(plotdata_scores["logodds" + suffix])
    ax.text(0.05, 0.95, f"r = {cor:.2f}\nodds = {odds:.1f}", transform = ax.transAxes, va = "top", fontsize = 9)
    
    texts = []
    for _, row in plotdata_oi.iterrows():
        label = motifs_oi.loc[row.name]["gene_label"]
        text = ax.text(row["expression_lfc"], row["logodds" + suffix], label, fontsize = 8, ha = "center")
        texts.append(text)
        
    adjustText.adjust_text(texts, ax = ax)

# ax = axes[1]
# ax.scatter(plotdata["expression_lfc"], plotdata["logodds_region"])
# ax.scatter(plotdata_oi["expression_lfc"], plotdata_oi["logodds_region"])
# ax.axvline(0, color = "#333")
# ax.axhline(0, color = "#333")
# cor = np.corrcoef(plotdata["expression_lfc"], plotdata["logodds_region"])[0, 1]
# ax.text(0.05, 0.95, f"r = {cor:.2f}", transform = ax.transAxes, va = "top")

# %%
# plotdata.sort_values("logodds_region")

# %%
np.corrcoef(motifscores_oi["expression_lfc"], motifscores_oi["logodds_region"])

# %%
np.corrcoef(motifscores_oi["expression_lfc"], motifscores_oi["logodds_peak"])

# %%
# motifscores_oi.sort_values("logodds_peak", ascending = False)[["logodds_peak", "logodds_region"]].style.bar()

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
ax.scatter(
    np.exp(motifscores_oi["logodds_peak"]),
    np.exp(motifscores_oi["logodds_region"]),
    s = 1
)

ax.set_ylim(1/4, 4)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(1/4, 4)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["¼", "½", "1", "2", "4"])

for i, label in zip([1/2, 1/np.sqrt(2), np.sqrt(2), 2], ["½", r"$\frac{1}{\sqrt{2}}$", "$\sqrt{2}$", "2"]):
    intercept = 1
    slope = i
    ax.axline((1, slope * 1), (intercept*2, slope * 2), color = "grey", dashes = (1, 1))
    
    if i > 1:
        x = 4
        y = intercept + slope * i
        ax.text(x, y, label, fontsize = 8)
    # ax.text(np.sqrt(1/i), np.sqrt(i), label, fontsize = 8)
ax.axvline(1, color = "grey")
ax.axhline(1, color = "grey")
ax.set_xlabel("Odds-ratio differential peaks")
ax.set_ylabel("Odds-ratio\ndifferential\nChromatinHD\nregions", rotation = 0, va = "center", ha = "right")

linreg = scipy.stats.linregress(motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"])
slope = linreg.slope
intercept = linreg.intercept
print(1/slope)

ax.axline((np.exp(0), np.exp(intercept)), (np.exp(1), np.exp(1/slope)), color = "orange")

# %%
scores = {}
for ix, score_folder in design["score_folder"].items():
    try:
        scores_peaks = pd.read_pickle(
            score_folder / "scores_peaks.pkl"
        )
        scores_regions = pd.read_pickle(
            score_folder / "scores_regions.pkl"
        )

        # scores[ix] = scores_peaks
        scores = pd.merge(scores_peaks, scores_regions, on = ["cluster", "motif"], suffixes = ("_peak", "_region"))
        print(scores.groupby("cluster").apply(calculate_overenrichment))
    except BaseException as e:
        print(e)
# scores = pd.concat(scores, names=["dataset", "latent", "method", "predictor", *scores_.index.names])
