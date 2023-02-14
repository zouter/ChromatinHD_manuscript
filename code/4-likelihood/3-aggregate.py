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
                "cluster_vs_background_gc",
            ],
        }
    ),
)

# %%
promoter_name = "10k10k"


# %%
def get_score_folder(x):
    return chd.get_output() / "prediction_likelihood" / x.dataset / promoter_name / x.latent / x.method / "scoring" / x.peakcaller / x.diffexp / x.motifscan / x.enricher
design["score_folder"] = design.apply(get_score_folder, axis = 1)

# %%
import scipy.stats

# %%
design_row = (
    design
    .query("dataset == 'pbmc10k'")
    # .query("dataset == 'lymphoma'")
    # .query("dataset == 'e18brain'")
    # .query("dataset == 'alzheimer'")
    # .query("dataset == 'brain'")
    # .query("dataset == 'alzheimer'")
    
    .query("peakcaller == 'macs2_improved'")
    # .query("enricher == 'cluster_vs_background_gc'")
    # .query("enricher == 'cluster_vs_background'")
    .query("enricher == 'cluster_vs_clusters'")
    .iloc[0]
)
score_folder = design_row["score_folder"]

# %%
print(score_folder)
scores_peaks = pd.read_pickle(
    score_folder / "scores_peaks.pkl"
)
scores_regions = pd.read_pickle(
    score_folder / "scores_regions.pkl"
)

# scores[ix] = scores_peaks
motifscores = pd.merge(scores_peaks, scores_regions, on = ["cluster", "motif"], suffixes = ("_peak", "_region"), how = "outer")

# %%
dataset_name = design_row["dataset"]

# %%
folder_data_preproc = chd.get_output() / "data" / dataset_name

# fragments
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "20kpromoter", np.array([-10000, 0])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

transcriptome = chd.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# %%
latent_name = design_row["latent"]

# %%
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %%
motifscan_name = design_row["motifscan"]
motifscan_folder = (
    chd.get_output()
    / "motifscans"
    / dataset_name
    / promoter_name
    / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)

# %%
sc.tl.rank_genes_groups(transcriptome.adata, "cluster")


# %%
# transcriptome.obs.groupby("cluster").size()

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
    slope_logodds_diffexp = 1/linreg_peakslope.slope
    
    motifscores_all = motifscores.loc[cluster_oi]
    linreg_peakslope = scipy.stats.linregress(motifscores_all["logodds_region"], motifscores_all["logodds_peak"])
    slope_logodds_all = 1/linreg_peakslope.slope
    
    # if cluster_oi == "Monocytes":
    #     fig, ax = plt.subplots()
    #     ax.scatter(motifscores_all["logodds_region"], motifscores_all["logodds_peak"])
    #     break
    
    score.update({"slope_logodds_diffexp":slope_logodds_diffexp, "slope_logodds_all":slope_logodds_all})
    
    score["cluster"] = cluster_oi
    score["n_cells"] = (transcriptome.adata.obs["cluster"] == cluster_oi).sum()
    
    scores.append(score)

# %%
scores = pd.DataFrame(scores)

# %%
pd.DataFrame(scores).query("n_cells > 100")["cor_diff_all"].mean()

# %%
scores.sort_values("n_cells").query("n_cells > 100").style.bar()

# %%
scores.sort_values("n_cells").query("n_cells > 100")[["cor_diff_all", "slope_diff_all","r2_diff_all", "cor_diff_significant", "slope_diff_significant", "r2_diff_significant", "slope_logodds_diffexp", "slope_logodds_all"]].mean()

# %%
# scores_.sort_values("n_cells").query("n_cells > 100")[["cor_diff_all", "slope_diff_all","r2_diff_all", "cor_diff_significant", "slope_diff_significant", "r2_diff_significant", "slope_logodds_diffexp", "slope_logodds_all"]].mean()

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
