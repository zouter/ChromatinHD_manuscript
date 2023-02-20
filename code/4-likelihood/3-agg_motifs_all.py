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

# %%
from chromatinhd_manuscript.designs import dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations as design

# %%
promoter_name = "10k10k"


# %%
def get_score_folder(x):
    return chd.get_output() / "prediction_likelihood" / x.dataset / promoter_name / x.latent / x.method / "scoring" / x.peakcaller / x.diffexp / x.motifscan / x.enricher
design["score_folder"] = design.apply(get_score_folder, axis = 1)

# %%
design = design.query("dataset != 'alzheimer'")

# %% [markdown]
# ## Aggregate

# %%
import scipy.stats


# %%
def calculate_motifscore_expression_correlations(motifscores):
    if motifscores["expression_lfc"].std() == 0:
        slope_peak = 0
        r2_peak = 0
        slope_region = 0
        r2_region = 0
    else:
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
    
    contingency_peak = pd.crosstab(
        index=pd.Categorical(motifscores_oi["expression_lfc"] > 0, [False, True]),
        columns=pd.Categorical(motifscores_oi["logodds_peak"] > 0, [False, True]),
        dropna = False
    )
    contingency_region = pd.crosstab(
        index=pd.Categorical(motifscores_oi["expression_lfc"] > 0, [False, True]),
        columns=pd.Categorical(motifscores_oi["logodds_region"] > 0, [False, True]),
        dropna = False
    )
    
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
design["force"] = False
# design.loc[design["peakcaller"] == 'macs2_leiden_0.1', "force"] = True

# %%
for design_ix, subdesign in tqdm.tqdm(design.iterrows(), total = len(design)):    
    design_row = subdesign
    
    desired_outputs = [(subdesign["score_folder"] / "aggscores.pkl")]
    force = subdesign["force"]
    if not all([desired_output.exists() for desired_output in desired_outputs]):
        force = True

    if force:
        # load motifscores
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
            continue

        # load latent, data, transcriptome, etc
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

        scores = []
        print(design_row)
        for cluster_oi in cluster_info.index:
            score = {}
            diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, cluster_oi)
            diffexp = diffexp.set_index("names")

            motifs_oi = motifscan.motifs.loc[motifscan.motifs["gene"].isin(diffexp.index)]
            
            if cluster_oi not in motifscores.index.get_level_values(0):
                continue

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

            score.update({"logodds_ratio_diffexp":np.exp(motifscores_significant["logodds_region"].abs().mean() - motifscores_significant["logodds_peak"].abs().mean())})
            score.update({"logodds_ratio_all":np.exp(motifscores_all["logodds_region"].abs().mean() - motifscores_all["logodds_peak"].abs().mean())})

            score["cluster"] = cluster_oi

            score["n_cells"] = (transcriptome.obs["cluster"] == cluster_oi).sum()

            # score["design_ix"] = design_ix
            score["cluster"] = cluster_oi

            scores.append(score)
        scores = pd.DataFrame(scores)
        pickle.dump(scores, (subdesign["score_folder"] / "aggscores.pkl").open("wb"))
# scores = pd.DataFrame(scores)

# %% [markdown]
# ## Summarize

# %%
scores = []
for design_ix, design_row in tqdm.tqdm(design.iterrows(), total = len(design)):    
    try:
        scores.append(pickle.load((design_row["score_folder"] / "aggscores.pkl").open("rb")).assign(design_ix = design_ix))
    except FileNotFoundError as e:
        pass
scores = pd.concat(scores)

# %%
scores_joined = scores.set_index("design_ix").join(design)

# %%
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 100").query("peakcaller == 'macs2_improved'")["cor_diff_all"].mean()

# %%
scores_joined["logslope_logodds_all"] = np.log(scores_joined["slope_logodds_all"])
scores_joined["logslope_logodds_diffexp"] = np.log(scores_joined["slope_logodds_diffexp"])

# %%
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 100").groupby(["dataset", "peakcaller"])["logodds_difference_significant"].mean().unstack().T.plot(kind = "bar", lw = 0)
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 100").groupby(["peakcaller"])["logodds_difference_significant"].mean().plot(kind = "bar", alpha = 1., zorder = 0, color = "black", lw = 0)

# %%
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 100").groupby(["dataset", "peakcaller"])["cor_diff_significant"].mean().unstack().T.plot(kind = "bar", lw = 0)
scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > 100").groupby(["peakcaller"])["cor_diff_significant"].mean().plot(kind = "bar", alpha = 1., zorder = 0, color = "black", lw = 0)

# %%
# np.log(scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > 100").query("slope_logodds_diffexp > 0").groupby(["dataset", "peakcaller"])["slope_logodds_all"].mean())

# %%
ncell_cutoff = 100

scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > @ncell_cutoff").query("slope_logodds_all > 0").groupby(["dataset", "peakcaller", "diffexp"])["logslope_logodds_all"].mean().unstack().unstack().T.plot(kind = "bar", lw = 0)
# np.log(scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > @ncell_cutoff").query("slope_logodds_all > 0").groupby(["peakcaller", "diffexp"])["slope_logodds_all"].mean()).plot(kind = "bar", alpha = 1., zorder = 0, color = "black", lw = 0)

# %% [markdown]
# ## Plot all combinations

# %%
methods_info = design.groupby(["peakcaller", "diffexp"]).first().index.to_frame(index = False)

methods_info["type"] = "predefined"
methods_info.loc[methods_info["peakcaller"].isin(["cellranger", "macs2_improved", "macs2_leiden_0.1_merged", "macs2_leiden_0.1", "genrich"]), "type"] = "peak"
methods_info.loc[methods_info["peakcaller"].str.startswith("rolling").fillna(False), "type"] = "rolling"

methods_info.loc[methods_info["type"] == "baseline", "color"] = "#888888"
methods_info.loc[methods_info["type"] == "ours", "color"] = "#0074D9"
methods_info.loc[methods_info["type"] == "rolling", "color"] = "#FF851B"
methods_info.loc[methods_info["type"] == "peak", "color"] = "#FF4136"
methods_info.loc[methods_info["type"] == "predefined", "color"] = "#2ECC40"
methods_info.loc[pd.isnull(methods_info["color"]), "color"] = "#DDDDDD"

methods_info["type"] = pd.Categorical(methods_info["type"], ["peak", "predefined", "rolling"])

methods_info = methods_info.set_index(["peakcaller", "diffexp"])
methods_info["label"] = methods_info.index.get_level_values("peakcaller")

methods_info = methods_info.sort_values(["diffexp", "type", "peakcaller"])
                                        
methods_info["ix"] = -np.arange(methods_info.shape[0])

# %%
datasets_info = pd.DataFrame(index = design.groupby(["dataset", "promoter", "latent"]).first().index)
datasets_info["label"] = datasets_info.index.get_level_values("dataset")
datasets_info["ix"] = np.arange(len(datasets_info))

# %%
group_ids = [*methods_info.index.names, *datasets_info.index.names]

# %%
meanscores = pd.concat([
    scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > @ncell_cutoff").groupby(group_ids)["logodds_difference_significant"].mean().to_frame(),
    scores_joined.query("enricher == 'cluster_vs_clusters'").query("n_cells > @ncell_cutoff").groupby(group_ids)["cor_diff_significant"].mean(),
    scores_joined.query("enricher == 'cluster_vs_background'").query("n_cells > @ncell_cutoff").query("slope_logodds_all > 0").groupby(group_ids)["logslope_logodds_all"].mean(),
], axis = 1)

# %%
import textwrap

# %%
metrics_info = pd.DataFrame([
    {"label":r"$\Delta$ log-odds", "metric":"logodds_difference_significant", "limits":(np.log(1/1.5), np.log(1.5)), "ticks":[-0.5, 0, 0.5], "ticklabels":["-0.5", "0", "+0.5"]},
    {"label":r"$\Delta$ cor", "metric":"cor_diff_significant", "limits":(-0.1, 0.1)},
    {"label":r"Slope logodds", "metric":"logslope_logodds_all", "limits":(np.log(1/2), np.log(2)), "ticks":[np.log(1/2), 0, np.log(2)], "ticklabels":["½", "1", "2"]}
]).set_index("metric")
metrics_info["ix"] = np.arange(len(metrics_info))

# %%
metrics_info["ticks"] = metrics_info["ticks"].fillna(metrics_info.apply(lambda metric_info: [metric_info["limits"][0], 0, metric_info["limits"][1]], axis = 1))
metrics_info["ticklabels"] = metrics_info["ticklabels"].fillna(metrics_info.apply(lambda metric_info: metric_info["ticks"], axis = 1))

# %%
panel_width = 4/4
panel_resolution = 1/4

fig, axes = plt.subplots(
    len(metrics_info),
    len(datasets_info),
    figsize=(len(datasets_info) * panel_width, len(metrics_info) * len(methods_info) * panel_resolution),
    gridspec_kw={"wspace": 0.05, "hspace": 0.2},
    squeeze=False,
)

for dataset, dataset_info in datasets_info.iterrows():
    axes_dataset = axes[:, dataset_info["ix"]].tolist()
    for metric, metric_info in metrics_info.iterrows():
        ax = axes_dataset.pop(0)
        ax.set_xlim(metric_info["limits"])
        plotdata = pd.DataFrame(index = pd.MultiIndex.from_tuples([dataset], names = datasets_info.index.names)).join(meanscores).reset_index()
        plotdata = pd.merge(plotdata, methods_info, on = methods_info.index.names)
        
        ax.barh(
            width=plotdata[metric],
            y=plotdata["ix"],
            color=plotdata["color"],
            lw=0,
            zorder = 0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        
        # out of limits values
        metric_limits = metric_info["limits"]
        plotdata_annotate = plotdata.loc[(plotdata[metric] < metric_limits[0]) | (plotdata[metric] > metric_limits[1])]
        transform = mpl.transforms.blended_transform_factory(
            ax.transAxes,
            ax.transData
        )
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x = 0.03 if left else 0.97,
                y = plotdata_row["ix"],
                s = f"{plotdata_row[metric]:+.2f}",
                transform = transform,
                va = "center",
                ha = "left" if left else "right",
                color = "#FFFFFFCC",
                fontsize = 6
            )

# Datasets
for dataset, dataset_info in datasets_info.iterrows():
    ax = axes[0, dataset_info["ix"]]
    ax.set_title(dataset_info["label"], fontsize = 8)
    
# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[metric_info["ix"], 0]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])
    
    ax = axes[metric_info["ix"], 1]
    ax.set_xlabel(metric_info["label"])
    
# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"])
    
for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min()-0.5, 0.5)

# %% [markdown]
# ----

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
