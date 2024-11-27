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
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm
import pickle

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
from chromatinhd_manuscript.designs_diff import dataset_latent_method_combinations as design_chd
design_chd = design_chd.query("method == 'v31'").query("splitter == '5x1'")
from chromatinhd_manuscript.designs_diff import dataset_latent_peakcaller_diffexp_combinations as design_diffexp
from chromatinhd_manuscript.designs_motif import design as design_motif

design = pd.concat([
    design_chd.rename(columns = {"method":"reference_method", "splitter":"reference_splitter"}).merge(design_chd),
    design_chd.rename(columns = {"method":"reference_method", "splitter":"reference_splitter"}).merge(design_diffexp)
], ignore_index = True).merge(design_motif)

# design = design.query("regions == '10k10k'")
design = design.query("regions == '100k100k'")
design = design.query("dataset != 'hepatocytes'")
design = design.query("dataset != 'alzheimer'")
design = design.query("dataset != 'pbmc20k'")
# design = design.query("dataset == 'pbmc10k'")

design["method"] = [f"{row.peakcaller}/{row.diffexp}" if not pd.isnull(row.peakcaller) else "chd" for _, row in design.iterrows()]

design.shape

# %% [markdown]
# ## Define motifs of interest

# %%
regions_name = "100k100k"

# %%
motifs_oi_datasets = {}
diffexps = {}

for _, (dataset_name, latent, motifscan_name, organism) in design[["dataset", "latent", "motifscan", "organism"]].drop_duplicates().iterrows():
    motifscan_name = "hocomocov12_1e-4"
    motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
    if organism == "mm":
        motifscan.motifs["symbol"] = motifscan.motifs["MOUSE_gene_symbol"]
    else:
        motifscan.motifs["symbol"] = motifscan.motifs["HUMAN_gene_symbol"]

    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    # pure fold change
    cluster_transcriptome = pd.DataFrame(transcriptome.layers["magic"][:], index = transcriptome.obs.index, columns = transcriptome.var.index).groupby(clustering.labels).mean()
    cluster_transcriptome.index.name = "cluster"
    diffexp = cluster_transcriptome - cluster_transcriptome.mean(0)
    diffexp = (cluster_transcriptome - cluster_transcriptome.mean(0)).T.unstack().to_frame(name = "score")
    diffexp["significant_up"] = diffexp["score"] > np.log(1.2)
    diffexp["significant_down"] = diffexp["score"] < -np.log(1.2)
    diffexp["significant"] = diffexp["significant_up"] | diffexp["significant_down"]

    # t-test
    import scanpy as sc
    adata_raw = transcriptome.adata.raw.to_adata()
    adata_raw = adata_raw[:, transcriptome.var.index]
    adata_raw.obs["cluster"] = clustering.labels
    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)
    import scanpy as sc
    sc.tl.rank_genes_groups(adata_raw, groupby="cluster", method="t-test")

    diffexp = sc.get.rank_genes_groups_df(adata_raw, None).rename(columns = {"names":"gene", "group":"cluster"}).set_index(["cluster", "gene"])
    diffexp["significant_up"] = (diffexp["pvals_adj"] < 0.05) & (diffexp["scores"] > 10)
    diffexp["significant_down"] = (diffexp["pvals_adj"] < 0.05) & (diffexp["scores"] < -10)

    diffexp["significant"] = diffexp["significant_up"] | diffexp["significant_down"]
    diffexp["score"] = diffexp["scores"]

    motifs_oi = motifscan.motifs.sort_values("quality").copy().reset_index().groupby("symbol").first().reset_index().set_index("motif")
    motifs_oi["gene"] = [transcriptome.gene_id(symbol) if symbol in transcriptome.var["symbol"].tolist() else None for symbol in motifs_oi["symbol"]]
    motifs_oi = motifs_oi.dropna(subset=["gene"])

    print(dataset_name, latent, len(motifs_oi),)
    motifs_oi_datasets[(dataset_name, latent, motifscan_name)] = motifs_oi
    diffexps[(dataset_name, latent, motifscan_name)] = diffexp

# %% [markdown]
# ## Load scores

# %%
enrichments = {}
for design_ix, design_row in tqdm.tqdm(design.iterrows(), total=len(design)):
    dataset_name = design_row["dataset"]
    regions_name = design_row["regions"]
    peakcaller = design_row["peakcaller"]
    diffexp = design_row["diffexp"]
    motifscan_name = design_row["motifscan"]

    if pd.isnull(peakcaller):
        enrichment_folder = (
            chd.get_output()
            / "diff"
            / dataset_name
            / regions_name
            / "5x1"
            / "v31"
            / "scoring"
            / "regionpositional"
            / "differential"
            # / "0-1.5"
            / "-1-3"
            / "enrichment"
            / motifscan_name
        )
    else:
        enrichment_folder = (
            chd.get_output()
            / "diff"
            / dataset_name
            / regions_name
            / "5x1"
            / "v31"
            / "scoring"
            / "regionpositional"
            / "differential"
            # / "0-1.5"
            / "-1-3"
            / "enrichment"
            / motifscan_name
            / peakcaller
            / diffexp
        )

    try:
        enrichment = pickle.load((enrichment_folder / "enrichment.pkl").open("rb"))
        enrichments[design_ix] = enrichment
    except FileNotFoundError:
        design.loc[design_ix, "found"] = False
        continue
# scores = pd.DataFrame.from_dict(scores, orient="index")
# scores.index.name = "design_ix"

# %%
from collections import defaultdict
dataset_enrichments = defaultdict(list)
for design_ix, enrichment in enrichments.items():
    # if design.loc[design_ix, "method"] in ["chd", "macs2_leiden_0.1/t-test"]:
    # if True:
    if design.loc[design_ix, "method"] in ["chd", "encode_screen/t-test"]:
    # if design.loc[design_ix, "method"] in ["encode_screen/t-test"]:
    # if design.loc[design_ix, "method"] in ["macs2_leiden_0.1/t-test"]:
        (dataset_name, latent, motifscan_name) = design.loc[design_ix, ["dataset", "latent", "motifscan"]]
        dataset_enrichments[(dataset_name, latent, motifscan_name)].append(enrichment.assign(design_ix = design_ix))

dataset_enrichments = {k:pd.concat(v) for k,v in dataset_enrichments.items()}

# %%
motifclustermappings = {}
for _, (dataset_name, latent, motifscan_name, organism) in design[["dataset", "latent", "motifscan", "organism"]].drop_duplicates().iterrows():
    diffexp = diffexps[(dataset_name, latent, motifscan_name)]
    motifs_oi = motifs_oi_datasets[(dataset_name, latent, motifscan_name)]
    if (dataset_name, latent, motifscan_name) not in dataset_enrichments:
        continue
    dataset_enrichment = dataset_enrichments[(dataset_name, latent, motifscan_name)]
    dataset_enrichment["significant_qval"] = dataset_enrichment["q_value"] < 0.05
    dataset_enrichment["significant"] = dataset_enrichment["significant_qval"]

    enrichment = dataset_enrichment.groupby(["cluster", "motif"]).mean(numeric_only = True)
    enrichment["n_significant"] = dataset_enrichment.groupby(["cluster", "motif"])["significant"].sum()

    enrichment["gene"] = motifs_oi["gene"].reindex(enrichment.index.get_level_values("motif")).values
    enrichment = enrichment.dropna(subset = ["gene"])
    enrichment = enrichment.reset_index().set_index(["cluster", "gene"]).join(diffexp, rsuffix = "_diffexp")

    motifclustermappings[(dataset_name, latent, motifscan_name)] = enrichment

# %% [markdown]
# ## Score

# %%
import scipy.stats

def score_diffexp_enrichment(enrichment:pd.DataFrame, diffexp:pd.DataFrame, motifs_oi):
    """
    Compares the differential expression of TFs with their differential enrichment
    """
    if "cluster" not in enrichment.index.names:
        raise ValueError("enrichment must contain a level 'cluster' in the index")
    if "gene" not in motifs_oi.columns:
        raise ValueError("motifs_oi must contain a column 'gene' with the gene id of the motif")

    scores = []
    subscores = []
    clusters = enrichment.index.get_level_values("cluster").unique()
    for cluster in clusters:
        # subscore contains the differential expression and enrichment of the TFs in the cluster
        subscore = pd.DataFrame(
            {
                "score": diffexp.loc[cluster].loc[motifs_oi["gene"]]["score"],
                "odds": enrichment.loc[cluster].loc[motifs_oi.index]["odds"].values,
                "significant": diffexp.loc[cluster].loc[motifs_oi["gene"]]["significant"].values,
            }
        )
        subscore["logodds"] = np.log(subscore["odds"])
        # subscore["logodds"] = np.clip(subscore["logodds"], -2, 2)
        subscore = subscore.dropna()

        # select differential TFs
        subscore = subscore.loc[subscore["significant"]]
        # subscore = subscore.sort_values("odds").iloc[:5]

        contingency = (
            np.array(
                [
                    [
                        subscore.query("score > 0").query("logodds > 0").shape[0],
                        subscore.query("score > 0").query("logodds < 0").shape[0],
                    ],
                    [
                        subscore.query("score < 0").query("logodds > 0").shape[0],
                        subscore.query("score < 0").query("logodds < 0").shape[0],
                    ],
                ]
            )
            + 1
        )
        if len(subscore) > 4:
            odds = (contingency[1, 1] * contingency[0, 0] + 1) / (contingency[1, 0] * contingency[0, 1] + 1)

            if (subscore["score"].std() == 0) or (subscore["logodds"].std() == 0):
                cor = 0
                spearman = 0
            else:
                cor = np.corrcoef(subscore["score"], subscore["logodds"])[0, 1]
                spearman = scipy.stats.spearmanr(subscore["score"], subscore["logodds"])[0]
                
            log_avg_odds = np.concatenate(
                [subscore.query("score > 5")["logodds"], -subscore.query("score < -5")["logodds"]]
            ).mean()

            log_avg_abs_odds = subscore["logodds"].abs().mean()
        else:
            cor = 0
            spearman = 0
            odds = 1
            log_avg_odds = 0.0
            log_avg_abs_odds = 0.0

        subscores.append(
            subscore.assign(
                cluster = cluster
            ).reset_index()
        )

        scores.append(
            {
                "cluster": cluster,
                "contingency": contingency,
                "cor": cor,
                "spearman": spearman,
                "odds": odds,
                "log_odds": np.log(odds),
                "log_avg_odds": log_avg_odds,
                "log_avg_abs_odds": log_avg_abs_odds,
                "avg_odds": np.exp(log_avg_odds),
            }
        )
    subscores = pd.concat(subscores).set_index(diffexp.index.names)
    if len(scores):
        scores = pd.DataFrame(scores).set_index("cluster")
    else:
        scores = pd.DataFrame(columns=["cluster", "odds", "log_odds", "cor"]).set_index("cluster")
    return scores, subscores


# %%
rawscores = {}
for design_ix, enrichment in tqdm.tqdm(enrichments.items()):
    design_row = design.loc[design_ix]
    motifs_oi = motifs_oi_datasets[(design_row["dataset"], design_row["latent"], design_row["motifscan"])]
    diffexp = diffexps[(design_row["dataset"], design_row["latent"], design_row["motifscan"])]

    rawscores[design_ix] = score_diffexp_enrichment(enrichment, diffexp, motifs_oi)[0]
rawscores = pd.concat(rawscores, names = ["design_ix"])

# remove clusters with <50 nuclei in hspc dataset
rawscores = rawscores.loc[rawscores.index.get_level_values("cluster") != "Plasma"]
rawscores = rawscores.loc[(rawscores.index.get_level_values("cluster") != "NK")]
rawscores = rawscores.loc[(rawscores.index.get_level_values("cluster") != "Unknown 1")]
rawscores = rawscores.loc[(rawscores.index.get_level_values("cluster") != "Unknown 2")]

# %%
design.index.name = "design_ix"

# %%
scores = rawscores.groupby("design_ix").mean(numeric_only = True)

# %%
# scores.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean()["spearman"].unstack().T.style.bar()

# %%
cors = scores.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean(numeric_only = True)["cor"].unstack().T
cors.style.bar(vmin = 1)

# %%
avg_odds = np.exp(scores.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean(numeric_only = True)["log_avg_odds"]).unstack().T
avg_odds.style.bar(vmin = 1)

# %%
plotdata = avg_odds.fillna(avg_odds.median(0))
plotdata = avg_odds
plotdata = plotdata.loc[plotdata.mean(1).sort_values().index]
fig, ax = plt.subplots()
plotdata.mean(1).plot.barh(ax=ax)
ax.set_xlim(1, 1.175)

# %%
# odds = np.exp(scores.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean(numeric_only = True)["log_odds"]).unstack().T
odds = scores.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean(numeric_only = True)["odds"].unstack().T
plotdata = odds.fillna(odds.median(0))
plotdata = plotdata.loc[plotdata.mean(1).sort_values().index]
fig, ax = plt.subplots()
plotdata.mean(1).plot.barh(ax=ax, width = 0.8, lw = 0)
ax.set_xlim(1)

# %%
cors = scores.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean(numeric_only = True)["cor"].unstack().T
plotdata = cors.fillna(cors.median(0))
plotdata = plotdata.loc[plotdata.mean(1).sort_values().index]
fig, ax = plt.subplots()
plotdata.mean(1).plot.barh(ax=ax)

# %%
main_features = ["dataset", "regions", "latent", "motifscan"]
plotdata = scores.join(design)

plotdata_mean = plotdata.groupby("method").mean(numeric_only = True).reset_index()
plotdata_mean["n"] = plotdata.groupby("method").size().values
plotdata_mean = plotdata_mean.loc[plotdata_mean["n"] == plotdata_mean["n"].max()]
plotdata_mean[main_features] = "mean"
plotdata = plotdata.loc[plotdata["method"].isin(plotdata_mean["method"])]

plotdata = pd.concat([
    plotdata,
    plotdata_mean
])

# %%
plotdata_groupby = plotdata.groupby(main_features)

# %%
reference_method = "chd"

# %%
y_info = pd.DataFrame(plotdata_groupby.groups.keys(), columns=main_features).set_index(main_features, drop = False)
y_info["label"] = y_info["regions"] + " | " + y_info["dataset"]
y_info.loc[y_info["dataset"] == "mean", "label"] = "Mean"
def determine_setting(x):
    if "-" in x:
        return "test"
    elif x == "mean":
        return "mean"
    else:
        return "test_cells"
y_info["setting"] = [determine_setting(x) for x in y_info["dataset"]]
y_info["sorter"] = pd.Series({"test":0, "test_cells":1, "mean":2})[y_info["setting"]].values + plotdata.query("method == @reference_method").set_index(main_features)["avg_odds"].loc[y_info.index]
y_info = y_info.sort_values("sorter")
y_info["ix"] = np.arange(len(y_info))

# %%
import chromatinhd_manuscript as chdm

# %%
method_info = chdm.methods.differential_methods
method_info.loc["chd", "type"] = "ours"

# %%
import matplotlib as mpl
import seaborn as sns
import chromatinhd.plot.quasirandom

fig, ax = plt.subplots(figsize=(4, len(plotdata_groupby) * 0.25))

score = "avg_odds"
# score = "cor"
# score = "odds"
# score = "r2"
# score = "scored"
# score = "cor"

for dataset, plotdata_dataset in plotdata_groupby:
    x = y_info.loc[dataset]["ix"]

    plotdata_dataset[score] = plotdata_dataset[score].fillna(0.)
    y = np.array(chd.plot.quasirandom.offsetr(np.array(plotdata_dataset[score].values.tolist()), adjust=0.1)) * 0.8 + x
    plotdata_dataset["y"] = y

    plotdata_dataset["color"] = method_info.loc[
        plotdata_dataset["method"], "color"
    ].values
    plotdata_dataset.loc[plotdata_dataset["method"] == "v32", "color"] = "pink"
    plotdata_dataset.loc[plotdata_dataset["method"] == "v31", "color"] = "turquoise"

    ax.axhspan(x - 0.5, x + 0.5, color="#33333308" if dataset != "mean" else "#33333311", ec="white", zorder = -2)
    ax.scatter(plotdata_dataset[score], plotdata_dataset["y"], s=5, color=plotdata_dataset["color"], lw=0)

    plotdata_dataset["type"] = method_info.loc[
        plotdata_dataset["method"], "type"
    ].values
    plotdata_top = plotdata_dataset.sort_values(score, ascending=False).groupby("type").first()
    for i, (type, plotdata_top_) in enumerate(plotdata_top.groupby("type")):
        ax.plot(
            [plotdata_top_[score]] * 2,
            [x - 0.45, x + 0.45],
            color=plotdata_top_["color"].values[0],
            lw=2,
            zorder=0,
            alpha=0.9,
            solid_capstyle = "butt",
        )
    plotdata_ours = plotdata_dataset.loc[plotdata_dataset["method"] == "chd"].iloc[0]
    plotdata_top_others = plotdata_top.loc[plotdata_top.index != "ours"]

    try:
        plotdata_others_max = plotdata_top_others.loc[plotdata_top_others[score].idxmax()]
        rectangle = mpl.patches.Rectangle(
            (plotdata_others_max[score], x-0.45),
            plotdata_ours[score] - plotdata_others_max[score],
            0.9,
            fc=plotdata_ours["color"] if plotdata_ours[score] > plotdata_others_max[score] else plotdata_others_max["color"],
            ec="none",
            zorder=-1,
            alpha=1/3,
        )
        ax.add_patch(rectangle)
    except ValueError:
        pass
ax.set_yticks(y_info["ix"])
ax.set_yticklabels(y_info["label"])
for ticklabel in ax.get_yticklabels():
    if ticklabel.get_text() == "Mean":
        ticklabel.set_fontweight("bold")
    else:
        ticklabel.set_fontsize(9)

if score == "avg_odds":
    ax.set_xlabel("Average odds ratio in TFBS of\ndifferentially expressed TFs")
    ax.set_xlim(1)
    ax.set_xscale("log")
    ax.set_xticks([1., 1.1, 1.2])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xticks([], minor = True)
elif score == "cor":
    ax.set_xlabel("Pearson $r$ between transcriptomics fold-change\nof a TF's gene and the TF's motif enrichment")
    ax.set_xlim(-0.07)
sns.despine(ax=ax, left=True, bottom=True)
ax.set_ylim(-0.5, y_info["ix"].max() + 0.5)

manuscript.save_figure(fig, "3", f"aggregate_motif_{score}")

# %%
plotdata_mean["peakcaller"] = plotdata_mean["method"].str.split("/").str[0]
plotdata_mean.to_csv(chd.get_output() / "aggregate_motif_enrichment.csv")

# %%
plotdata_datasets = plotdata.groupby(["method", "dataset"])[["log_avg_odds"]].mean().reset_index()
plotdata_datasets["peakcaller"] = plotdata_datasets["method"].str.split("/").str[0]
plotdata_datasets.to_csv(chd.get_output() / "aggregate_motif_enrichment_datasets.csv")

# %% [markdown]
# ## 2

# %%
scores2 = {}
for design_ix, enrichment in enrichments.items():
    design_row = design.loc[design_ix]
    motifclustermapping = motifclustermappings[(design_row["dataset"], design_row["latent"], design_row["motifscan"])].reset_index().set_index(["cluster", "motif"])
    # motifclustermapping["link"] =  (motifclustermapping["score"] > 5) & (motifclustermapping["pvals_adj"] < 0.05)
    # motifclustermapping["link"] = (motifclustermapping["log_odds"] > np.log(1.2)) & (motifclustermapping["score"] > 2) & (motifclustermapping["q_value"] < 0.05)
    # motifclustermapping["link"] = (motifclustermapping["log_odds"] > np.log(1.2)) & (motifclustermapping["score"] > 2) & (motifclustermapping["q_value"] < 0.05)
    motifclustermapping["link"] = (motifclustermapping["log_odds"] < np.log(1.2)) & (motifclustermapping["n_significant"] >= 1) & (motifclustermapping["q_value"] < 0.01)
    # motifclustermapping["link"] = (motifclustermapping["n_significant"] > 10) & (motifclustermapping["q_value"] < 0.05) & (motifclustermapping["scores"] > 0.0)
    # motifclustermapping["link"] = (motifclustermapping["log_odds"] < np.log(2/3))

    enrichment_oi = motifclustermapping[["logfoldchanges"]].join(enrichment)
    enrichment_linked = motifclustermapping[["link"]].join(enrichment).query("link == True")

    rawscore = {}
    rawscore["cor"] = enrichment_oi.groupby("cluster").apply(lambda x: np.corrcoef(x["logfoldchanges"], x["log_odds"])[0,1]).mean()
    rawscore["avg_odds"] = enrichment_linked["log_odds"].mean()
    scores2[design_ix] = rawscore

scores2 = pd.DataFrame.from_dict(scores2, orient="index")
scores2.index.name = "design_ix"

# %%
cors = scores2.join(design).groupby(["dataset", "regions", "latent", "motifscan", "method"]).mean(numeric_only = True)["avg_odds"].unstack().T
plotdata = cors.fillna(cors.median(0))
plotdata = plotdata.loc[plotdata.mean(1).sort_values().index]
fig, ax = plt.subplots()
plotdata.mean(1).plot.barh(ax=ax)
# ax.set_xlim(1)
