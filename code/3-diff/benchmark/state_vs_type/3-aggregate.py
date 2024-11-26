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
# %config InlineBackend.figure_format = 'retina'

import pickle

import scanpy as sc
import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %%
from design import dataset_pair_combinations

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
regions_name = "100k100k"
peakcaller = "macs2_summits"
latent = "leiden_0.1"

# %%
import pybedtools

# %%
# differential = "-1-1.5"
differential = "-1-3"

intersections = []
cors = []

for _, row in dataset_pair_combinations.iterrows():
    dataset_name = row["dataset"]
    celltype_a = row["celltype_a"]
    celltype_b = row["celltype_b"]

    print(dataset_name, celltype_a, celltype_b)

    clusters_oi = [celltype_a, celltype_b]

    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
    fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    # load chd
    folder = chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional" / f"differential_{celltype_a}_{celltype_b}" / differential

    scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / "t-test" / f"{clusters_oi[0]}_{clusters_oi[1]}scoring" / "regionpositional"
    scoring_folder.mkdir(parents=True, exist_ok=True)

    differential_slices = pickle.load(open(folder / "differential_slices.pkl", "rb"))

    slicescores = differential_slices.get_slice_scores(regions = fragments.regions, clustering = clustering, cluster_info = clustering.cluster_info.loc[clusters_oi])

    slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
    slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

    n_desired_positions = slicescores.groupby("cluster")["length"].sum()

    # load peak
    differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))
    differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
    differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
    differential_slices_peak.window = fragments.regions.window

    # match # of differential within each cluster
    slicescores_peak_full = differential_slices_peak.get_slice_scores(regions = fragments.regions, clustering = clustering)
    slicescores_peak_full = slicescores_peak_full.loc[slicescores_peak_full["cluster"].isin(clusters_oi)]
    slicescores_peak_full["cluster"] = pd.Categorical(slicescores_peak_full["cluster"], categories = clusters_oi)
    slicescores_peak = []
    for cluster in n_desired_positions.index:
        peakscores_cluster = slicescores_peak_full.query("cluster == @cluster")
        peakscores_cluster = peakscores_cluster.sort_values("score", ascending=False)
        # peakscores_cluster = peakscores_cluster.sort_values("logfoldchanges", ascending=False)
        n_desired_positions_cluster = n_desired_positions[cluster]

        # peakscores_cluster["cumulative_length"] = peakscores_cluster["length"].cumsum() # at the latest as large
        peakscores_cluster["cumulative_length"] = np.pad(peakscores_cluster["length"].cumsum()[:-1], (1, 0)) # at least as large

        peakscores_cluster = peakscores_cluster.query("cumulative_length <= @n_desired_positions_cluster")
        slicescores_peak.append(peakscores_cluster)
    slicescores_peak = pd.concat(slicescores_peak)
    slicescores_peak["slice"] = pd.Categorical(slicescores_peak["region"].astype(str) + ":" + slicescores_peak["start"].astype(str) + "-" + slicescores_peak["end"].astype(str))
    slices_peak = slicescores_peak.groupby("slice")[["region", "start", "end"]].first()

    # slices["region_ix"] = fragments.var.index.get_indexer(slices["region"])
    slices_peak["region_ix"] = fragments.var.index.get_indexer(slices_peak["region"])

    pr1 = pybedtools.BedTool.from_dataframe(pd.concat([slices_peak[["region_ix"]], slices_peak[["start", "end"]] - fragments.regions.window[0]], axis = 1))
    pr2 = pybedtools.BedTool.from_dataframe(pd.concat([ slices[["region_ix"]], slices[["start", "end"]] - fragments.regions.window[0]], axis = 1))

    intersect = pr1.intersect(pr2)
    intersect = intersect.to_dataframe()

    if len(intersect) == 0:
        jaccard = 0
    else:
        intersect["size"] = intersect["end"] - intersect["start"]

        union = pr1.cat(pr2, postmerge=False).sort().merge()
        union = union.to_dataframe()
        union["size"] = union["end"] - union["start"]

        jaccard = intersect["size"].sum() / union["size"].sum()

    intersections.append({
        "dataset": dataset_name,
        "celltype_a": celltype_a,
        "celltype_b": celltype_b,
        "jaccard": jaccard,
        "total": union["size"].sum(),
    })

    # cor
    # compare using expression
    import scanpy as sc
    adata_raw = transcriptome.adata.raw.to_adata()
    adata_raw.obs["celltype"] = clustering.labels
    adata_raw = adata_raw[adata_raw.obs["celltype"].isin(clusters_oi), transcriptome.var.index].copy()
    adata_raw.obs["cluster"] = clustering.labels
    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)

    corr = pd.DataFrame(adata_raw.X.todense(), index=adata_raw.obs.index, columns=adata_raw.var.index).groupby(clustering.labels).mean().T.corr()
    cors.append(pd.DataFrame({
        "celltype_a":[celltype_a],
        "celltype_b":celltype_b,
        "dataset":dataset_name,
        "cor":corr.loc[celltype_a, celltype_b],
    }))

intersections = pd.DataFrame(intersections).sort_values("jaccard").set_index(["dataset", "celltype_a", "celltype_b"])
cors = pd.concat(cors).set_index(["dataset", "celltype_a", "celltype_b"])

# %%
intersections["cor"] = cors["cor"]

# %%
intersections.sort_values("cor")

# %%
fig, ax = plt.subplots(figsize = (4, 2))
sns.regplot(x = intersections["cor"], y = intersections["jaccard"], color = "#333", scatter_kws = {"s": 30, "alpha": 1.0, "lw":0}, ax = ax)
ax.set_xlabel("Correlation between gene expression")
ax.set_ylabel("Jaccard\nbetween DARs", rotation = 0, ha = "right", va = "center")

intersections_oi = intersections.loc[[
    ("hspc", "HSPC", "MPP"),
    ("lymphoma", "B", "Lymphoma"),
    ("pbmc10k", "CD14+ Monocytes", "CD4 memory T")
]]
for (dataset, celltype_a, celltype_b), row in intersections_oi.iterrows():
    ax.annotate(
        f"{dataset}\n{celltype_a}\n{celltype_b}",
        xy = (row["cor"], row["jaccard"]),
        xytext = (row["cor"], 0.9),
        textcoords = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), ha = "center", va = "center", arrowprops = dict(edgecolor = "black", arrowstyle = "-"))

ax.set_ylim(0, 0.5)
sns.despine(ax = ax)

manuscript.save_figure(fig, "3", "statevstype_cor_vs_overlap")

# %% [markdown]
# ## Check difference in enrichment

# %%
# run and store the enrichment

differential = "-1-1.5"
# differential = "-1-3"

# force = True
force = False

for _, row in dataset_pair_combinations.iterrows():
    dataset_name = row["dataset"]
    celltype_a = row["celltype_a"]
    celltype_b = row["celltype_b"]

    print(dataset_name, celltype_a, celltype_b)

    clusters_oi = [celltype_a, celltype_b]

    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
    fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    # load chd
    folder = chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional" / f"differential_{celltype_a}_{celltype_b}" / differential

    scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / "t-test" / f"{clusters_oi[0]}_{clusters_oi[1]}scoring" / "regionpositional"
    scoring_folder.mkdir(parents=True, exist_ok=True)

    if force or ((not (folder / "enrichment.pkl").exists()) and (not (scoring_folder / "enrichment.pkl").exists())):
        differential_slices = pickle.load(open(folder / "differential_slices.pkl", "rb"))

        slicescores = differential_slices.get_slice_scores(regions = fragments.regions, clustering = clustering, cluster_info = clustering.cluster_info.loc[clusters_oi])

        slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
        slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

        n_desired_positions = slicescores.groupby("cluster")["length"].sum()

        motifscan_name = "hocomocov12_1e-4"
        motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

        slicecounts = motifscan.count_slices(slices)
        enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores, slicecounts)
        enrichment["log_odds"] = np.log(enrichment["odds"])

        # store chd enrichment
        pickle.dump(enrichment, open(folder / "enrichment.pkl", "wb"))

        # load peak
        differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))
        differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
        differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
        differential_slices_peak.window = fragments.regions.window

        # match # of differential within each cluster
        slicescores_peak_full = differential_slices_peak.get_slice_scores(regions = fragments.regions, clustering = clustering)
        slicescores_peak_full = slicescores_peak_full.loc[slicescores_peak_full["cluster"].isin(clusters_oi)]
        slicescores_peak_full["cluster"] = pd.Categorical(slicescores_peak_full["cluster"], categories = clusters_oi)
        slicescores_peak = []
        for cluster in n_desired_positions.index:
            peakscores_cluster = slicescores_peak_full.query("cluster == @cluster")
            peakscores_cluster = peakscores_cluster.sort_values("score", ascending=False)
            # peakscores_cluster = peakscores_cluster.sort_values("logfoldchanges", ascending=False)
            n_desired_positions_cluster = n_desired_positions[cluster]

            # peakscores_cluster["cumulative_length"] = peakscores_cluster["length"].cumsum() # at the latest as large
            peakscores_cluster["cumulative_length"] = np.pad(peakscores_cluster["length"].cumsum()[:-1], (1, 0)) # at least as large

            peakscores_cluster = peakscores_cluster.query("cumulative_length <= @n_desired_positions_cluster")
            slicescores_peak.append(peakscores_cluster)
        slicescores_peak = pd.concat(slicescores_peak)
        slicescores_peak["slice"] = pd.Categorical(slicescores_peak["region"].astype(str) + ":" + slicescores_peak["start"].astype(str) + "-" + slicescores_peak["end"].astype(str))
        slices_peak = slicescores_peak.groupby("slice")[["region", "start", "end"]].first()

        slicecounts_peak = motifscan.count_slices(slices_peak)
        enrichment_peak = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores_peak, slicecounts_peak)
        enrichment_peak["log_odds"] = np.log(enrichment_peak["odds"])

        # store enrichment peak
        pickle.dump(enrichment_peak, open(scoring_folder / "enrichment.pkl", "wb"))

# %% [markdown]
# Join with diffexp

# %%
scores = []
scores_peak = []

scores_full = []
scores_full_peak = []

for _, row in dataset_pair_combinations.iterrows():
    dataset_name = row["dataset"]
    celltype_a = row["celltype_a"]
    celltype_b = row["celltype_b"]

    print(dataset_name, celltype_a, celltype_b)

    clusters_oi = [celltype_a, celltype_b]

    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
    fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    folder = chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional" / f"differential_{celltype_a}_{celltype_b}" / differential

    scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / "t-test" / f"{clusters_oi[0]}_{clusters_oi[1]}scoring" / "regionpositional"
    scoring_folder.mkdir(parents=True, exist_ok=True)

    # load
    try:
        enrichment = pickle.load(open(folder / "enrichment.pkl", "rb"))
        enrichment_peak = pickle.load(open(scoring_folder / "enrichment.pkl", "rb"))
    except FileNotFoundError:
        continue

    motifscan_name = "hocomocov12_1e-4"
    motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

    if dataset_name in ["e18brain", "liver"]:
        motifscan.motifs["gene"] = transcriptome.var.reset_index().groupby("symbol").first().reindex(motifscan.motifs["MOUSE_gene_symbol"])["gene"].values
    else:
        motifscan.motifs["gene"] = transcriptome.var.reset_index().set_index("symbol").reindex(motifscan.motifs["HUMAN_gene_symbol"])["gene"].values

    # compare using expression
    import scanpy as sc
    adata_raw = transcriptome.adata.raw.to_adata()
    adata_raw.obs["celltype"] = clustering.labels
    adata_raw = adata_raw[adata_raw.obs["celltype"].isin(clusters_oi), transcriptome.var.index].copy()
    adata_raw.obs["cluster"] = clustering.labels
    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)

    sc.tl.rank_genes_groups(adata_raw, groupby="cluster", groups = [clusters_oi[0]], reference = clusters_oi[1], method="t-test", key_added = "lymphoma_vs_cycling")
    lymphoma_vs_cycling = pd.DataFrame({
        "gene":adata_raw.uns["lymphoma_vs_cycling"]["names"].tolist(),
        "scores":adata_raw.uns["lymphoma_vs_cycling"]["scores"].tolist(),
        "pvals_adj":adata_raw.uns["lymphoma_vs_cycling"]["pvals_adj"].tolist(),
        "logfoldchanges":adata_raw.uns["lymphoma_vs_cycling"]["logfoldchanges"].tolist(),
    }).apply(lambda x:x.str[0]).set_index("gene")

    scores_full.append(enrichment.assign(celltype_a = celltype_a, celltype_b = celltype_b, dataset = dataset_name))
    scores_full_peak.append(enrichment_peak.assign(celltype_a = celltype_a, celltype_b = celltype_b, dataset = dataset_name))

    # compare
    enrichment_joined = enrichment.loc[clusters_oi[0]].loc[motifscan.motifs.loc[enrichment.loc[clusters_oi[0]].index, "gene"].dropna().index]
    enrichment_joined = enrichment_joined.join(motifscan.motifs.loc[enrichment_joined.index, "gene"])
    enrichment_joined[["scores", "pvals_adj", "logfoldchanges"]] = lymphoma_vs_cycling[["scores", "pvals_adj", "logfoldchanges"]].loc[enrichment_joined["gene"]].values

    enrichment_peak_joined = enrichment_peak.loc[clusters_oi[0]].loc[motifscan.motifs.loc[enrichment_peak.loc[clusters_oi[0]].index, "gene"].dropna().index]
    enrichment_peak_joined = enrichment_peak_joined.join(motifscan.motifs.loc[enrichment_peak_joined.index, "gene"])
    enrichment_peak_joined[["scores", "pvals_adj", "logfoldchanges"]] = lymphoma_vs_cycling[["scores", "pvals_adj", "logfoldchanges"]].loc[enrichment_peak_joined["gene"]].values

    score_cutoff = 0

    scores.append(enrichment_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff").assign(direction = 1).assign(celltype_a = celltype_a, celltype_b = celltype_b, dataset = dataset_name))
    scores.append(enrichment_joined.query("pvals_adj < 0.05").query("scores < -@score_cutoff").assign(direction = -1).assign(celltype_a = celltype_a, celltype_b = celltype_b, dataset = dataset_name))

    scores_peak.append(enrichment_peak_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff").assign(direction = 1).assign(celltype_a = celltype_a, celltype_b = celltype_b, dataset = dataset_name))
    scores_peak.append(enrichment_peak_joined.query("pvals_adj < 0.05").query("scores < -@score_cutoff").assign(direction = -1).assign(celltype_a = celltype_a, celltype_b = celltype_b, dataset = dataset_name))

# %%
scores = pd.concat(scores)
scores_peak = pd.concat(scores_peak)

# %%
scores_full = pd.concat(scores_full)
scores_full_peak = pd.concat(scores_full_peak)

# %% [markdown]
# ### All differentially expressed

# %%
scores.loc[scores["q_value"] > 0.05, "log_odds"] = 0.
scores_peak.loc[scores_peak["q_value"] > 0.05, "log_odds"] = 0.

# %%
scores_cutoff = 0.
(scores).query("direction == 1").query("scores > @scores_cutoff")["log_odds"].mean(), (scores_peak).query("direction == 1").query("scores > @scores_cutoff")["log_odds"].mean()

# %%
scores_cutoff = 0.
(scores).query("direction == -1").query("scores < -@scores_cutoff")["log_odds"].mean(), (scores_peak).query("direction == -1").query("scores < -@scores_cutoff")["log_odds"].mean()

# %%
scores_cutoff = 0.

scoring = (scores).query("direction == 1").query("scores > @scores_cutoff").groupby(["dataset", "celltype_a", "celltype_b"])["log_odds"].mean() - (scores).query("direction == -1").query("scores < @scores_cutoff").groupby(["dataset", "celltype_a", "celltype_b"])["log_odds"].mean()
scoring_peak = (scores_peak).query("direction == 1").query("scores > @scores_cutoff").groupby(["dataset", "celltype_a", "celltype_b"])["log_odds"].mean() - (scores_peak).query("direction == -1").query("scores < @scores_cutoff").groupby(["dataset", "celltype_a", "celltype_b"])["log_odds"].mean()

# %%
diffscores = (scoring - scoring_peak).to_frame("diff")
differential = scores_full.groupby(["dataset", "celltype_a", "celltype_b"]).first()["contingency"].str[1].str[0]
diffscores["differential"] = differential
diffscores["cor"] = cors["cor"]

# %%
fig, ax = plt.subplots()
ax.scatter(np.log1p(diffscores["differential"]), y = diffscores["diff"])

# %% [markdown]
# ### Joined

# %%
scores_full = scores_full.query("celltype_a == cluster").reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"])
scores_full_peak = scores_full_peak.query("celltype_a == cluster").reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"])

# %%
scores_all = scores_full.join(scores_full_peak, lsuffix = "_chd", rsuffix = "_peak")
scores_all.loc[scores_all["q_value_chd"] > 0.001, "odds_chd"] = 1.
scores_all.loc[scores_all["q_value_peak"] > 0.001, "odds_peak"] = 1.
scores_all["odds"] = scores_all["odds_peak"] * scores_all["odds_chd"]
# scores_all["odds"] = scores_all["odds_chd"]
scores_all.loc[scores_all["q_value_chd"] > 0.001, "logodds_chd"] = 0
scores_all.loc[scores_all["q_value_peak"] > 0.001, "logodds_peak"] = 0

# %%
n = 20

dataset_motif_differential_mapping = pd.concat([
    scores_all.dropna(subset = "odds").sort_values("odds").reset_index().groupby(["dataset", "celltype_a", "celltype_b"]).head(n = n).reset_index()[["dataset", "celltype_a", "celltype_b", "motif"]].assign(lr = -1),
    scores_all.dropna(subset = "odds").sort_values("odds").reset_index().groupby(["dataset", "celltype_a", "celltype_b"]).tail(n = n).reset_index()[["dataset", "celltype_a", "celltype_b", "motif"]].assign(lr = 1),
])

# %%
scores_oi = dataset_motif_differential_mapping.set_index(["dataset", "celltype_a", "celltype_b", "motif"]).join(
    scores_full.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"])
)
scores_oi["final_score"] = scores_oi["log_odds"] * scores_oi["lr"]
print(np.exp((scores_oi["final_score"]).mean()))
scores_oi

# %%
scores_oi_peak = dataset_motif_differential_mapping.set_index(["dataset", "celltype_a", "celltype_b", "motif"]).join(
    scores_full_peak.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"])
)
scores_oi_peak["final_score"] = scores_oi_peak["log_odds"] * scores_oi_peak["lr"]
print(np.exp((scores_oi_peak["final_score"]).mean()))
scores_oi_peak

# %%
scores.groupby(["dataset", "celltype_a", "celltype_b"]).apply(lambda x:np.abs(x["log_odds"]).mean()).mean()

# %%
scores_peak.groupby(["dataset", "celltype_a", "celltype_b"]).apply(lambda x:np.abs(x["log_odds"]).mean()).mean()

# %%
scores_full = scores_full.query("celltype_b != 'Lymphoma cycling'")
scores_full_peak = scores_full_peak.query("celltype_b != 'Lymphoma cycling'")

# %%
differential = scores_full.groupby(["dataset", "celltype_a", "celltype_b"]).first()["contingency"].str[1].str[0]

# %%
diffscores = (scores_oi["final_score"] - scores_oi_peak["final_score"]).to_frame("diff")
diffscores["final_score_chd"] = scores_oi["final_score"]
diffscores["final_score_peak"] = scores_oi_peak["final_score"]
diffscores["log_odds_chd"] = scores_oi["log_odds"]
diffscores["differential"] = differential
diffscores["cor"] = cors["cor"]

# %%
fig, ax = plt.subplots()
ax.scatter(np.log(diffscores["differential"]), y = diffscores["diff"])
sns.regplot(x = np.log1p(diffscores["differential"]), y = diffscores["diff"])
ax.set_ylim(-1, 1)

# %%
fig, ax = plt.subplots()
sns.regplot(x = np.log1p(diffscores["differential"]), y = diffscores["final_score_chd"])

sns.regplot(x = np.log1p(diffscores["differential"]), y = diffscores["final_score_peak"])
ax.set_ylim(-1, 5)

# %%
fig, ax = plt.subplots()
sns.regplot(x = (diffscores["cor"]), y = diffscores["final_score_chd"])

sns.regplot(x = (diffscores["cor"]), y = diffscores["final_score_peak"])
ax.set_ylim(-1, 5)

# %%
fig, ax = plt.subplots()
ax.scatter((diffscores["cor"]), y = diffscores["diff"])
sns.regplot(x = (diffscores["cor"]), y = diffscores["diff"])
ax.set_ylim(-1, 2)

# %% [markdown]
# ### Fixed

# %%
dataset_motif_differential_mapping = pd.DataFrame([
    ["pbmc10k", "CD4 naive T", "CD4 memory T", motifscan.select_motif("TBX21"), -1],
    ["pbmc10k", "CD4 naive T", "CD4 memory T", motifscan.select_motif("TCF7"), 1],
    
    # ["pbmc10k", "naive B", "memory B", motifscan.select_motif("PAX5"), 1],
    # ["pbmc10k", "naive B", "memory B", motifscan.select_motif("PRDM1"), -1],

    ["lymphoma", "B", "Lymphoma", motifscan.select_motif("PO2F2"), -1],
    ["lymphoma", "B", "Lymphoma", motifscan.select_motif("BCL6"), 1],

    ["lymphoma", "B", "T", motifscan.select_motif("PAX5"), 1],
    ["lymphoma", "B", "T", motifscan.select_motif("TBX21"), -1],

    ["lymphoma", "Lymphoma", "Lymphoma cycling", motifscan.select_motif("E2F1"), -1],

    ["hspc", "HSPC", "MPP", motifscan.select_motif("TAL1"), 1],
    ["hspc", "HSPC", "MPP", motifscan.select_motif("SPI1"), -1],

    ["hspc", "HSPC", "GMP", motifscan.select_motif("TAL1"), 1],
    ["hspc", "HSPC", "GMP", motifscan.select_motif("SPI1"), -1],

    ["hspc", "HSPC", "MEP", motifscan.select_motif("GATA1"), -1],
    ["hspc", "HSPC", "MEP", motifscan.select_motif("TAL1"), -1],

    ["hspc", "Erythroblast", "Erythrocyte precursors", motifscan.select_motif("GATA1"), 1],
    ["hspc", "Erythroblast", "Erythrocyte precursors", motifscan.select_motif("FLI1"), -1],

    ["hspc", "Erythroblast", "Megakaryocyte", motifscan.select_motif("NFE2"), -1],
    ["hspc", "Erythroblast", "Megakaryocyte", motifscan.select_motif("GATA1"), 1],
    
    ["pbmc10k", "CD14+ Monocytes", "CD4 memory T", motifscan.select_motif("SPI1"), 1],
    ["pbmc10k", "CD14+ Monocytes", "CD4 memory T", motifscan.select_motif("FOXO1"), -1], 
    
    ["pbmc10k", "FCGR3A+ Monocytes", "CD14+ Monocytes", motifscan.select_motif("NR4A1"), 1],
    ["pbmc10k", "FCGR3A+ Monocytes", "CD14+ Monocytes", motifscan.select_motif("CEBPB"), -1],
    
    # ["pbmc10k", "CD14+ Monocytes", "CD4 memory T", motifscan.select_motif("FOXO1"), -1], 

], columns = ["dataset", "celltype_a", "celltype_b", "motif", "lr"])

# %%
scores_oi = dataset_motif_differential_mapping.set_index(["dataset", "celltype_a", "celltype_b", "motif"]).join(scores_full.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"]))
np.exp((scores_oi["log_odds"] * scores_oi["lr"]).mean())

# %%
scores_oi = dataset_motif_differential_mapping.set_index(["dataset", "celltype_a", "celltype_b", "motif"]).join(
    scores_full.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"])
)
scores_oi.loc[scores_oi["q_value"] > 0.001, "odds"] = 1.
scores_oi.loc[scores_oi["q_value"] > 0.001, "log_odds"] = 0.
scores_oi["final_score"] = scores_oi["log_odds"] * scores_oi["lr"]
print(np.exp((scores_oi["final_score"]).mean()))
scores_oi

# %%
scores_oi_peak = dataset_motif_differential_mapping.set_index(["dataset", "celltype_a", "celltype_b", "motif"]).join(
    scores_full_peak.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"])
)
scores_oi_peak.loc[scores_oi_peak["q_value"] > 0.001, "odds"] = 1.
scores_oi_peak.loc[scores_oi_peak["q_value"] > 0.001, "log_odds"] = 0.
scores_oi_peak["final_score"] = scores_oi_peak["log_odds"] * scores_oi_peak["lr"]
print(np.exp((scores_oi_peak["final_score"]).median()))
scores_oi_peak

# %%
differential = scores_full.groupby(["dataset", "celltype_a", "celltype_b"]).first()["contingency"].str[1].str[0]

# %%
diffscores = scores_oi[["final_score"]] - scores_oi_peak[["final_score"]]
diffscores.columns = ["diff_final_score"]
diffscores["differential"] = differential

# %%
diffscores.sort_values("differential").style.bar()

# %%
fig, ax = plt.subplots()
ax.scatter(np.log1p(diffscores["differential"]), y = diffscores["final_score"])

# %%
diffscores.groupby(["dataset", "celltype_a", "celltype_b"]).mean().sort_values("final_score")

# %%
# celltype_a, celltype_b = "CD14+ Monocytes", "CD4 memory T"
celltype_a, celltype_b = "CD4 naive T", "CD4 memory T"

# %%
for _, row in dataset_pair_combinations.iterrows():
    print(f"Name the most important transcription factor that would be more active in {row['celltype_a']} compared to {row['celltype_b']}")
    print(f"Name the most important transcription factor that would be more active in {row['celltype_b']} compared to {row['celltype_a']}")

# %% [markdown]
# ### Examples

# %%
scores_joined = (
    scores.reset_index()
    .set_index(["dataset", "celltype_a", "celltype_b", "motif"])
    .join(
        scores_peak.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"]),
        lsuffix="_chd",
        rsuffix="_peak",
    )
)
motifs_quality = motifscan.motifs.loc[motifscan.motifs["quality"].isin(["A", "B"])].index
scores_joined = scores_joined.loc[scores_joined.index.get_level_values("motif").isin(motifs_quality)]

# %%
diffscores.sort_values("diff")

# %%
# scores_oi = scores_all.loc["hspc"].loc["HSPC"].loc["MEP"].query("q_value_chd < 0.05")
# scores_oi = scores.reset_index().set_index(["dataset", "celltype_a", "celltype_b", "motif"]).loc["hspc"].loc["HSPC"].loc["MEP"]
# scores_oi = scores_joined.loc["hspc"].loc["HSPC"].loc["MEP"]
# scores_oi = scores_joined.loc["pbmc10k"].loc["CD14+ Monocytes"].loc["CD4 memory T"]
# scores_oi = scores_joined.loc["pbmc10k"].loc["FCGR3A+ Monocytes"].loc["CD14+ Monocytes"]
# scores_oi = scores_joined.loc["pbmc10k"].loc["CD4 naive T"].loc["MAIT"]
# scores_oi = scores_joined.loc["hspc"].loc["Granulocyte 1"].loc["Granulocyte 2"]
scores_oi = scores_joined.loc["pbmc10k"].loc["CD8 naive T"].loc["CD8 activated T"]
# scores_oi = scores_joined.loc["hspc"].loc["Erythroblast"].loc["Megakaryocyte"] # CG effects

# filter on concordance
scores_oi = scores_oi.loc[
    ((scores_oi["log_odds_chd"] > 0.) & (scores_oi["logfoldchanges_peak"] > 0.)) |
    ((scores_oi["log_odds_chd"] < 0.) & (scores_oi["logfoldchanges_peak"] < 0.))
]

# filter on diffexp
# scores_oi = scores_oi.loc[
#     ((np.abs(scores_oi["scores_peak"]) > 5.))
# ]
plotdata = scores_oi

motifs_to_highlight = []
motifs_to_highlight = plotdata.sort_values("log_odds_chd").tail(5).index.get_level_values("motif").tolist() + plotdata.sort_values("log_odds_chd").head(5).index.get_level_values("motif").tolist()
# motifs_to_highlight = [motifscan.select_motif("CEBPD"), motifscan.select_motif("SPI1"), "LEF1.H12CORE.0.PSM.A", motifscan.select_motif("ZEB1")]
# https://www.cell.com/trends/immunology/pdf/S1471-4906(20)30282-9.pdf for FOS/JUN
# motifs_to_highlight = [motifscan.select_motif("CEBPD"), motifscan.select_motif("JUN"), motifscan.select_motif("IRF7"), motifscan.select_motif("PO2F2"), "NFAC1.H12CORE.2.SM.B"]
# motifs_to_highlight = [motifscan.select_motif("GATA1"), motifscan.select_motif("ZEB1"), motifscan.select_motif("CEBPE")]
plotdata_oi = plotdata.loc[motifs_to_highlight]

# %%
fig, ax = plt.subplots(figsize = (2, 2))

cmap = mpl.colormaps["RdBu_r"]
norm = mpl.colors.Normalize(vmin = -1, vmax = 1)

scatter = ax.scatter(x = plotdata["log_odds_chd"], y = plotdata["log_odds_peak"], s = 5., color = "black")
# ax.scatter(x = plotdata_oi["log_odds_chd"], y = plotdata_oi["log_odds_peak"], s = 5., color = "red", zorder = 1)

texts = []
for motif, row in plotdata_oi.iterrows():
    offset = -20 if row["logfoldchanges_peak"] > 0 else 20
    texts.append(ax.annotate(
        motifscan.motifs.loc[motif]["HUMAN_gene_symbol"], 
        (row["log_odds_chd"], row["log_odds_peak"]), 
        (0, offset),
        textcoords = "offset points",
        fontsize = 8, ha = "center", va = "center", bbox = dict(facecolor = "white", edgecolor = "none", alpha = 1., pad = 0.),
        arrowprops=dict(arrowstyle = "-", lw = 2, ec = "black"),
    ))

ax.scatter(x = plotdata["log_odds_chd"], y = plotdata["log_odds_peak"], s = 5., c = cmap(norm(plotdata["logfoldchanges_peak"])))
ax.set_ylim(np.log(1/4), np.log(4))
ax.set_yticks(np.log([0.25, 0.5, 1, 2, 4]))
ax.set_yticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(np.log(1/4), np.log(4))
ax.set_xticks(np.log([0.25, 0.5, 1, 2, 4]))
ax.set_xticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

ax.axline((0, 0), (1, 1), color = "black", dashes = (2, 2))
ax.axline((0, 0), (np.log(2), 1), color = "black", dashes = (2, 2))
ax.axline((0, 0), (1, np.log(2)), color = "black", dashes = (2, 2))
ax.axvline(0, color = "black", dashes = (2, 2))
ax.axhline(0, color = "black", dashes = (2, 2))

# %%
fig, ax = plt.subplots()
ax.barh(y = plotdata_oi.index, width = np.exp(np.abs(plotdata_oi["log_odds_peak"])), color = "red")
ax.barh(y = plotdata_oi.index, width = np.exp(np.abs(plotdata_oi["log_odds_chd"])), color = "blue", alpha = 0.5)
ax.set_xlim(1)

# %%
# (plotdata_oi["n_gained_chd"] - plotdata_oi["n_gained_peak"]) / plotdata_oi["n_gained_peak"]

# %%
plotdata_oi["n_gained_chd"] = [row["contingency_chd"][1, 1] if row["odds_chd"] > 1 else row["contingency_chd"][0, 1] for _, row in plotdata_oi.iterrows()]
plotdata_oi["n_gained_peak"] = [row["contingency_peak"][1, 1] if row["odds_chd"] > 1 else row["contingency_peak"][0, 1] for _, row in plotdata_oi.iterrows()]

plotdata_oi["n_lost_chd"] = [row["contingency_chd"][1, 1] if row["odds_chd"] < 1 else row["contingency_chd"][0, 1] for _, row in plotdata_oi.iterrows()]
plotdata_oi["n_lost_peak"] = [row["contingency_peak"][1, 1] if row["odds_chd"] < 1 else row["contingency_peak"][0, 1] for _, row in plotdata_oi.iterrows()]

# %%
plotdata_oi[["odds_chd", "odds_peak", "contingency_chd", "contingency_peak", "direction_peak", "n_gained_chd", "n_gained_peak", "n_lost_chd", "n_lost_peak"]]

# %%
fig, ax = plt.subplots()
ax.barh(y = plotdata_oi.index, width = (np.abs(plotdata_oi["n_gained_peak"])), color = "red")
ax.barh(y = plotdata_oi.index, width = (np.abs(plotdata_oi["n_gained_chd"])), color = "blue", alpha = 0.5)
ax.barh(y = plotdata_oi.index, width = -(np.abs(plotdata_oi["n_lost_peak"])), color = "red")
ax.barh(y = plotdata_oi.index, width = -(np.abs(plotdata_oi["n_lost_chd"])), color = "blue", alpha = 0.5)

# %%
plotdata_oi["rel_gained_peak"] = plotdata_oi["n_gained_peak"] / plotdata_oi[["n_gained_peak"]].max(axis = 1)
plotdata_oi["rel_gained_chd"] = plotdata_oi["n_gained_chd"] / plotdata_oi[["n_gained_peak"]].max(axis = 1)
plotdata_oi["rel_lost_peak"] = plotdata_oi["n_lost_peak"] / plotdata_oi[["n_lost_peak", "n_lost_chd"]].max(axis = 1)
plotdata_oi["rel_lost_chd"] = plotdata_oi["n_lost_chd"] / plotdata_oi[["n_lost_peak", "n_lost_chd"]].max(axis = 1)

fig, ax = plt.subplots()
ax.barh(y = plotdata_oi.index, width = (np.abs(plotdata_oi["rel_gained_peak"])), color = "red")
ax.barh(y = plotdata_oi.index, width = (np.abs(plotdata_oi["rel_gained_chd"])), color = "blue", alpha = 0.5)
# ax.barh(y = plotdata_oi.index, width = -(np.abs(plotdata_oi["rel_lost_peak"])), color = "red")
# ax.barh(y = plotdata_oi.index, width = -(np.abs(plotdata_oi["rel_lost_chd"])), color = "blue", alpha = 0.5)

# %%
fig, ax = plt.subplots()
ax.barh(y = plotdata_oi.index, width = np.exp(np.abs(plotdata_oi["log_odds_peak"])), color = "red")
ax.barh(y = plotdata_oi.index, width = np.exp(np.abs(plotdata_oi["log_odds_chd"])), color = "blue", alpha = 0.5)
ax.set_xlim(1)
ax.set_xscale("log")

# %%
plotdata.sort_values("log_odds_peak", ascending = True).head(20)[["log_odds_chd", "log_odds_peak", "logfoldchanges_peak", "contingency_peak", "contingency_chd"]].style.bar()
