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
import crispyKC

crispyKC.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import polyptich
import tempfile

# %% [markdown]
# ## Get the dataset

# %%
dataset_name = "liverkia_lsecs"
regions_name = "100k100k"
latent = "celltype2"
transcriptome = chd.data.Transcriptome(
    chd.get_output() / "datasets" / dataset_name / "transcriptome"
)
fragments = chd.flow.Flow.from_path(
    chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name
)
clustering = chd.data.clustering.Clustering(
    chd.get_output() / "datasets" / dataset_name / "latent" / latent
)

folds = chd.data.folds.Folds(
    chd.get_output() / "datasets" / dataset_name / "folds" / "5x1"
)
fold = folds[0]

# %%
model_params = dict(
    encoder="shared",
    encoder_params=dict(
        binwidths=(5000, 1000, 500, 100),
    ),
)
train_params = dict(n_cells_step=5000, early_stopping=False, n_epochs=150, lr=1e-3)

chd.models.diff.model.binary.Models(
    chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31",
    # reset=True,
)
models = chd.models.diff.model.binary.Models.create(
    path=chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31",
    model_params=model_params,
    train_params=train_params,
    # overwrite=True,
    fragments=fragments,
    clustering=clustering,
    folds=folds,
)
models.train_models(n_workers_train=10, n_workers_validation=5)

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    models.path / "scoring" / "regionpositional",
    # reset=True,
)
regionpositional.score(
    models,
    device="cpu",
)
regionpositional

# %%
# %%
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype2",
    method="wilcoxon",
    use_raw=False,
    key_added="rank_genes_groups",
)
diffexp = transcriptome.get_diffexp()

# %% [markdown]
# ## Get the dataset

# %%
dataset_name = "liverphx"
regions_name = "100k100k"
dataset_folder = chd.get_output() / "datasets" / dataset_name
fragments_bulk = chd.flow.Flow.from_path(dataset_folder / "fragments" / regions_name)
fragments_bulk.obs["cluster"] = (
    fragments_bulk.obs["celltype"]
    + "-"
    + fragments_bulk.obs["zonation"]
    + "-"
    + fragments_bulk.obs["treatment"]
)
clustering_bulk = chd.data.Clustering.from_labels(
    fragments_bulk.obs["cluster"],
    var=fragments_bulk.obs.groupby("cluster")[
        ["celltype", "zonation", "treatment", "replicate"]
    ].first(),
    path=dataset_folder / "clusterings" / "cluster",
    # overwrite=True,
)

folds = chd.data.folds.Folds(
    chd.get_output() / "datasets" / dataset_name / "folds" / "all"
)
folds.folds = [
    {
        "cells_train": np.arange(len(fragments.obs)),
        "cells_test": np.arange(len(fragments.obs)),
        "cells_validation": np.arange(len(fragments.obs)),
    }
]

model_folder = (
    chd.get_output()
    / "diff"
    / "liverphx"
    / "binary"
    / "split"
    / regions_name
    / "cluster"
)

# clustering = chd.data.Clustering(dataset_folder / "clusterings" / "cluster_replicate")
# model_folder = chd.get_output() / "diff" / "liverphx" / "binary" / "split" / regions_name / "cluster_replicate"

# %%
import chromatinhd.models.diff.model.binary

model = chd.models.diff.model.binary.Model.create(
    fragments,
    clustering,
    fold=fold,
    encoder="shared",
    # encoder = "split",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale=0.5,
        bias_regularization=True,
        bias_p_scale=0.5,
        # binwidths = (5000, 1000)
        binwidths = (5000, 1000, 500, 100)
        # binwidths=(5000, 1000, 500, 100, 50, 25),
    ),
    path=model_folder / "model",
    # overwrite=True,
)

# %%
model.train_model(
    n_epochs=40, n_regions_step=50, early_stopping=False, do_validation=True, lr=1e-2
)

# %%
model.trace.plot()
# %%
model.save_state()

# %%
regionpositional_bulk = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    # reset=True,
)
regionpositional_bulk.score(
    [model],
    fragments=fragments_bulk,
    clustering=clustering_bulk,
    device="cpu",
)
regionpositional_bulk


# %% [markdown]
# ## Look at differential

# %%
dataset_name = "liverkia_lsecs"

# %%
motifscan_name = "hocomocov12_1e-4"
# motifscan_name = "hocomocov12_5e-4"
# motifscan_name = "hocomocov12_cutoff_5"
motifscan = chd.data.motifscan.MotifscanView(
    chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / regions_name
    / motifscan_name
)
motifscan.parent = chd.data.motifscan.Motifscan("/srv/data/wouters/projects/ChromatinHD_manuscript/output/genomes/mm10/motifscans/hocomocov12_5")
motifscan.motifs["label"] = motifscan.motifs["HUMAN_gene_symbol"]
clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype2",
    method="wilcoxon",
    use_raw=False,
    key_added="rank_genes_groups",
)
diffexp = transcriptome.get_diffexp()

# %% [markdown]
# ### Differential slices

# %%
regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

regionpositional_bulk.fragments = fragments_bulk
regionpositional_bulk.regions = fragments_bulk.regions
regionpositional_bulk.clustering = clustering_bulk

# %%
slices = regionpositional.calculate_slices(-1.0, step=25)
differential_slices = regionpositional.calculate_differential_slices(
    slices, fc_cutoff=1.3
)

# %%
slices_bulk = regionpositional_bulk.calculate_slices(-1.5, step=25)
differential_slices_bulk = regionpositional_bulk.calculate_differential_slices(
    slices_bulk, fc_cutoff=1.3
)

# slices = regionpositional_bulk.calculate_slices(-1.0, step=25)
# differential_slices = regionpositional_bulk.calculate_differential_slices(
#     slices, fc_cutoff=1.3
# )

# %%
# slices = regionpositional.calculate_slices(-2.0, step=25)
# differential_slices = regionpositional.calculate_differential_slices(
#     slices, fc_cutoff=1.6
# )

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
pd.DataFrame(
    {
        "chd": slicescores.groupby("cluster")["length"]
        .sum()
        .sort_values(ascending=False),
    }
)

# %% [markdown]
# ### Enrichment

# %%
diffexp = pd.concat(
    [
        sc.get.rank_genes_groups_df(transcriptome.adata, group=group)
        .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
        .assign(cluster=group)
        .rename(columns={"names": "gene"})
        for group in clustering.cluster_info.index
    ]
).set_index(["cluster", "gene"])
diffexp["significant"] = (diffexp["logfoldchanges"] > 0.2)
diffexp.query("symbol == 'Dll4'")

# %%
diffexp["significant"].sum()
diffexp.query("significant")

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
slicescores["significant"] = diffexp.loc[pd.MultiIndex.from_frame(slicescores[["cluster", "region"]]), "significant"].values
slicescores["start"] = slicescores["start"]
slicescores["end"] = slicescores["end"]
slicescores["slice"] = slicescores.index
# slicescores = slicescores.query("significant")
slices = slicescores.groupby(level=0)[["region_ix", "start", "end"]].first()

motifscan = chd.data.motifscan.MotifscanView(
    chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / regions_name
    / motifscan_name
)

# count motifs in slices
slicecounts = motifscan.count_slices(slices)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    slicescores, slicecounts
)
enrichment["log_odds"] = np.log(enrichment["odds"])

motifs_selected = motifscan.motifs.loc[motifscan.motifs.quality.isin(["A", "B"])]
enrichment = enrichment.query("motif in @motifs_selected.index")

# %%
genescores = pd.DataFrame(
    {"n_diff": slicescores.groupby("region_ix")["length"].sum().sort_values()}
)
genescores.index = transcriptome.var["symbol"].iloc[genescores.index]
genescores

# %%
enrichment.loc["LSEC_central"].query("q_value < 0.05").sort_values(
    "odds", ascending=False
).head(30)

# %%
enrichment.xs("SRBP1.H12CORE.0.P.B", level = "motif")

# %%
motifs_oi = motifs_selected.loc[
    [
        motifs_selected.index[motifs_selected.index.str.contains("FOXP1")][0],
        motifs_selected.index[motifs_selected.index.str.contains("IRF8")][0],
        motifs_selected.index[motifs_selected.index.str.contains("TF7L2")][0],
        *motifs_selected.index[motifs_selected.index.str.contains("MAF")],
        *motifs_selected.index[motifs_selected.index.str.contains("GATA4")],
        *motifs_selected.index[motifs_selected.index.str.contains("RFX")],
        *motifs_selected.index[motifs_selected.index.str.contains("IRF")],
        *motifs_selected.index[motifs_selected.index.str.contains("SRBP1")],
        *motifs_selected.index[motifs_selected.index.str.contains("SUH")],
    ]
]
enrichment.loc[(slice(None), motifs_oi.index), :]["log_odds"].unstack().T.sort_values(
    "LSEC_central"
).style.bar(axis=0)


# %% [markdown]
# ### Link to transcriptome
# %%
motifs_oi = (
    motifscan.motifs.sort_values("quality")
    .copy()
)
motifs_oi["gene"] = [
    transcriptome.gene_id(symbol)
    if symbol in transcriptome.var["symbol"].tolist()
    else None
    for symbol in motifs_oi["symbol"]
]
motifs_oi = motifs_oi.dropna(subset=["gene"])
len(motifs_oi)

# %%
diffexp = transcriptome.get_diffexp()
diffexp["significant_up"] = diffexp["pvals_adj"] < 0.01
diffexp["significant_down"] = diffexp["pvals_adj"] < 0.01
diffexp["significant"] = diffexp["significant_up"] | diffexp["significant_down"]
diffexp["score"] = diffexp["scores"]

diffexp["expression"] = np.log(
    (np.exp(pd.DataFrame(transcriptome.layers["normalized"][:], index=transcriptome.obs.index, columns = transcriptome.var.index))-1)
    .groupby(clustering.labels.values, observed=True)
    .mean()
).T.unstack()

# %%
enrichment["gene"] = (
    motifs_oi["gene"].reindex(enrichment.index.get_level_values("motif")).values
)

# %%
geneclustermapping = diffexp.T.unstack().to_frame("score")

# %%
diffexp_oi = diffexp.reindex(
    enrichment.reset_index().set_index(["cluster", "gene"]).index
)
diffexp_oi.index = enrichment.index
enrichment[diffexp_oi.columns] = diffexp_oi

# %%
enrichment.dropna().query("significant")

# %% [markdown]
# ### Group
# %%
merge_cutoff = 0.2
q_value_cutoff = 0.01
odds_cutoff = 1.1
min_found = 100

enrichment_grouped = []
for cluster_oi in clustering.cluster_info.index:
    slicecors = pd.DataFrame(
        np.corrcoef(slicecounts.T > 0), index=slicecounts.columns, columns=slicecounts.columns
    )
    enrichment["found"] = enrichment["contingency"].map(lambda x: x[1, 1].sum())
    enrichment_oi = (
        enrichment.loc[cluster_oi]
        .query("q_value < @q_value_cutoff")
        .query("odds > @odds_cutoff")
        .sort_values("q_value")
        .query("found > @min_found")
    )
    enrichment_oi = enrichment_oi.loc[
        (~enrichment_oi.index.get_level_values("motif").str.contains("ZNF")) &
        (~enrichment_oi.index.get_level_values("motif").str.startswith("ZN")) &
        (~enrichment_oi.index.get_level_values("motif").str.contains("KLF")) &
        (~enrichment_oi.index.get_level_values("motif").str.contains("WT"))
    ]
    motif_grouping = {}
    for motif_id in enrichment_oi.index:
        slicecors_oi = slicecors.loc[motif_id, list(motif_grouping.keys())]
        if (slicecors_oi < merge_cutoff).all():
            motif_grouping[motif_id] = [motif_id]
            enrichment_oi.loc[motif_id, "group"] = motif_id
        else:
            group = slicecors_oi.sort_values(ascending=False).index[0]
            motif_grouping[group].append(motif_id)
            enrichment_oi.loc[motif_id, "group"] = group
    enrichment_group = enrichment_oi.sort_values("odds", ascending=False).loc[list(motif_grouping.keys())]
    enrichment_group["members"] = [motif_grouping[group] for group in enrichment_group.index]
    enrichment_grouped.append(enrichment_group.assign(cluster=cluster_oi))
enrichment_grouped = pd.concat(enrichment_grouped).reset_index().set_index(["cluster", "motif"])
enrichment_grouped.sort_values("odds", ascending=False)
enrichment_grouped["color"] = sns.color_palette("tab10", len(enrichment_grouped))

# %%
enrichment_grouped.sort_values(["cluster", "odds"], ascending = False)

# %%
import eyck

eyck.modalities.transcriptome.plot_umap(
    transcriptome,
    ["Rfx3", "Rfx7", "Gata4", "Ntn4", "Vwf", "celltype2"],
).display()

# %%
# pickle.dump(enrichment, open("enrichment.pkl", "wb"))
pickle.dump(enrichment, open("enrichment2.pkl", "wb"))

# %%
enrichment_multiome = pickle.load(open("enrichment.pkl", "rb"))
enrichment = pickle.load(open("enrichment2.pkl", "rb"))

# %%
fig, ax = plt.subplots()
ax.scatter(
    enrichment["log_odds"],
    enrichment_multiome["log_odds"],
)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

for motif in enrichment.query("q_value < 0.05").sort_values("log_odds", ascending = False).index[:5]:
    ax.text(
        enrichment.loc[motif, "log_odds"],
        enrichment_multiome.loc[motif, "log_odds"],
        motif[1],
    )

# %% [markdown]
# ### Visualize a gene

# %%
plot_folder = pathlib.Path("/home/wouters/fs4/u_mgu/private/wouters/grants/2024_ERC_AdG_Martin/multiome_vs_minibulk")
plot_folder.mkdir(exist_ok=True, parents=True)

# %%
diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_portal")
    .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
    .set_index("names")
)
# pip install openpyxl
diffexp.head(50).to_excel(plot_folder / "diffexp_portal.xlsx")

diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_central")
    .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
    .set_index("names")
)
# pip install openpyxl
diffexp.head(50).to_excel(plot_folder / "diffexp_central.xlsx")

# %%
diffexp.head(10)


# %%
diffexp = transcriptome.get_diffexp()
design = pd.concat([
    pd.DataFrame({
        "symbol":["Kit", "Wnt2", "Wnt9b", "Cdh13", "Plpp1", "Plcb1", "Prickle1", "Ralgapa2", "Dlc1", "Stox2"],
        "cluster":"LSEC_central",
    }),
    pd.DataFrame({
        "symbol":[*diffexp.loc["LSEC_central"].query("logfoldchanges > 0.5").iloc[:10]["symbol"]],
        "cluster":"LSEC_central",
    }),
    pd.DataFrame({
        "symbol":["Msr1", "Dll4", "Itga9", "Ldb2", "Adam23", "Ntn4"],
        "cluster":"LSEC_portal",
    }),
    pd.DataFrame({
        "symbol":[*diffexp.loc["LSEC_portal"].query("logfoldchanges > 0.5").iloc[:10]["symbol"]],
        "cluster":"LSEC_portal",
    }),
]).drop_duplicates()

# %%
for symbol, cluster_oi in design.values:
# for symbol, cluster_oi in design.query("symbol == 'Dll4'").values:
    gene_id = transcriptome.gene_id(symbol)
    relative_to = "LSEC_portal" if cluster_oi == "LSEC_central" else "LSEC_central"
    relative_to_bulk = "lsec-portal-sham" if cluster_oi == "LSEC_central" else "lsec-central-sham"

    slicescores = differential_slices.get_slice_scores(
        regions=fragments.regions, clustering=clustering
    )
    slicescores = slicescores.loc[slicescores["region"] == gene_id]
    slicecounts_oi = slicecounts.loc[slicescores.index][
        enrichment.query("q_value < 0.01").index.get_level_values("motif").unique()
    ]
    slices["length"] = slices["end"] - slices["start"]
    slicecounts_oi = (slicecounts_oi / slicecounts.sum(0)) / (slices.loc[slicecounts_oi.index, "length"].sum() / slices["length"].sum())
    enrichment_gene = slicecounts_oi.sum(0).sort_values(ascending=False)
    enrichment_gene.head(10)

    windows = regionpositional.select_windows(gene_id, prob_cutoff=-1)
    if len(windows) == 0:
        continue
    breaking = polyptich.grid.Breaking(windows, resolution = 3000)

    transcriptome.layers["normalized"] = np.array(transcriptome.adata.X.copy().todense())

    fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

    region = fragments.regions.coordinates.loc[gene_id]
    panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
        region, breaking=breaking, genome="mm10" if "liver" in dataset_name else "GRCh38"
    )
    fig.main.add_under(panel_genes)

    cluster_info = clustering.cluster_info.loc[
            clustering.cluster_info.index != relative_to
        ].assign(label = "multiome")
    plotdata, plotdata_mean = regionpositional.get_plotdata(
        gene_id, clusters=cluster_info.index
    )

    panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
        region.name,
        regionpositional,
        # cluster_info=clustering.cluster_info,
        cluster_info=cluster_info,
        breaking=breaking,
        # order=panel_expression.order,
        relative_to=relative_to,
        ymax=3,
        label_accessibility=False,
    )
    # panel_differential.add_differential_slices(differential_slices)

    fig.main.add_under(panel_differential)

    panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
        transcriptome=transcriptome,
        clustering=clustering,
        gene=gene_id,
        order=True,
        cluster_info=cluster_info,
        layer="normalized",
        show_n_cells=False,
        show_cluster=False,
        relative_to = relative_to,
        annotate_expression=True,
    )
    fig.main.add_right(panel_expression, row=panel_differential)

    panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
        region.name,
        regionpositional_bulk,
        cluster_info=clustering_bulk.cluster_info.loc[
            clustering_bulk.cluster_info.index != relative_to_bulk
        ].assign(label = "minibulk"),
        # cluster_info=clustering_bulk.cluster_info,
        breaking=breaking,
        # order=panel_expression.order,
        relative_to=relative_to_bulk,
        ymax=3,
        label_accessibility=False,
    )
    fig.main.add_under(panel_differential)

    motifs_oi = enrichment_grouped[["group", "members", "color", "expression"]].explode("members").set_index("members")
    motifs_oi["label"] = motifscan.motifs.loc[motifs_oi.index, "HUMAN_gene_symbol"]
    motifs_oi = motifs_oi.loc[motifs_oi.index.isin(enrichment_gene.index[enrichment_gene > 1])]
    group_info = pd.DataFrame({"group":motifs_oi["group"].unique()}).set_index("group")
    # group_info["color"] = 
    # motifs_oi["color"] = motifs_oi["group"].map(group_info["color"])
    # group_info["label"] = motifs_oi.loc[group_info.index, "label"] + ""
    panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
        motifscan, gene_id, motifs_oi, breaking, group_info = group_info,
        slices_oi = differential_slices.get_slice_scores(
            regions=fragments.regions, clustering=clustering
        ).query("region == @gene_id"),
    )

    panel_differential[0][0, 0].ax.set_ylabel(
        f"{clustering.cluster_info.loc[cluster_oi, 'label']}\nvs\n{clustering.cluster_info.loc[relative_to, 'label']}",
        rotation = 0,
        ha = "right",
        va = "bottom"
    )

    fig.main.add_under(panel_motifs)

    fig.plot()
    fig.savefig(plot_folder / f"{symbol}.png", dpi=300, bbox_inches="tight")

# %%
# individual example
# prepare plotting data
# symbol = "Msr1"; cluster_oi = "LSEC_portal"
# symbol = "Ntn4"; cluster_oi = "LSEC_portal"
# symbol = "Dll4"; cluster_oi = "LSEC_portal"
# symbol = "Kit"; cluster_oi = "LSEC_central"
# symbol = "Mecom"; cluster_oi = "LSEC_portal"
# symbol = "Wnt2"; cluster_oi = "LSEC_central"
gene_id = transcriptome.gene_id(symbol)
relative_to = "LSEC_portal" if cluster_oi == "LSEC_central" else "LSEC_central"
relative_to_bulk = "lsec-portal-sham" if cluster_oi == "LSEC_central" else "lsec-central-sham"

slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
slicescores = slicescores.loc[slicescores["region"] == gene_id]
slicecounts_oi = slicecounts.loc[slicescores.index][
    enrichment.query("q_value < 0.01").index.get_level_values("motif").unique()
]
slices["length"] = slices["end"] - slices["start"]
slicecounts_oi = (slicecounts_oi / slicecounts.sum(0)) / (slices.loc[slicecounts_oi.index, "length"].sum() / slices["length"].sum())
enrichment_gene = slicecounts_oi.sum(0).sort_values(ascending=False)
enrichment_gene.head(10)

windows = regionpositional.select_windows(gene_id, prob_cutoff=-1)
breaking = polyptich.grid.Breaking(windows, resolution = 3000)

transcriptome.layers["normalized"] = np.array(transcriptome.adata.X.copy().todense())

# do the actual plotting
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
    region, breaking = breaking, genome="mm10", show_others = False, xticks = "extent",
)
fig.main.add_under(panel_genes)

cluster_info = clustering.cluster_info.loc[
        clustering.cluster_info.index != relative_to
    ].assign(label = "multiome")

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional_bulk,
    cluster_info=clustering_bulk.cluster_info.loc[
        clustering_bulk.cluster_info.index != relative_to_bulk
    ].assign(label = "minibulk"),
    # cluster_info=clustering_bulk.cluster_info,
    breaking=breaking,
    # order=panel_expression.order,
    relative_to=relative_to_bulk,
    ymax=3,
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional,
    cluster_info=cluster_info,
    breaking=breaking,
    relative_to=relative_to,
    ymax=3,
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

group_info = pd.concat([
    enrichment_grouped.loc[enrichment_grouped.index.get_level_values("motif").str.contains("GATA")],
    enrichment_grouped.loc[enrichment_grouped.index.get_level_values("motif").str.contains("MEF2")],
    enrichment_grouped.loc[enrichment_grouped.index.get_level_values("motif").str.contains("SUH")]
]).droplevel("cluster")
group_info["label"] = ["GATA4", "MEF2C", "RBPJ"]
motifs_oi = group_info.explode("members").set_index("members")
panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
    motifscan, gene_id, motifs_oi, breaking, group_info = group_info,
    # slices_oi = differential_slices.get_slice_scores(
    slices_oi = differential_slices_bulk.get_slice_scores(
        regions=fragments.regions, clustering=clustering
    ).query("region == @gene_id"),
)

panel_differential[0][0, 0].ax.set_ylabel(
    f"{clustering.cluster_info.loc[cluster_oi, 'label']}\nvs\n{clustering.cluster_info.loc[relative_to, 'label']}",
    rotation = 0,
    ha = "right",
    va = "bottom"
)

fig.main.add_under(panel_motifs)

fig.plot()
# fig.savefig(plot_folder / f"{symbol}.png", dpi=300, bbox_inches="tight")
fig.savefig(plot_folder / f"{symbol}.pdf", dpi=300, bbox_inches="tight")
fig.plot()

# %% [markdown]
# ## Sigmoid

# %%
symbol = "Ntn4"; cluster_oi = "LSEC_portal"
regions = pd.DataFrame({"start":[37850], "end":[39050]})
y_func = lambda x: 1 / (1 + np.exp(-10 * (x - 0.8)))

symbol = "Kit"; cluster_oi = "LSEC_central"
regions = pd.DataFrame({"start":[-84350], "end":[-83150]})
y_func = lambda x: 1 / (1 + np.exp(-5 * (0.5- x)))

breaking = polyptich.grid.Breaking(regions, resolution = 3000)

gene_id = transcriptome.gene_id(symbol)

# %%
plotdata, plotdata_mean = regionpositional.get_plotdata(
    gene_id, relative_to = relative_to
)

# %%

# %%
xs = np.linspace(0, 1, 100)
# sigmoid
ys = 1 / (1 + np.exp(-10 * (xs - 0.8)))

plt.scatter(xs, ys)

# %%
unstacked = plotdata["prob"].unstack()
diff = unstacked.loc["LSEC_portal"] - unstacked.loc["LSEC_central"]
ref = unstacked.loc["LSEC_central"]

xs = np.linspace(0, 1, 5)
cluster_info = pd.DataFrame({
    "label": ["0", "1", "2", "3", "4"],
}).set_index("label", drop = False)
cluster_info.index.name = "cluster"
cluster_info["label"] = ""

final = ref.values[:, None] + diff.values[:, None] * y_func(xs)[None, :] * 1.5
plotdata = pd.DataFrame({"prob":pd.DataFrame(final, index = diff.index, columns = cluster_info.index).T.stack()})
plotdata_mean = plotdata.loc["0"]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
    region, breaking = breaking, genome="mm10", show_others = False, xticks = "extent",
)
fig.main.add_under(panel_genes)

panel_differential = chd.models.diff.plot.DifferentialBroken(
    plotdata,
    plotdata_mean,
    cluster_info=cluster_info,
    breaking=breaking,
    ymax=2,
    label_accessibility=False,
    panel_height = 0.4
)

fig.main.add_under(panel_differential)


group_info = pd.concat([
    enrichment_grouped.loc[enrichment_grouped.index.get_level_values("motif").str.contains("GATA")],
    enrichment_grouped.loc[enrichment_grouped.index.get_level_values("motif").str.contains("MEF2")],
    enrichment_grouped.loc[enrichment_grouped.index.get_level_values("motif").str.contains("SUH")]
]).droplevel("cluster")
group_info["label"] = ["GATA4", "MEF2C", "RBPJ"]
motifs_oi = group_info.explode("members").set_index("members")
panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
    motifscan, gene_id, motifs_oi, breaking, group_info = group_info,
    # slices_oi = differential_slices.get_slice_scores(
    slices_oi = differential_slices_bulk.get_slice_scores(
        regions=fragments.regions, clustering=clustering
    ).query("region == @gene_id"),
)
fig.main.add_under(panel_motifs)

if symbol == "Ntn4":
    fig.savefig(plot_folder / "Ntn4_sigmoid.pdf")
elif symbol == "Kit":
    fig.savefig(plot_folder / "Kit_sigmoid.pdf")
fig.display()

# %%
eyck.modalities.transcriptome.plot_umap(
    transcriptome,
    color = [gene_id, "celltype2"],
).display()

# %% [markdown]
# ## FOcus on TF

# %%
slicescores_a = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
).query("cluster == 'LSEC_central'")
slicescores_b = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
).query("cluster == 'LSEC_portal'")

# %%
slicescores_a["motifs"] = slicecounts["MAF.H12CORE.1.PSM.A"]
slicescores_b["motifs"] = slicecounts["MAF.H12CORE.1.PSM.A"]

# %%
genescores = pd.DataFrame(
    {
        "central": slicescores_a.groupby("region")["motifs"].sum(),
        "central_n": slicescores_a.groupby("region")["motifs"].size(),
    }
)
slicescores_a.groupby("region")["motifs"].sum().sort_values(ascending=False).head(10)

# %% [markdown]
# ## Portal

# %%
# MAF removed => more central
# MAF => important for portal

# %%
motifs_oi = motifscan.motifs.loc[
    [
        *motifscan.motifs.index[motifscan.motifs.index.str.contains("MAF")],
        # motifscan.motifs.index[motifscan.motifs.index.str.contains("MAF")][0],
    ]
]

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
slicescores.index = (
    slicescores["region_ix"].astype(str)
    + ":"
    + slicescores["start"].astype(str)
    + "-"
    + slicescores["end"].astype(str)
)
slicescores = slicescores.loc[(slicecounts[motifs_oi.index] > 0).any(axis=1)]

# %%
# critical portal genes
gene_ids = transcriptome.gene_id(["Wnt2", "Wnt9b"])
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype2",
    method="wilcoxon",
    use_raw=False,
    key_added="rank_genes_groups",
)
gene_ids = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_portal")
    .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
    .query("logfoldchanges > 0.5")
    .head(100)["names"]
)

# %%
slicescores_oi = slicescores  # .loc[slicescores["region"].isin(gene_ids)]
slicescores_oi.groupby("cluster").size()

# %%
diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_portal")
    .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
    .set_index("names")
)
diffexp.loc[slicescores_oi["region"], "scores"].mean()

# %% [markdown]
# ## Central

# %%
# GATA => important for central

# %%
motifs_oi = motifscan.motifs.loc[
    [
        *motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA")],
    ]
]

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
slicescores.index = (
    slicescores["region_ix"].astype(str)
    + ":"
    + slicescores["start"].astype(str)
    + "-"
    + slicescores["end"].astype(str)
)

slicescores = slicescores.loc[(slicecounts[motifs_oi.index] > 0).any(axis=1)]

# %%
# critical portal genes
gene_ids = transcriptome.gene_id(["Dll4"])
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype2",
    method="wilcoxon",
    use_raw=False,
    key_added="rank_genes_groups",
)
gene_ids = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_central")
    .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
    .query("logfoldchanges > 0.5")
    .head(100)["names"]
)

# %%
slicescores_oi = slicescores  # .loc[slicescores["region"].isin(gene_ids)]
slicescores_oi.groupby("cluster").size()

# GATA => important for portal
# %%
# %%
diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_central")
    .assign(symbol=lambda x: transcriptome.var.loc[x.names]["symbol"].values)
    .set_index("names")
)
diffexp.loc[slicescores_oi["region"], "scores"].median()
# %%
