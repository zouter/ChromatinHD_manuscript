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
# cd output/data/liverphx
# cp /srv/data/liesbetm/Projects/u_mgu/JeanFrancois/epiPipeline/outputBowtie_filtered/* ./

# %%
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

import chromatinhd as chd

# %%
obs = pd.DataFrame([
    # ["lsec", "central", "24h", 1, "LsecCvPh24hAtacRep1.bam"],
    # ["lsec", "central", "24h", 2, "LsecCvPh24hAtacRep2.bam"],
    # ["lsec", "portal", "24h", 1, "LsecPvPh24hAtacRep1.bam"],
    # ["lsec", "portal", "24h", 2, "LsecPvPh24hAtacRep2.bam"],
    ["lsec", "central", "sham", 1, "LsecCvShamAtacRep1.bam"],
    ["lsec", "central", "sham", 2, "LsecCvShamAtacRep2.bam"],
    ["lsec", "portal", "sham", 1, "LsecPvShamAtacRep1.bam"],
    ["lsec", "portal", "sham", 2, "LsecPvShamAtacRep2.bam"],
    # ["kc", "-", "-", 1, "KcAtacRep1.bam"],
    # ["kc", "-", "-", 2, "KcAtacRep2.bam"],
], columns = ["celltype", "zonation", "treatment", "replicate", "path"])
obs["path"] = "/home/wouters/projects/ChromatinHD_manuscript/output/data/liverphx/" + obs["path"]

import pysam
obs["alignment"] = obs["path"].apply(lambda x: pysam.AlignmentFile(x, "rb"))

# %%
dataset_folder = chd.get_output() / "datasets" / "liverphx"

# %% [markdown]
# ## Create view for specific transcripts

# %%
# get transcripts from kia dataset
folder_data_preproc2 = chd.get_output() / "data" / "liverkia" / "liver_control_JVG28"
folder_dataset2 = chd.get_output() / "datasets" / "liverkia_lsecs"
transcriptome = chd.data.transcriptome.Transcriptome(folder_dataset2 / "transcriptome")
transcripts = pickle.load(
    (folder_data_preproc2 / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions_name = "100k100k"
regions = chd.data.regions.Regions.from_transcripts(
    transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k", overwrite = True
)

# %%
fragments = chd.data.fragments.Fragments.from_alignments(
    obs,
    regions = regions,
    alignment_column = "alignment",
    path = dataset_folder / "fragments" / regions_name,
    overwrite = True
)

# %%
fragments.create_regionxcell_indptr(
    overwrite = True,
)
fragments.var["symbol"] = transcripts["symbol"]
fragments.var = fragments.var


# fragments = chd.data.Fragments.from_alignments(
#     obs,
#     regions=regions,
#     alignment_column="alignment",
#     path = dataset_folder / "fragments" / regions_name,
#     overwrite=True,
#     batch_size = 10e7,
# )
# fragments.create_regionxcell_indptr()
# fragments.var["symbol"] = transcripts["symbol"]
# fragments.var = fragments.var

# %%
# minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.n_cells), np.array([fragments.var.index.get_loc(fragments.var.index[fragments.var["symbol"] == "Icam5"][0])]))
# loader = chd.loaders.fragments.Cuts(fragments, 10000)
# plt.hist(loader.load(minibatch).coordinates)

# %% [markdown]
# ## Motifscan

# %%
dataset_name = "liverphx"

# %%
motifscan_name = "hocomocov12" + "_" + "1e-4"
genome_folder = chd.get_output() / "genomes" / "mm10"
parent = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)

motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
    parent,
    regions,
    path=chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
)

# %% [markdown]
# ## Clustering

# %%
# obs["cluster"] = obs["celltype"] + "-" + obs["zonation"] + "-" + obs["treatment"] + "-" + obs["replicate"].astype(str)
# clustering = chd.data.Clustering.from_labels(obs["cluster"], var = obs.groupby("cluster")[["celltype", "zonation", "treatment", "replicate"]].first(), path = dataset_folder / "clusterings" / "cluster_replicate", overwrite = True)

obs["cluster"] = obs["celltype"] + "-" + obs["zonation"] + "-" + obs["treatment"]
clustering = chd.data.Clustering.from_labels(obs["cluster"], var = obs.groupby("cluster")[["celltype", "zonation", "treatment", "replicate"]].first(), path = dataset_folder / "clusterings" / "cluster", overwrite = True)

# %% [markdown]
# ## Training

# %%
fold = {
    "cells_train":np.arange(len(fragments.obs)),
    "cells_test":np.arange(len(fragments.obs)),
    "cells_validation":np.arange(len(fragments.obs)),
}

# %%
clustering = chd.data.Clustering(dataset_folder / "clusterings" / "cluster")
model_folder = chd.get_output() / "diff" / "liverphx" / "binary" / "split" / regions_name / "cluster"

# clustering = chd.data.Clustering(dataset_folder / "clusterings" / "cluster_replicate")
# model_folder = chd.get_output() / "diff" / "liverphx" / "binary" / "split" / regions_name / "cluster_replicate"

# %%
import chromatinhd.models.diff.model.binary
model = chd.models.diff.model.binary.Model.create(
    fragments,
    clustering,
    fold = fold,
    encoder = "shared",
    # encoder = "split",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale = 0.5,
        bias_regularization=True,
        bias_p_scale = 0.5,
        # binwidths = (5000, 1000)
        # binwidths = (5000, 1000, 500, 100, 50)
        binwidths = (5000, 1000, 500, 100, 50, 25)
    ),
    path = model_folder / "model",
    overwrite = True,
)

# %%
# loader = chd.loaders.clustering_fragments.ClusteringCuts(
#     fragments,
#     clustering,
#     cellxregion_batch_size = 500
# )

# model = model.to("cpu")
# for region_ix in tqdm.tqdm(range(len(regions))):
#     minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.n_cells), np.array([region_ix]))
#     data = loader.load(minibatch)
#     model.forward(data)

# %%
model.train_model(n_epochs = 40, n_regions_step = 50, early_stopping=False, do_validation = True, lr = 1e-2)

# %%
model.trace.plot();

# %%
model.save_state()

# %%
# !du -sh {model_folder / "scoring" / "genepositional"}

# %%
genepositional = chd.models.diff.interpret.RegionPositional.create(path = model_folder / "scoring" / "genepositional")
if not len(genepositional.probs) == len(fragments.var):
    genepositional.score(
        fragments = fragments,
        clustering = clustering,
        models = [model],
        # regions = fragments.var.reset_index().set_index("symbol").loc[["Kit", "Odc1", "Dll4", "Dll1", "Jag1", "Meis1", "Efnb2"]]["gene"],
        force = True,
        normalize_per_cell=1,
        device = "cpu",
    )

# %%
import sklearn.decomposition

# %%
prob_cutoff = 1.
# prob_cutoff = 0.

import xarray as xr
probs = xr.concat([scores for _, scores in genepositional.probs.items()], dim = pd.Index(genepositional.probs.keys (), name = "gene"))
probs = probs.load()
lr = probs - probs.mean("cluster")

probs_stacked = probs.stack({"coord-gene":["coord", "gene"]})
probs_stacked = probs_stacked.values[:, (probs_stacked.mean("cluster") > prob_cutoff).values]
probs_stacked = (probs_stacked - probs_stacked.mean(axis = 0)) / probs_stacked.std(axis = 0)
probs_stacked = pd.DataFrame(probs_stacked, index = probs.coords["cluster"])
sns.heatmap(probs_stacked.T.corr())

# %%
out = sklearn.decomposition.PCA(n_components = 3, whiten = True).fit_transform(probs_stacked)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1])
texts = []
for i, gene in enumerate(probs_stacked.index):
    text = ax.annotate(gene, out[i, :2])
    texts.append(text)
import adjustText
adjustText.adjust_text(texts)

# %%
plotdata = pd.DataFrame({
    "modelled":np.exp(probs).sum(["gene", "coord"]).to_pandas().sort_values(),
    # "modelled":probs.sum(["gene", "coord"]).to_pandas().sort_values(),
    "libsize":pd.Series(model.libsize, index = clustering.labels).sort_values()
})
plotdata.sort_values("libsize").style.bar()

# %%
probs_mask = (probs > 0.5).any("cluster")
lr_masked = lr.where(probs_mask).fillna(0.)

genes_oi = (lr_masked.mean("coord") **2).mean("cluster").to_pandas().sort_values(ascending = False).head(40).index

# %%
plotdata = lr_masked.sel(gene = genes_oi).mean("coord").to_pandas()
# plotdata = plotdata.loc[fragments.var.index]
plotdata.index = fragments.var.loc[plotdata.index,"symbol"]

fig, ax = plt.subplots(figsize = (3, len(plotdata) * 0.2))
sns.heatmap(plotdata, vmax = 0.2, vmin = -0.2, cmap = "RdBu_r", center = 0, cbar_kws = dict(label = "log likelihood ratio"))

# %%
# symbol = "Apln"
# symbol = "Pdgfb"
# symbol = "Thbd"
# symbol = "Kit"
# symbol = "Icam1"
# symbol = "Rspo3"
# symbol = "Dll1"
# symbol = "Mecom"

# symbol = "Odc1"
symbol = "Wnt9b"

# symbol = "Wnt2"
# symbol = "Cdh13"
# symbol = "Ltbp4"

gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
gene_ix = fragments.var.index.get_loc(gene_id)

# %%
genepositional.clustering = clustering
genepositional.regions = fragments.regions

# %%
from scipy.ndimage import convolve
def spread_true(arr, width = 5):
    kernel = np.ones(width, dtype=bool)
    result = convolve(arr, kernel, mode='constant', cval=False)
    result = result != 0
    return result

plotdata, plotdata_mean = genepositional.get_plotdata(gene_id)
selection = pd.DataFrame(
    {"chosen":(plotdata["prob"].unstack() > 0.5).any()}
)
selection["chosen"] = spread_true(selection["chosen"], width = 10)

# select all contiguous regions where chosen is true
selection["selection"] = selection["chosen"].cumsum()

regions = pd.DataFrame(
    {
        "start":selection.index[(np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == 1)[:-1]],
        "end":selection.index[(np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == -1)[1:]]
    }
)
regions["distance_to_next"] = regions["start"].shift(-1) - regions["end"]

# merge regions that are close to each other
# regions["merge"] = (regions["distance_to_next"] < 200).fillna(False)
# regions["merge"] = regions["merge"] | regions["merge"].shift(1).fillna(False)
# regions["group"] = (~regions["merge"]).cumsum()
# regions = regions.groupby("group").agg({"start":"min", "end":"max", "distance_to_next":"last"}).reset_index(drop=True)

regions["length"] = regions["end"] - regions["start"]
regions = regions[regions["length"] > 25]

# %%
import dataclasses
@dataclasses.dataclass
class Breaking():
    regions: pd.DataFrame
    gap: int
    resolution: int = 2000

breaking = Breaking(regions, 0.05)

# %%
# motifs
motifs_oi = pd.DataFrame([
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SUH")][0], "Notch->Rbpj", "Notch", mpl.cm.Blues(0.7)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("EVI1")][0], "Notch-->Mecom"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("HEY1")][0], "Notch-->Hey1", "Notch", mpl.cm.Blues(0.5)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("HES1")][0], "Notch-->Hes1", "Notch", mpl.cm.Blues(0.6)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("TCF7")][0], "Wnt->Tcf7", "Wnt", mpl.cm.Greens(0.6)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("LEF1")][0], "Wnt->Lef1", "Wnt", mpl.cm.Greens(0.5)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX7")][0], "Wnt-->Sox7", "Wnt", mpl.cm.Greens(0.7)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX18")][0], "Wnt-->Sox18", "Wnt", mpl.cm.Greens(0.9)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("FOS")][0], "Fos", "AP1", mpl.cm.Purples(0.6)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("JUND")][0], "Jund", "AP1", mpl.cm.Purples(0.5)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("BACH2")][0], "Bach2", "AP1", mpl.cm.Purples(0.7)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA3")][0], "Gata3"],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF9")][0], "Irf9"],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("NFKB2")][0], "Nfkb2"],
], columns = ["motif", "label", "group", "color"]
).set_index("motif")

# %%
dataset_folder2 = chd.get_output() / "datasets" / "liverphx_48h"

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

window = fragments.regions.window

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(region, breaking, window = window, genome = "mm10")
fig.main.add_under(panel_genes)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    gene_id, genepositional, breaking, panel_height=0.4, window = window
)
fig.main.add_under(panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(motifscan, gene_id, motifs_oi, breaking)
fig.main.add_under(panel_motifs)

fig.plot()

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

window = fragments.regions.window
# window = [-20000, 20000]
# window = [-10000, 10000]
# locus_oi = 55700
# window = [locus_oi - 5000, locus_oi + 5000]

width = (window[1] - window[0]) / 2000

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10")
fig.main.add_under(panel_genes)

panel_differential = chd.models.diff.plot.Differential.from_regionpositional(
    gene_id, genepositional, panel_height=0.4, width=width, window = window
)

fig.main.add_under(panel_differential)

import chromatinhd.data.peakcounts
panel_peaks = chd.data.peakcounts.plot.Peaks.from_bed(fragments.regions.coordinates.loc[gene_id], pd.DataFrame(
    {"path":[
        chd.get_output() / "data" / "liverphx" / "LsecCvPh24h.fwp.filter.non_overlapping.bed",
        chd.get_output() / "data" / "liverphx" / "LsecPvPh24h.fwp.filter.non_overlapping.bed"
    ], "label":[
        "LSEC CV PH 24h",
        "LSEC PV PH 24h",
    ]}
), width = width, window = window)

fig.main.add_under(panel_peaks)

panel_motifs = chd.data.motifscan.plot.GroupedMotifs(motifscan, gene_id, motifs_oi, width = width, window = window)
fig.main.add_under(panel_motifs)

fig.plot()

# %%
fig.savefig(chd.get_output() / (symbol + ".png"), dpi = 300, bbox_inches='tight')

# %%
for cell_ix in np.arange(fragments.n_cells):
    minibatch = chd.loaders.minibatches.Minibatch(
        np.array([cell_ix]), 
        np.array([fragments.var.index.get_loc(gene_id)])
    )
    loader = chd.loaders.fragments.Cuts(fragments, 10000)
    plt.hist(loader.load(minibatch).coordinates, bins = np.linspace(*fragments.regions.window, 50), lw = 0, alpha = 0.5)
""

# %%
cluster_info_oi = clustering.var
# cluster_info_oi = clustering.var.query("celltype == 'lsec'").query("zonation == 'portal'")
# cluster_info_oi = clustering.var.query("celltype == 'lsec'").query("treatment == 'sham'")

# %%
genepositional.fragments = fragments
genepositional.regions = fragments.regions
genepositional.clustering = clustering

# slices = genepositional.calculate_slices(1., clusters_oi = cluster_info_oi.index.tolist(), step = 25)
slices = genepositional.calculate_slices(0., clusters_oi = cluster_info_oi.index.tolist(), step = 25)
differential_slices = genepositional.calculate_differential_slices(slices, 1.5)

# %%
panel_differential.add_differential_slices(differential_slices)
fig.plot()
fig


# %%
def symbol_to_gene(symbols):
    return fragments.var.index[fragments.var["symbol"].isin(symbols)].tolist()


# %%
slicescores = differential_slices.get_slice_scores(regions = fragments.regions, cluster_info = cluster_info_oi)
# slicescores = slicescores.loc[slicescores["region"].isin(symbol_to_gene(["Kit", "Apln", ""]))]
# slicescores = slicescores.loc[slicescores["region"].isin(symbol_to_gene(own_genes))]
# slicescores.query("cluster == 'Plasma'").groupby("region")["length"].sum().sort_values()

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
import scipy.stats

def enrichment_foreground_vs_background(slicescores_foreground, slicescores_background, slicecounts, motifs=None):
    if motifs is None:
        motifs = slicecounts.columns

    x_foreground = slicecounts.loc[slicescores_foreground["slice"], motifs].sum(0)
    x_background = slicecounts.loc[slicescores_background["slice"], motifs].sum(0)
    n_foreground = slicescores_foreground["length"].sum()
    n_background = slicescores_background["length"].sum()

    contingencies = (
        np.stack(
            [
                n_background - x_background,
                x_background,
                n_foreground - x_foreground,
                x_foreground,
            ],
            axis=1,
        )
        .reshape(-1, 2, 2)
        .astype(np.int64)
    )

    odds = (contingencies[:, 1, 1] * contingencies[:, 0, 0] + 1) / (contingencies[:, 1, 0] * contingencies[:, 0, 1] + 1)

    p_values = np.array([scipy.stats.chi2_contingency(c).pvalue if (c > 0).all() else 1.0 for c in contingencies])
    q_values = chd.utils.fdr(p_values)

    return pd.DataFrame(
        {
            "odds": odds,
            "p_value": p_values,
            "q_value": q_values,
            "motif": motifs,
            "contingency": [c for c in contingencies],
        }
    ).set_index("motif")


def enrichment_cluster_vs_clusters(slicescores, slicecounts, slicefeatures=None, clusters=None, motifs=None, pbar=True):
    if clusters is None:
        clusters = slicescores["cluster"].cat.categories
    enrichment = []

    progress = clusters
    if pbar:
        progress = tqdm.tqdm(progress)
    for cluster in progress:
        selected_slices = slicescores["cluster"] == cluster
        slicescores_foreground = slicescores.loc[selected_slices]
        slicescores_background = slicescores.loc[~selected_slices]

        enrichment.append(
            enrichment_foreground_vs_background(
                slicescores_foreground,
                slicescores_background,
                slicecounts,
                motifs=motifs,
            )
            .assign(cluster=cluster)
            .reset_index()
        )

    enrichment = pd.concat(enrichment, axis=0).set_index(["cluster", "motif"])
    return enrichment



# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

motifscan.motifs["label"] = motifscan.motifs["MOUSE_gene_symbol"]
clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
slicecounts = motifscan.count_slices(slices)
enrichment = enrichment_cluster_vs_clusters(slicescores, slicecounts)

# %%
enrichment["log_odds"] = np.log2(enrichment["odds"])
# enrichment = enrichment.loc[enrichment.index.get_level_values("motif").isin(motifscan.motifs.index[motifscan.motifs["quality"].isin(["A", "B"])])]
# enrichment["significant"] = True
enrichment["significant"] = (enrichment["q_value"] < 0.05)# & (enrichment["odds"] > 1.5)

# %%
enrichment_oi = enrichment.loc[enrichment.groupby("motif")["significant"].any()[enrichment.index.get_level_values("motif")].values]
sns.heatmap(enrichment_oi["log_odds"].unstack().T.corr(), vmax = 0.5, vmin = -0.5, cmap = "RdBu_r", center = 0)

# %%
enrichment.query("q_value < 0.05").sort_values("odds", ascending = False).head(50).style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'FOS'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'RBPJ'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'GATA3'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'TCF7'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'HEY1'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'HES1'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %%
motifs_oi = pd.DataFrame([
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA3")][0], "Notch->Rbpj"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SUH")][0], "Notch->Rbpj"],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("EVI1")][0], "Notch-->Mecom"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("HEY1")][0], "Notch-->Hey1"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("HES1")][0], "Notch-->Hes1"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("TCF7")][0], "Wnt->Tcf7"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("LEF1")][0], "Wnt->Lef1"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX7")][0], "Wnt-->Sox7"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX18")][0], "Wnt-->Sox18"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("FOS")][0], "Fos"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("JUND")][0], "Jund"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF9")][0], "Irf9"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("RELB")][0], "RELB"],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("ONEC3")][0], ""],
], columns = ["motif", "label"]
).set_index("motif")
motifs_oi["label"] = motifscan.motifs.loc[motifs_oi.index, "MOUSE_gene_symbol"]

# %%
plotdata = enrichment["log_odds"].unstack()[motifs_oi.index]

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))
panel, ax = fig.main.add_under(polyptich.grid.Panel(np.array(plotdata.shape) * 0.2))

ax.imshow(plotdata.T, cmap = "RdBu_r", vmin = -1, vmax = 1, aspect = "auto")
ax.set_xticks(np.arange(plotdata.shape[0]))
ax.set_xticklabels(plotdata.index, rotation = 90)

ax.set_yticks(np.arange(plotdata.shape[1]))
ax.set_yticklabels(motifs_oi["label"])
fig.plot()

# %%
plotdata.sort_values("log_odds")

# %%
cluster_oi = "lsec-central-sham-1"
fig, ax = plt.subplots()
plotdata = enrichment.loc[cluster_oi].sort_values("log_odds")
ax.scatter(
    plotdata["log_odds"],
    plotdata["q_value"],
    c=plotdata["significant"].map({True: "r", False: "k"}),
    s=5,
)
ax.set_xlabel("log odds")
ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlim(np.log2(0.25), np.log2(4))

motif_ids = [
    motifscan.motifs.index[motifscan.motifs.index.str.contains("SUH")][0],
    motifscan.motifs.index[motifscan.motifs.index.str.contains("HEY1")][0],
    motifscan.motifs.index[motifscan.motifs.index.str.contains("HES1")][0],
    motifscan.motifs.index[motifscan.motifs.index.str.contains("FOS")][0],
    motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA3")][0],
]
texts = []
for motif, row in plotdata.loc[motif_ids].iterrows():
    texts.append(
        ax.annotate(
            motifscan.motifs.loc[motif]["symbol"],
            (row["log_odds"], row["q_value"]),
            ha="center",
            va="center",
            stretch="condensed",
        )
    )
    text = texts[-1]
    text.set_path_effects([mpl.patheffects.Stroke(linewidth=2, foreground='#FFFFFFDD'),mpl.patheffects.Normal()])
import adjustText
adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=1.))

# %%
enrichment.xs(motifscan.motifs.query("HUMAN_gene_symbol == 'HES1'").index[0], level = "motif").style.bar(subset = ["log_odds"], color = "#d65f5f")

# %% [markdown]
# ## Enrichment

# %%
geneposi

# %%
import chromatinhd.models.diff.differential
import chromatinhd.models.diff.enrichment

# %%
probs_diff = probs - probs.mean("cluster")
desired_x = np.arange(*window) - fragments.regions.window[0]
x = probs.coords["coord"].values - fragments.regions.window[0]
y = probs.values

# %%
y_interpolated = chd.utils.interpolate_1d(
    torch.from_numpy(desired_x), torch.from_numpy(x), torch.from_numpy(y)
).numpy()

# %%
prob_cutoff = 2.0
basepair_ranking = y_interpolated - y_interpolated.mean(1, keepdims=True)
basepair_ranking[y_interpolated < prob_cutoff] = -np.inf

# %%
regionresult = chd.models.diff.differential.DifferentialSlices.from_basepair_ranking(basepair_ranking, fragments.regions.window, np.log(1.5))
# regionresult = chd.models.diff.differential.DifferentialSlices.from_basepair_ranking(basepair_ranking, fragments.regions.window, np.log(2.0))

# %%
regions = regionresult.get_slicescores()
regions["region"] = probs.coords["gene"].values[regions["region_ix"]]
regions["region_ix"] = fragments.var.index.get_indexer(regions["region"])
regions["cluster"] = pd.Categorical(probs.coords["cluster"].values[regions["cluster_ix"]])
regions["symbol"] = fragments.var.iloc[regions["region_ix"]]["symbol"].values
# regions = regions.loc[
#     regions["symbol"].isin(["Dll4", "Rspo3", "Jag1", "Wnt2", "Wnt9b", "Angpt2", "Jag2", "Sox18", "Sox9", "Kit"])
# ]
regions = regions.query("length > 50")
# regions = regions.loc[(regions.groupby("region")["length"].sum() < 500)[regions["region"]].values]
# regions["start"] = np.clip(regions["start"] - 500, 0, fragments.regions.width)
# regions["end"] = np.clip(regions["end"] + 500, 0, fragments.regions.width)
regions.groupby("symbol")["length"].sum().sort_values()

for cluster in regions["cluster"].unique():
    regions.query("cluster == @cluster")["mid"].plot.hist(alpha = 0.5)

# %%
sns.heatmap(basepair_ranking[pd.Index(probs.coords["gene"]).get_loc(fragments.var.index[fragments.var["symbol"] == "Kit"][0])], vmax = np.log(2), vmin = -np.log(2), cmap = "RdBu_r", center = 0)

# %%
enrichmentscores = chd.models.diff.enrichment.enrich_cluster_vs_clusters(
    motifscan, fragments.regions.window, regions, "cluster", fragments.n_regions
)

# %%
enrichmentscores["qval"].plot.hist()

# %%
enrichmentscores["symbol"] = motifscan.motifs.loc[enrichmentscores.index.get_level_values("motif")]["gene_label"].values

# %%
pd.DataFrame({
    "n":enrichmentscores.xs(motifscan.motifs.query("gene_label == 'NFKB2'").index[0], level = "motif")["n_gene"][0],
    "gene":fragments.var.index,
    "symbol":fragments.var["symbol"]
}).sort_values("n", ascending = False)

# %%
regions_oi = regions.query("region == @gene_id")
for _, region in regions_oi.iterrows():
    panel_differential[0].ax.axvspan(region["start"] + window[0], region["end"] + window[0], fc = "black", alpha = 0.1)
fig.plot()
fig

# %%
enrichmentscores.xs(motifscan.motifs.query("gene_label == 'RBPJ'").index[0], level = "motif")

# %%
enrichmentscores.loc["lsec-portal-24h"].query("qval < 0.1").sort_values("odds", ascending = False).head(25)
# enrichmentscores.loc["lsec-central-24h"].query("qval < 0.1").sort_values("odds", ascending = False).head(25)
# enrichmentscores.loc["lsec-central-24h"].sort_values("odds", ascending = False).head(25)

# %%
plt.scatter(
    (plotdata.loc["lsec-central-24h"]["prob"] + plotdata.loc["lsec-central-sham"]["prob"])/2,
    # plotdata.loc["lsec-central-24h"]["prob"],
    plotdata.loc["lsec-central-sham"]["prob"] - plotdata.loc["lsec-central-24h"]["prob"],
)

# %%
plotdata_stacked = plotdata.unstack().T
plotdata_stacked["mean"] = plotdata_stacked.mean(axis = 1)
plotdata_stacked["diff"] = plotdata_stacked["lsec-portal-24h"] - plotdata_stacked["mean"]
# plotdata_stacked["diff"] = plotdata_stacked["lsec-central-24h"] - plotdata_stacked["mean"]
plotdata_stacked.loc[plotdata_stacked["mean"] > 0].sort_values("diff")

# %%
plt.hist(fragments.coordinates[:, 0], bins = 100)
""

# %%
biomart_dataset = chd.biomart.Dataset.from_genome("mm10")
