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
# # # cd output/data/liverphx
# # ssh -J liesbetm@cp0001.irc.ugent.be:22345 liesbetm@cn2031
# # on cn2031, proxied via 
# # /srv/data/liesbetm/Projects/u_mgu/JeanFrancois/epiPipeline/outputBowtie_filtered
# # sftp -o "ProxyJump=liesbetm@cp0001.irc.ugent.be:22345" liesbetm@cn2031:/srv/data/liesbetm/Projects/u_mgu/JeanFrancois/epiPipeline/outputBowtie_filtered

# get KcAtacRep1.bam
# get KcAtacRep2.bam
# get LsecCvPh24hAtacRep1.bam
# get LsecCvPh24hAtacRep2.bam
# get LsecPvPh24hAtacRep1.bam
# get LsecPvPh24hAtacRep2.bam
# get LsecCvShamAtacRep1.bam
# get LsecCvShamAtacRep2.bam
# get LsecPvShamAtacRep1.bam
# get LsecPvShamAtacRep2.bam

# get KcAtacRep1.bam.bai
# get KcAtacRep2.bam.bai
# get LsecCvPh24hAtacRep1.bam.bai
# get LsecCvPh24hAtacRep2.bam.bai
# get LsecPvPh24hAtacRep1.bam.bai
# get LsecPvPh24hAtacRep2.bam.bai
# get LsecCvShamAtacRep1.bam.bai
# get LsecCvShamAtacRep2.bam.bai
# get LsecPvShamAtacRep1.bam.bai
# get LsecPvShamAtacRep2.bam.bai



# # cd /srv/data/liesbetm/Projects/u_mgu/JeanFrancois/epiPipeline/outputMergePeaks/LsecCvPh24h/
# get LsecCvPh24h.fwp.filter.non_overlapping.bed

# # cd /srv/data/liesbetm/Projects/u_mgu/JeanFrancois/epiPipeline/outputMergePeaks/LsecPvPh24h
# get ./LsecPvPh24h.fwp.filter.non_overlapping.bed

# %%
# %load_ext autoreload
# %autoreload 2

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

import pysam

# %%
obs = pd.DataFrame([
    ["lsec", "central", "24h", 1, "LsecCvPh24hAtacRep1.bam"],
    ["lsec", "central", "24h", 2, "LsecCvPh24hAtacRep2.bam"],
    ["lsec", "portal", "24h", 1, "LsecPvPh24hAtacRep1.bam"],
    ["lsec", "portal", "24h", 2, "LsecPvPh24hAtacRep2.bam"],
    ["lsec", "central", "sham", 1, "LsecCvShamAtacRep1.bam"],
    ["lsec", "central", "sham", 2, "LsecCvShamAtacRep2.bam"],
    ["lsec", "portal", "sham", 1, "LsecPvShamAtacRep1.bam"],
    ["lsec", "portal", "sham", 2, "LsecPvShamAtacRep2.bam"],
    # ["kc", "-", "-", 1, "KcAtacRep1.bam"],
    # ["kc", "-", "-", 2, "KcAtacRep2.bam"],
], columns = ["celltype", "zonation", "treatment", "replicate", "path"])
obs["path"] = "/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/liverphx/" + obs["path"]

obs["alignment"] = obs["path"].apply(lambda x: pysam.AlignmentFile(x, "rb"))

# %%
import chromatinhd as chd

# %%
dataset_folder = chd.get_output() / "datasets" / "liverphx"

# %%
# import genomepy
# genomepy.install_genome("mm10", genomes_dir="/data/genome/")
fasta_file = "/data/genome/mm10/mm10.fa"
chromosomes_file = "/data/genome/mm10/mm10.fa.sizes"

regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = dataset_folder / "regions" / "all")

fragments_all = chd.data.Fragments.from_alignments(
    obs,
    regions=regions,
    alignment_column="alignment",
    path=dataset_folder / "fragments" / "all",
    # overwrite=True,
    overwrite = False,
    batch_size = 10e7,
)

# %%
phx_genes = """
Myadm
Lgals1
Actg1
Kit
Jam2
Sgk1
Egr1
Gja4
Armcx4
Dusp1
Zfp36
Junb
Klf4
Jpt1
Cyp4b1
Apoe
Orm2
Mir6236
""".strip().split("\n")

own_genes = """
Cdkn1a
Thbd
Fabp4
Armcx4
Ttr
C1qb
""".strip().split("\n")

# %%
regions_name = "100k100k"
regions_name = "10k10k"
transcripts = chd.biomart.get_canonical_transcripts(chd.biomart.Dataset.from_genome("mm10"), filter_canonical = False, 
    # symbols = [
        # "Lyve1", "Stab2",  # LSEC
        # "Gja5", "Adgrg6", "Rspo3", "Wnt9b", "Wnt2", "Dll4", "Dll1", "Hes1", "Hey1", # zonation
        # "Mecom", "Adam15", "Meis1", "Sox18", "Sox17", "Sox7",
        # *phx_genes,
        # *own_genes,
        # "Clec4f",
        # "H2-Q7", "Fcgr2b",
    # ]
)
# transcripts = transcripts.iloc[:2500]
# regions = chd.data.Regions.from_transcripts(transcripts, [-100000, 100000], path = dataset_folder / "regions" / regions_name, overwrite = True)
# transcripts = transcripts.query("chrom == 'chr19'").iloc[:10]
regions = chd.data.Regions.from_transcripts(transcripts, [-10000, 10000], path = dataset_folder / "regions" / regions_name, overwrite = True)

# %%
fragments = chd.data.fragments.FragmentsView.from_fragments(
    fragments_all,
    regions = regions,
    path = dataset_folder / "fragments" / regions_name,
    overwrite = False
)
fragments.create_regionxcell_indptr2(overwrite = False)

# %%
# minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.n_cells), np.array([fragments.var.index.get_loc(fragments.var.index[fragments.var["symbol"] == "Icam5"][0])]))
# loader = chd.loaders.fragments.Cuts(fragments, 10000)
# plt.hist(loader.load(minibatch).coordinates)

# %% [markdown]
# ## Motifscan

# %%
# pwms, motifs = chd.data.motifscan.download.get_hocomoco("motifs", "mouse")
pwms, motifs = chd.data.motifscan.download.get_hocomoco("motifs2", "human", variant = "full", overwrite = True)

motifscan_name = "hocomoco_0001"
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    # cutoff_col="cutoff_0005",
    cutoff_col="cutoff_0001",
    fasta_file=fasta_file,
    path = dataset_folder / "motifscans" / regions_name / motifscan_name,
    overwrite = False, reuse = True
)

motifscan.create_region_indptr()

plt.hist(motifscan.coordinates[(motifscan.coordinates[:] < (10000 - fragments.regions.window[0])) & (motifscan.coordinates[:] > (-10000 - fragments.regions.window[0]))])

# %% [markdown]
# ## Clustering

# %%
obs["cluster"] = obs["celltype"] + "-" + obs["zonation"] + "-" + obs["treatment"]
clustering = chd.data.Clustering.from_labels(obs["cluster"], var = obs.groupby("cluster")[["celltype", "zonation", "treatment"]].first(), path = dataset_folder / "clusterings" / "cluster", overwrite = False)

# %% [markdown]
# ## Training

# %%
fold = {
    "cells_train":np.arange(len(fragments.obs)),
    "cells_test":np.arange(len(fragments.obs)),
    "cells_validation":np.arange(len(fragments.obs)),
}

# %%
model_folder = chd.get_output() / "diff" / "liverphx" / "binary" / "split" / regions_name

# %%
import chromatinhd.models.diff.model.binary
model = chd.models.diff.model.binary.Model.create(
    fragments,
    clustering,
    fold = fold,
    # encoder = "shared",
    encoder = "split",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale = 1.5,
        bias_regularization=True,
        bias_p_scale = 1.5,
        # binwidths = (5000, 1000)
        # binwidths = (5000, 1000, 500, 100, 50)
        binwidths = (5000, 1000, 500, 100, 50, 25)
    ),
    path = model_folder / "model",
)

# %%
model.train_model(n_epochs = 200, early_stopping=False, do_validation = False)

# %%
# model.trace.plot();

# %%
model.save_state()

# %%
fragments.var["symbol"] = transcripts["symbol"]

# %%
# !du -sh {chd.get_output() / "diff" / "liverphx" / "binary" / "split" / regions_name / "scoring" / "genepositional"}

# %%
genepositional = chd.models.diff.interpret.GenePositional(path = model_folder / "scoring" / "genepositional")
genepositional.score(
    fragments,
    clustering,
    [model],
    # genes = fragments.var.reset_index().set_index("symbol").loc[["Kit", "Apoe", "Apln", "Odc1", "Dll4", "Dll1", "Jag1", "Meis1", "Efnb1", "Efnb2"]]["gene"],
    force = False,
    normalize_per_cell=2
)

# %%
prob_cutoff = 1.

# %%
import xarray as xr
probs = xr.concat([scores for _, scores in genepositional.probs.items()], dim = pd.Index(genepositional.probs.keys(), name = "gene"))
probs = probs.load()
# probs = probs.sel(cluster = ["lsec-central-sham", "lsec-portal-sham"])
# probs = probs.sel(cluster = ["lsec-central-24h", "lsec-portal-24h"])
# probs = probs.sel(cluster = ["lsec-central-24h", "lsec-central-sham"])
probs = probs.sel(cluster = ["lsec-portal-24h", "lsec-portal-sham"])
probs = probs.sel(coord = (probs.coords["coord"] > -10000) & (probs.coords["coord"] < 10000))
lr = probs - probs.mean("cluster")

probs_stacked = probs.stack({"coord-gene":["coord", "gene"]})
probs_stacked = probs_stacked.values[:, (probs_stacked.mean("cluster") > prob_cutoff).values]
probs_stacked = pd.DataFrame(probs_stacked, index = probs.coords["cluster"])
sns.heatmap(probs_stacked.T.corr())

# %%
probs_mask = (probs > 0.5).any("cluster")
lr_masked = lr.where(probs_mask).fillna(0.)

genes_oi = (lr_masked.mean("coord") **2).mean("cluster").to_pandas().sort_values(ascending = False).head(40).index

# %%
plotdata = lr_masked.sel(gene = genes_oi).mean("coord").to_pandas()
# plotdata = plotdata.loc[fragments.var.index]
plotdata.index = fragments.var.loc[plotdata.index,"symbol"]

fig, ax = plt.subplots(figsize = (2, len(plotdata) * 0.3))
sns.heatmap(plotdata, vmax = 0.2, vmin = -0.2, cmap = "RdBu_r", center = 0, cbar_kws = dict(label = "log likelihood ratio"))

# %%
symbol = "Dll4"
symbol = "Dll1"
symbol = "Jag1"
symbol = "Efnb2"
symbol = "Gnai3"
symbol = "Axin2"
symbol = "Egfl6"
symbol = "Adamtsl4"
symbol = "Acr"
symbol = "Mbd1"
symbol = "Dll4"
symbol = "Fos"
symbol = "Thbd"
symbol = "Kit"
symbol = "Dhh"
# symbol = "Rabgef1"
# symbol = "Lmna"
# symbol = "Vps72"
# symbol = "Pycard"
# symbol = "Icam4"

gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
gene_ix = fragments.var.index.get_loc(gene_id)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))
width = 10

window = fragments.regions.window
window = [-10000, 10000]
# window = [-50000, 50000]
# window = [-100000, -40000]
# window = [-80000, -70000]
# window = [-20000, 20000]
# window = [-100000, -40000]
# window = [-60000, -50000]
# window = [-50000, -40000]
# window = [-20000, 0]
# window = [0, 100000]
# locus_oi = 55700
# locus_oi = 2600
# window = [locus_oi - 5000, locus_oi + 5000]

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10")
fig.main.add_under(panel_genes)

plotdata, plotdata_mean = genepositional.get_plotdata(gene_id, clusters = clustering.var.query("celltype == 'lsec'").index)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=clustering.cluster_info, panel_height=0.4, width=width, window = window, relative_to = "lsec-portal-sham"
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
), width = 10, window = window)

fig.main.add_under(panel_peaks)

# motifs
motifs_oi = pd.DataFrame([
    [motifs.index[motifs.index.str.contains("SUH")][0], "Notch->Rbpj"],
    [motifs.index[motifs.index.str.contains("EVI1")][0], "Notch-->Mecom"],
    [motifs.index[motifs.index.str.contains("HEY1")][0], "Notch-->Hey1"],
    [motifs.index[motifs.index.str.contains("HES1")][0], "Notch-->Hes1"],
    [motifs.index[motifs.index.str.contains("TCF7")][0], "Wnt->Tcf7"],
    [motifs.index[motifs.index.str.contains("LEF1")][0], "Wnt->Lef1"],
    [motifs.index[motifs.index.str.contains("SOX7")][0], "Wnt-->Sox7"],
    [motifs.index[motifs.index.str.contains("SOX18")][0], "Wnt-->Sox18"],
    [motifs.index[motifs.index.str.contains("GATA4")][0], "Gata4"],
    [motifs.index[motifs.index.str.contains("FOS")][0], "Fos"],
    [motifs.index[motifs.index.str.contains("IRF9")][0], "Irf9"],
    [motifs.index[motifs.index.str.contains("NFKB2")][0], "NFKB2"],
], columns = ["motif", "label"]
).set_index("motif")
panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi, width = width, window = window)
fig.main.add_under(panel_motifs)

fig.plot()

# %%
for cell_ix in np.arange(fragments.n_cells):
    minibatch = chd.loaders.minibatches.Minibatch(
        np.array([cell_ix]), 
        np.array([fragments.var.index.get_loc(gene_id)])
    )
    loader = chd.loaders.fragments.Cuts(fragments, 10000)
    plt.hist(loader.load(minibatch).coordinates, bins = np.linspace(*fragments.regions.window, 50), lw = 0, alpha = 0.5)
""

# %% [markdown]
# ## Enrichment

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
