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
# sftp -o "ProxyJump=liesbetm@cp0001.irc.ugent.be:22345" liesbetm@cn2031:/srv/data/liesbetm/Projects/u_mgu/Wouter/ChrisGlass_atac/
# get -R outputBowtie

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

chd.set_default_device("cuda:1")

# %%
import chromatinhd as chd
data_folder = chd.get_output() /"data"/"glas_2019"
dataset_folder = chd.get_output() / "datasets" / "glas_2019"

# %% [markdown]
# ## Download

# %%
# ! wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE128nnn/GSE128662/suppl/GSE128662_RAW.tar -O {data_folder}/GSE128662_RAW.tar

# %%
(data_folder / "peaks").mkdir(exist_ok=True)
# ! tar -xvf {data_folder}/GSE128662_RAW.tar -C {data_folder}/peaks

# %%
# convert peaks
import glob
for file in glob.glob(str(data_folder / "peaks" / "GSM*.peak.txt.gz")):
    peaks = pd.read_table(file, comment = "#", names ="PeakID	chr	start	end	strand	Normalized Tag Count	focus ratio	findPeaks Score	Fold Change vs Local	p-value vs Local	Clonal Fold Change".split("\t"))[["chr", "start", "end"]]
    peaks.to_csv(pathlib.Path(re.sub("GSM[0-9]*_", "", file.split(".")[0])).with_suffix(".bed"), sep = "\t", index = False, header = False)

# %%
# !wget https://ars.els-cdn.com/content/image/1-s2.0-S1074761319303735-mmc5.xlsx -O {data_folder}/mmc5.xlsx

# %% [markdown]
# ## Load obs

# %%
obs = pd.read_table('/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/glas_2019/samples.tsv', sep = " ", engine="python").fillna(np.nan)

# %%
# add condition to obs
conditions = []
for _, row in obs.iterrows():
    condition = row["tissue"] + "_" + row["celltype"]
    if not pd.isnull(row["timepoint"]):
        condition += "_" + str(row["timepoint"])
    if not pd.isnull(row["treatment"]):
        condition += "_" + str(row["treatment"])
    if not pd.isnull(row["genetic"]):
        condition += "_" + str(row["genetic"])
    if not pd.isnull(row["sort"]):
        condition += "_" + str(row["sort"])
    conditions.append(condition)
obs["condition"] = conditions

obs["path"] = obs["file"].apply(lambda x: "/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/glas_2019/outputBowtie/" + x)

# %%
# import genomepy
# genomepy.install_genome("mm10", genomes_dir="/data/genome/")

# process all fragments
if not (dataset_folder / "fragments" / "all").exists():
    fasta_file = "/data/genome/mm10/mm10.fa"
    chromosomes_file = "/data/genome/mm10/mm10.fa.sizes"

    regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = dataset_folder / "regions" / "all")

    fragments_all = chd.data.Fragments.from_alignments(
        obs,
        regions=regions,
        path=dataset_folder / "fragments" / "all",
        overwrite = True,
        batch_size = 10e7,
        paired = False,
        remove_duplicates = True,
    )
    fragments_all.create_regionxcell_indptr()

# %% [markdown]
# ## Create dataset

# %%
fragments_all = chd.data.Fragments.from_path(dataset_folder / "fragments" / "all")

# %%
mmc5 = pd.read_excel(data_folder / "mmc5.xlsx", sheet_name = "Table S4")

# %%
mmc5["symbol"] = mmc5["Annotation/Divergence"].str.split("|").str[0]

# %%
mmc5["significant"] = (np.abs(mmc5["smadCTRt4.vs.smadKOt4n.log2FoldChange"]) > 1) & (mmc5["smadCTRt4.vs.smadKOt4n.padj"] < 0.05)
symbols = list(set(mmc5.sort_values("smadCTRt4.vs.smadKOt4n.log2FoldChange").query("significant")["symbol"].tolist()))
print(len(symbols))

# %%
regions_name = "100k100k"
transcripts = chd.biomart.get_canonical_transcripts(chd.biomart.Dataset.from_genome("mm10"), filter_canonical = False, 
    symbols = symbols
)
regions = chd.data.Regions.from_transcripts(transcripts, [-100000, 100000], path = dataset_folder / "regions" / regions_name, overwrite = True)

# %%
fragments = chd.data.fragments.FragmentsView.from_fragments(
    fragments_all,
    regions = regions,
    path = dataset_folder / "fragments" / regions_name,
    overwrite = True
)
fragments.create_regionxcell_indptr2(overwrite = False)

# %% [markdown]
# ## Motifscan

# %%
# import genomepy
# genomepy.install_genome("mm10", genomes_dir="/data/genome/")
fasta_file = "/data/genome/mm10/mm10.fa"

# %%
pwms, motifs = chd.data.motifscan.download.get_hocomoco("motifs", "mouse", variant = "full", overwrite = True)
# pwms, motifs = chd.data.motifscan.download.get_hocomoco("motifs2", "human", variant = "full", overwrite = True)

motifscan_name = "hocomoco_0001"
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    # cutoff_col="cutoff_0005",
    cutoff_col="cutoff_0001",
    fasta_file=fasta_file,
    path = dataset_folder / "motifscans" / regions_name / motifscan_name,
    overwrite = True
)

motifscan.create_region_indptr()

plt.hist(motifscan.coordinates[(motifscan.coordinates[:] < (10000 - fragments.regions.window[0])) & (motifscan.coordinates[:] > (-10000 - fragments.regions.window[0]))])

# %% [markdown]
# ## Clustering

# %%
clustering = chd.data.Clustering.from_labels(obs["condition"], path = dataset_folder / "clusterings" / "cluster", overwrite = False)
clustering = chd.data.Clustering.from_labels(obs["file"], path = dataset_folder / "clusterings" / "file", overwrite = False)

# %% [markdown]
# ## Training

# %%
fold = {
    "cells_train":np.arange(len(fragments.obs)),
    "cells_test":np.arange(len(fragments.obs)),
    "cells_validation":np.arange(len(fragments.obs)),
}

# %%
model_folder = chd.get_output() / "diff" / "glas_2019" / "binary" / "split" / regions_name

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
    overwrite = True,
)

# %%
model.train_model(n_epochs = 200, early_stopping=False, do_validation = False)

# %%
genepositional = chd.models.diff.interpret.GenePositional(
    path = model_folder / "scoring" / "genepositional",
    # reset = True
)
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
# probs = probs.sel(cluster = ["lsec-portal-24h", "lsec-portal-sham"])
# probs = probs.sel(coord = (probs.coords["coord"] > -10000) & (probs.coords["coord"] < 10000))
lr = probs - probs.mean("cluster")

probs_stacked = probs.stack({"coord-gene":["coord", "gene"]})
# probs_stacked = probs_stacked.values[:, (probs_stacked.mean("cluster") > prob_cutoff).values]
probs_stacked = pd.DataFrame(probs_stacked, index = probs.coords["cluster"])
sns.heatmap(probs_stacked.T.corr())

# %%
fragments.var["symbol"] = transcripts["symbol"]

# %%
symbol = "Nr1h3"
# symbol = "Tfec"
symbol = "Clec4f"
# symbol = "Spic"
symbol = "Cbr2"

gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
gene_ix = fragments.var.index.get_loc(gene_id)

# %%
# cluster_info_oi = obs.query("condition in ['liver_mo-kc_24h_DT', 'liver_mo-kc_48h_DT', 'liver_kc', 'blood_monocyte']").set_index("file").sort_values("condition")
cluster_info_oi = obs.query("condition in ['liver_kc_Smad4 flox/flox Clecf4-cre', 'liver_kc_Smad4 flox/flox']").set_index("file").sort_values("condition")

# cluster_info_oi["label"] = cluster_info_oi["condition"]
# cluster_info_oi["label"] = cluster_info_oi.index
cluster_info_oi["label"] = pd.Series({"liver_kc_Smad4 flox/flox":"Control", "liver_kc_Smad4 flox/flox Clecf4-cre":"Smad4 -/-"})[cluster_info_oi["condition"]].values

# %%
mmc5.sort_values("smadCTRt4.vs.smadKOt4n.log2FoldChange", ascending = True).query("significant").head(20)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

window = fragments.regions.window

symbol = "Apoc1"
gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
window = [-5000, -2000]

# symbol = "Cx3cr1"
# gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
# window = [-5000, 0]

symbol = "Lepr"
gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
window = [-100000, 100000]

# symbol = "Slco2b1"
# gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
# window = [1000, 6000]

width = (window[1] - window[0])/1000/20*10
# window = [-20000, 1000]
# window = [-10000, 20000]
# window = [-40000, -20000]
# window = [50000, 70000]
# window = [10000, 50000]

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10")
panel_genes.ax.set_xlabel("Distance to "+ symbol + " TSS")
fig.main.add_under(panel_genes)

plotdata, plotdata_mean = genepositional.get_plotdata(gene_id)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=cluster_info_oi, panel_height=0.4, width=width, window = window, relative_to = cluster_info_oi.index[:2]
)
panel_differential[0].ax.set_ylabel("")

fig.main.add_under(panel_differential)

# peaks
import chromatinhd.data.peakcounts
panel_peaks = chd.data.peakcounts.plot.Peaks.from_bed(fragments.regions.coordinates.loc[gene_id], pd.DataFrame(
    {"path":[
        data_folder / "peaks" / "Smad4KO_KC_ATAC_NoTx_366.bed",
        # data_folder / "peaks" / "Smad4KO_KC_ATAC_NoTx_382.bed",
    ], "label":[
        "Peaks",
        # "Smad4KO_KC_ATAC_NoTx_382.bed",
    ]}
), width = width, window = window)

fig.main.add_under(panel_peaks)
panel_peaks.ax.set_ylabel("")

# motifs
motifs_oi = pd.DataFrame([
    # [motifs.index[motifs.index.str.contains("SUH")][0], "Notch->Rbpj"],
    # [motifs.index[motifs.index.str.contains("EVI1")][0], "Notch-->Mecom"],
    # [motifs.index[motifs.index.str.contains("HEY1")][0], "Notch-->Hey1"],
    # [motifs.index[motifs.index.str.contains("HES1")][0], "Notch-->Hes1"],
    [motifs.index[motifs.index.str.contains("SMAD4")][0], "Smad4 motifs"],
    [motifs.index[motifs.index.str.contains("NR1H3")][0], "LXRa motifs"],
], columns = ["motif", "label"]
).set_index("motif")
panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi, width = width, window = window)
fig.main.add_under(panel_motifs)

fig.plot()

# %% [markdown]
# ## Enrichment

# %%
import chromatinhd.models.diff.differential
import chromatinhd.models.diff.enrichment

# %%
import xarray as xr
probs = xr.concat([scores for _, scores in genepositional.probs.items()], dim = pd.Index(genepositional.probs.keys(), name = "gene"))
probs = probs.load()

# %%
probs_oi = probs.sel(cluster = cluster_info_oi.index)
probs_diff = probs_oi - probs_oi.mean("cluster")
desired_x = np.arange(*fragments.regions.window) - fragments.regions.window[0]
x = probs_oi.coords["coord"].values - fragments.regions.window[0]
y = probs_oi.values

# %%
y_interpolated = chd.utils.interpolate_1d(
    torch.from_numpy(desired_x), torch.from_numpy(x), torch.from_numpy(y)
).numpy()

# %%
prob_cutoff = 1.
basepair_ranking = y_interpolated - y_interpolated.mean(1, keepdims=True)
basepair_ranking[y_interpolated < prob_cutoff] = -np.inf

# %%
symbol = "Lepr"
gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
gene_ix = fragments.var.index.get_loc(gene_id)
sns.heatmap(basepair_ranking[gene_ix][:, 120000:150000])

# %%
regionresult = chd.models.diff.differential.DifferentialSlices.from_basepair_ranking(basepair_ranking, fragments.regions.window, np.log(1.5))
# regionresult = chd.models.diff.differential.DifferentialSlices.from_basepair_ranking(basepair_ranking, fragments.regions.window, np.log(2.0))

# %%
regions = regionresult.get_slicescores()
regions["region"] = probs_oi.coords["gene"].values[regions["region_ix"]]
regions["region_ix"] = fragments.var.index.get_indexer(regions["region"])
regions["cluster"] = pd.Categorical(probs_oi.coords["cluster"].values[regions["cluster_ix"]])
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
motifscan.create_region_indptr()
motifscan.create_indptr()

# %%
enrichmentscores = chd.models.diff.enrichment.enrich_cluster_vs_clusters(
    motifscan, fragments.regions.window, regions, "cluster", fragments.n_regions
)

# %%
enrichmentscores["qval"].plot.hist()

# %%
enrichmentscores["symbol"] = motifscan.motifs.loc[enrichmentscores.index.get_level_values("motif")]["gene_label"].values

# %%
motifscan.motifs.query("gene_label == 'NR1H3'")

# %%
pd.DataFrame({
    "n":enrichmentscores.xs(motifscan.motifs.query("gene_label == 'Nr1h3'").index[0], level = "motif")["n_gene"][0],
    "gene":fragments.var.index,
    "symbol":fragments.var["symbol"]
}).sort_values("n", ascending = False)

# %%
motif_ix = motifscan.motifs.index.get_loc(motifscan.motifs.query("gene_label == 'Smad4'").index[0])

# %%
basepair_ranking = y_interpolated - y_interpolated.mean(1, keepdims=True)

# %%
onehots = chd.data.motifscan.motifscan.create_region_onehots(fragments.regions, fasta_file)
onehot_promoters = torch.from_numpy(np.stack(list(onehots.values()))).permute(1, 0, 2).float()

# %%
enrichment = chd.models.diff.enrichment.enrich_cluster_vs_background(motifscan, fragments.regions.window, regions.query("cluster == 'Smad4KO_KC_ATAC_NoTx_382.bam'"), "cluster", fragments.var.shape[0], onehot_promoters)

# %%
enrichment.sort_values("odds")

# %%
sites_oi = motifscan.indices[:] == motif_ix
coordinates = motifscan.coordinates[sites_oi]
region_ixs = motifscan.positions[sites_oi] // (fragments.regions.width-1)
found = scipy.sparse.coo_matrix((np.ones(len(region_ixs), dtype = bool), (region_ixs, coordinates)), shape = (fragments.n_regions, fragments.regions.width)).todense()

design = chd.utils.crossing(
    pd.DataFrame({"prob_cutoff":np.linspace(-0.5, 1.0, 10)}),
    pd.DataFrame({"probdiff_cutoff":np.linspace(-0.5, 1.0, 10)}),
)

scores = []
for _, row in tqdm.tqdm(design.iterrows(), total = len(design)):
    ranked = (basepair_ranking[:, 0] > row["prob_cutoff"]) & (y_interpolated[:, 0] > row["probdiff_cutoff"])
    contingency = np.array([
        [np.sum(~found & ~ranked), np.sum(found & ~ranked)],
        [np.sum(~found & ranked), np.sum(found & ranked)],
    ])
    test = scipy.stats.fisher_exact(contingency)

    scores.append({"pval":test.pvalue, "odds":test.statistic})

# %%
design["pval"] = [x["pval"] for x in scores]
design["odds"] = [x["odds"] for x in scores]

# %%
sns.heatmap(np.log(design.set_index(["prob_cutoff", "probdiff_cutoff"])["pval"].unstack()))

# %%
sns.heatmap(np.log2(design.set_index(["prob_cutoff", "probdiff_cutoff"])["odds"].unstack()), cmap = "RdBu_r", center = 0)

# %%
enrichmentscores.query("symbol == 'SMAD4'")

# %%
enrichmentscores.sort_values("qval").head(20)
