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
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %% [markdown]
# ## Data

# %% [markdown]
# ### Dataset

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "brain"
# dataset_name = "alzheimer"
# dataset_name = "GSE198467_H3K27ac"
# dataset_name = "GSE198467_single_modality_H3K27me3"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
# transcriptome = None
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
fragments.create_cut_data()

# %%
fragments.obs["lib"] = torch.bincount(fragments.cut_local_cell_ix, minlength = fragments.n_cells).numpy()

# %% [markdown]
# ### Latent space

# %%
# loading
# latent_name = "leiden_1"
latent_name = "leiden_0.1"
# latent_name = "celltype"
# latent_name = "overexpression"
folder_data_preproc = folder_data / dataset_name
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
latent_torch = torch.from_numpy(latent.values).to(torch.float)

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
cluster_info["color"] = sns.color_palette("husl", latent.shape[1])

fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])
if transcriptome is not None:
    transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %%
cluster_info["lib"] = fragments.obs.groupby("cluster")["lib"].sum().values

# %% [markdown]
# ### Prediction

# %%
method_name = 'v9_128-64-32'
class Prediction(chd.flow.Flow):
    pass
prediction = Prediction(chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / method_name)

# %%
model = pickle.load((prediction.path / "model_0.pkl").open("rb"))
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

# %%
prob_cutoff = np.log(1.)
# prob_cutoff = -np.inf

# %%
x = torch.from_numpy(probs)

# %%
y = np.cumsum(probs, -1)

# %%
n = 10


# %%
def moving_average(a, n=3) :
    dim = -1
    a_padded = torch.nn.functional.pad(a, (n//2, n//2), mode = "replicate")
    ret = torch.cumsum(a_padded, axis = dim)
    ret[..., :-n] = ret[..., n:] - ret[..., :-n]
    
    return ret[..., :a.shape[dim]] / n


# %%
x = torch.from_numpy(np.exp(probs))
probs_smoothened = moving_average(x, 100).numpy()

# %%
plt.plot(x[0][0])
plt.plot(probs_smoothened[0][0])

# %%
# mask = probs > prob_cutoff
mask = (probs > prob_cutoff) & ((np.exp(probs) / probs_smoothened) > 1.2)

# %%
probs_diff = probs - probs.mean(1, keepdims = True)
probs_diff_masked = probs_diff.copy()
probs_diff_masked[~mask] = 0.

# %%
pd.DataFrame({
    "prob_diff":(probs_diff_masked**2).mean(-1).mean(-1),
    "prob_diff_up":(probs_diff_masked > np.log(2)).mean(-1).sum(-1),
    "nonmasked":(mask).mean(-1).mean(-1),
    "gene":fragments.var.index,
    "symbol":promoters["symbol"],
    "ix":np.arange(len(promoters))
}).sort_values("prob_diff_up", ascending = False).head(20)

# %%
sns.histplot((probs_diff_masked > np.log(2)).mean(-1).sum(-1))

# %% [markdown]
# ### Genes

# %%
genes = pd.read_csv(folder_data_preproc / "genome/genes.csv", index_col = 0)

# %% [markdown]
# ### Motifscan

# %%
# motifscan_name = "cutoff_0001"
# motifscan_name = "cutoff_001"
# motifscan_name = "onek1k_0.2"
motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_cns"
# motifscan_name = "gtex_immune"

# %%
motifscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
motifscan = chd.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
motifscan.n_motifs = len(motifs)
motifs["ix"] = np.arange(motifs.shape[0])

# %%
# count number of motifs per gene
promoters["n"] = np.diff(motifscan.indptr).reshape((len(promoters), (window[1] - window[0]))).sum(1)

# %%
# distribution
plt.plot(np.diff(motifscan.indptr).reshape((len(promoters), (window[1] - window[0]))).sum(0))

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### Peakcounts

# %%
import chromatinhd.peakcounts

# %%
peaks_name = "macs2_improved"
peakcounts = chd.peakcounts.FullPeak(folder = chd.get_output() / "peakcounts" / dataset_name / peaks_name)

# %%
peak_diffexp = pd.read_csv(peakcounts.path / ("diffexp_" + latent_name + ".csv"), index_col = 0)
peak_lfc = peak_diffexp.set_index(["cluster", "peak"])["logFC"].unstack()
peak_lfc = peak_lfc.loc[cluster_info["dimension"]]
peak_lfc = peak_lfc.set_index(cluster_info.index)

# %%
peak_cluster_counts = pd.DataFrame(peakcounts.counts.todense(), index = cluster_info.reset_index().set_index("dimension").reindex(fragments.obs["cluster"])["cluster"], columns = peakcounts.var.index).groupby(level = 0).sum()

# %%
# peak_cluster_probs = peak_cluster_counts / cluster_info["lib"].values[:, None] * fragments.obs["lib"].mean()

# %%
var = peakcounts.var
var["size"] = (peakcounts.peaks.groupby("peak").first()["end"] - peakcounts.peaks.groupby("peak").first()["start"])

# %%
peak_probs = pd.Series(np.array(peakcounts.counts.sum(0))[0] / fragments.obs["lib"].sum() * fragments.obs["lib"].mean() * var["size"] / (window[1] - window[0]), index = peakcounts.var.index)

# %%
peak_cluster_probs = peak_probs.values[None, :] * 2**peak_lfc[peak_probs.index]

# %%
peak_cluster_prob_diffs = np.log(peak_cluster_probs) - np.log(peak_cluster_probs).mean(0)

# %% [markdown]
# ## Choose gene

# %%
# peaks_name = "cellranger"
# scores_dir = (prediction.path / "scoring" / peaks_name / motifscan_name)
# genemotifscores_all = pd.read_pickle(scores_dir / "genemotifscores_all.pkl")
# # genemotifscores_all.xs(motifs_oi.index[1], level = "motif").sort_values("n_region", ascending = False).assign(symbol = lambda x:transcriptome.symbol(x.index).values)
# genemotifscores_all.sort_values("n_region", ascending = False).assign(symbol = lambda x:transcriptome.symbol(x.index.get_level_values("gene")).values)

# %%
# gene_oi = 0; symbol = transcriptome.var.iloc[gene_oi]["symbol"];gene_id = transcriptome.var.index[gene_oi]

# symbol = "ZFP36"
# symbol = "IL3RA"
# symbol = "CCL4"
# symbol = "AAK1"
# symbol = "RGS1"
# symbol = "POU2AF1"
# symbol = "CD74"
# symbol = "KIF21B"
# symbol = "TP53"
# symbol = "CD19"
# symbol = "MS4A1"
# symbol = "FOSB"
# symbol = "JCHAIN"
# symbol = "IL1B"
# symbol = "FOSB"
# symbol = "HLA-DPA1"
# symbol = "HLA-DQA1"
# symbol = "SCO2"
# symbol = "TYMP"
# symbol = "SCO2"
# symbol = "ZNF263"
# symbol = "FOS"
# symbol = "IFNLR1"
# symbol = "CTLA4"
# symbol = "JAK2"
# symbol = "IRF7"
# symbol = "CCL4"
# symbol = "CD70"
# symbol = "TCF7"
# symbol = "CD8A"
# symbol = "FOS"
# symbol = "CD40" # Interesting
# symbol = "HLA-DQA1"
# symbol = "CD244" # Woohoo
# symbol = "POU2AF1" # Yays
symbol = "LRRC25"
# symbol = "TCF3"
# symbol = "SP140"
# symbol = "TCF3"


# symbol = "TNFRSF13C" # Certainly interesting, the promoter is opening up everywhere, no wonder with PU1 binding sites littered all over the place

# symbol = "Foxc1"
# symbol = "Neurod1"

# symbol = "GATA3"
# symbol = "S100A10"
# symbol = "SEMA3A"

# symbol = "Neurod2"
# symbol = "Ackr3"
# symbol = "Fzd9"
# symbol = "Dlx6"

# symbol = "Tgfb2"

# symbol = "CEMIP"
# symbol = "APOE"
# symbol = "CACNA1A"
# symbol = "NEGR1"

gene_id = transcriptome.gene_id(symbol)
gene_oi = transcriptome.gene_ix(symbol)

# promoters["ix"] = np.arange(len(promoters))
# gene_id = promoters.query("symbol == @symbol").index[0]
# gene_oi = promoters.query("symbol == @symbol")["ix"][0]

# gene_oi = 1127
# gene_id = promoters.index[gene_oi]

# %%
# print(promoters.loc[gene_id, "n"])

# %%
fig, ax = plt.subplots()
sns.heatmap(probs[gene_oi])
fig, ax = plt.subplots()
sns.heatmap(probs_diff[gene_oi])
fig, ax = plt.subplots()
sns.heatmap(probs_diff_masked[gene_oi],cmap=mpl.cm.RdBu_r, vmin = -1, vmax = 1)

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

nbins = np.array(model.mixture.transform.nbins)
bincuts = np.concatenate([[0], np.cumsum(nbins)])
binmids = bincuts[:-1] + nbins/2

ax = main[0, 0] = chd.grid.Ax((10, 0.25))
ax = ax.ax
plotdata = (model.mixture.transform.unnormalized_heights.data.cpu().numpy())[[gene_oi]]
ax.imshow(plotdata, aspect = "auto")
ax.set_yticks([])
for b in bincuts:
    ax.axvline(b-0.5, color = "black", lw = 0.5)
ax.set_xlim(0-0.5, plotdata.shape[1]-0.5)
ax.set_xticks([])
ax.set_ylabel("$h_0$", rotation = 0, ha = "right", va = "center")

ax = main[1, 0] = chd.grid.Ax(dim = (10, n_latent_dimensions * 0.25))
ax = ax.ax
plotdata = (model.decoder.logit_weight.data[gene_oi].data.cpu().numpy())
ax.imshow(plotdata, aspect = "auto", cmap = mpl.cm.RdBu_r, vmax = np.log(2), vmin = np.log(1/2))
ax.set_yticks(range(len(cluster_info)))
ax.set_yticklabels(cluster_info.index, rotation = 0, ha = "right")
for b in bincuts:
    ax.axvline(b-0.5, color = "black", lw = 0.5)
ax.set_xlim(-0.5, plotdata.shape[1]-0.5)

ax.set_xticks(bincuts-0.5, minor = True)
ax.set_xticks(binmids-0.5)
ax.set_xticklabels(nbins)
ax.xaxis.set_tick_params(length = 0)
ax.xaxis.set_tick_params(length = 5, which = "minor")
ax.set_ylabel("$\Delta h$", rotation = 0, ha = "right", va = "center")

ax.set_xlabel("Resolution")

fig.plot()

# %%
if transcriptome is not None:
    sc.pl.umap(transcriptome.adata, color = ["cluster", gene_id], legend_loc = "on data")
    gene_ids = transcriptome.gene_id([
        symbol
    ])
    sc.pl.umap(transcriptome.adata, color = gene_ids, title = transcriptome.symbol(gene_ids))

# %%
gene_id

# %%
# sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["TCF7"]), legend_loc = "on data")

# %% [markdown]
# ## Plot gene

# %% [markdown]
# ### Extract motif data

# %%
import itertools

# %%
# motifs_oi = pd.DataFrame([], columns = ["motif", "clusters"]).set_index("motif")

if motifscan_name in ["cutoff_0001", "cutoff_001"]:
    motifs_oi = pd.DataFrame([
        [motifs.loc[motifs.index.str.contains("SPI1")].index[0], ["B", "Monocytes", "cDCs"]],
        [motifs.loc[motifs.index.str.contains("CEBPB")].index[0], ["Monocytes", "cDCs"]],
        [motifs.loc[motifs.index.str.contains("PEBB")].index[0], ["NK"]],
        [motifs.loc[motifs.index.str.contains("RUNX2")].index[0], ["NK"]],
        [motifs.loc[motifs.index.str.contains("IRF8")].index[0], ["cDCs"]],
        [motifs.loc[motifs.index.str.contains("IRF4")].index[0], ["cDCs", "B", "pDCs", "CD4 T", "CD8 T"]],
        [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["B", "pDCs"]], # TCF3
        [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],   
        [motifs.loc[motifs.index.str.contains("BC11A")].index[0], ["pDCs"]],
        [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["B"]],
        [motifs.loc[motifs.index.str.contains("NFKB2")].index[0], ["B"]],
        [motifs.loc[motifs.index.str.contains("RUNX1")].index[0], ["CD4 T", "CD8 T", "MAIT"]],
        [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["CD4 T", "CD8 T", "MAIT"]],
        [motifs.loc[motifs.index.str.contains("GATA3")].index[0], ["CD4 T", "CD8 T", "MAIT"]],
        [motifs.loc[motifs.index.str.contains("TCF7")].index[0], ["CD4 T", "CD8 T", "MAIT"]],
    ], columns = ["motif", "clusters"]).set_index("motif")

# motifs_oi = pd.DataFrame([
#     [motifs.loc[motifs.index.str.contains("ZN586")].index[0], cluster_info.index] 
# ], columns = ["motif", "clusters"]).set_index("motif")

# motifs_oi = pd.DataFrame([
#     ["cd4et", ["CD4 T"]],
#     ["cd8et", ["CD8 T"]],
#     ["cd8nc", ["CD8 T"]],
#     ["cd4nc", ["CD4 T"]],
#     ["cd4sox4", ["CD4 T"]],
#     ["monoc", ["Monocytes"]],
#     ["dc", ["cDC"]],
#     ["nk", ["NK"]],
#     ["bin", ["B"]],
#     ["plasma", ["Plasma"]],
#     ["bmem", ["B"]],
# ], columns = ["motif", "clusters"]).set_index("motif")

if motifscan_name in ["gwas_immune"]:
    motifs_oi = pd.DataFrame([
        [x, [cluster_info.index[i]]] for x, i in zip(motifs.index, itertools.chain(range(len(cluster_info.index)), range(len(cluster_info.index))))
    ], columns = ["motif", "clusters"]).set_index("motif")

# motifs_oi = pd.DataFrame([
#     [motifs.loc[motifs.index.str.contains("NDF2")].index[0], ["leiden_2"]],
#     [motifs.loc[motifs.index.str.contains("DLX3")].index[0], ["leiden_4"]],
    
# ], columns = ["motif", "clusters"]).set_index("motif")


motifs_oi["ix"] = motifs.loc[motifs_oi.index, "ix"].values
assert len(motifs_oi) == len(motifs_oi.index.unique())
motifs_oi["color"] = sns.color_palette(n_colors = len(motifs_oi))
# motifs_oi["label"] = motifs.loc[motifs_oi.index, "gene_label"]
motifs_oi["label"] = motifs_oi.index

# %%
indptr_start = gene_oi * (window[1] - window[0])
indptr_end = (gene_oi + 1) * (window[1] - window[0])

# %%
motifdata = []
for motif in motifs_oi.index:
    motif_ix = motifs.loc[motif, "ix"]
    for pos in range(indptr_start, indptr_end):
        pos_indices = motifscan.indices[motifscan.indptr[pos]:motifscan.indptr[pos+1]]
        if motif_ix in pos_indices:
            motifdata.append({"position":pos - indptr_start + window[0], "motif":motif})
motifdata = pd.DataFrame(motifdata, columns = ["position", "motif"])
print(len(motifdata))

# %% [markdown] tags=[]
# ### Extract peaks

# %%
promoter = promoters.loc[gene_id]


# %%
def center_peaks(peaks, promoter):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns = ["start", "end", "method"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * promoter["strand"],
                (peak["end"] - promoter["tss"]) * promoter["strand"]
            ][::promoter["strand"]]

            for _, peak in peaks.iterrows()
        ]
    return peaks


# %%
peaks = []

import pybedtools
promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chr", "start", "end"]])

peak_methods = []

for peaks_name in [
    "cellranger",
    "macs2",
    "macs2_improved",
    "macs2_leiden_0.1",
    "macs2_leiden_0.1_merged",
    "genrich",
    # "rolling_500",
    # "rolling_50",
    "encode_screen",
]:
    peaks_bed = pybedtools.BedTool(chd.get_output() / "peaks" / dataset_name / peaks_name / "peaks.bed")
    
    if peaks_name in ["macs2_leiden_0.1"]:
        usecols = [0, 1, 2, 6]
        names = ["chr", "start", "end", "name"]
    else:
        usecols = [0, 1, 2]
        names = ["chr", "start", "end"]
    peaks_cellranger = promoter_bed.intersect(peaks_bed, wb = True, nonamecheck = True).to_dataframe(usecols = usecols, names = names)
    
    if peaks_name in ["macs2_leiden_0.1"]:
        peaks_cellranger = peaks_cellranger.rename(columns = {"name":"cluster"})
        peaks_cellranger["cluster"] = peaks_cellranger["cluster"].astype(int)
        
    # peaks_cellranger = promoter_bed.intersect(peaks_bed).to_dataframe()
    if len(peaks_cellranger) > 0:
        peaks_cellranger["peak"] = peaks_cellranger["chr"] + ":" + peaks_cellranger["start"].astype(str) + "-" + peaks_cellranger["end"].astype(str)
        peaks_cellranger["method"] = peaks_name
        peaks_cellranger = center_peaks(peaks_cellranger, promoter)
        peaks.append(peaks_cellranger.set_index("peak"))
    peak_methods.append({"method":peaks_name})

peaks = pd.concat(peaks).reset_index().set_index(["method", "peak"])
peaks["size"] = peaks["end"] - peaks["start"]

peak_methods = pd.DataFrame(peak_methods).set_index("method")
peak_methods = peak_methods.loc[peak_methods.index.isin(peaks.index.get_level_values("method"))]
peak_methods["ix"] = np.arange(peak_methods.shape[0])
peak_methods["label"] = peak_methods.index

# %%
fig, ax = plt.subplots(figsize = (2, 0.5))
ax.set_xlim(*window)
for i, (_, peak) in enumerate(peaks.query("method == @peaks_name").iterrows()):
    ax.plot([peak["start"], peak["end"]], [i, i])


# %% [markdown]
# ### Extract genes

# %%
def center(coords, promoter):
    coords = coords.copy()
    if promoter.strand == 1:
        coords["start"] = coords["start"] - promoter["start"] + window[0]
        coords["end"] = coords["end"] - promoter["start"] + window[0]
    else:
        coords["start"] = (window[1] - window[0]) - (coords["start"] - promoter["start"]) + window[0]
        coords["end"] = (window[1] - window[0]) - (coords["end"] - promoter["start"]) + window[0]

        coords = coords.rename(columns = {"start":"end", "end":"start"})
        
    return coords


# %%
genes = genes.rename(columns = {"Strand":"strand"})

# %%
plotdata_genes = center(genes.loc[genes["chr"] == promoter["chr"]].query("~((start > @promoter.end) | (end < @promoter.start))").copy(), promoter)
plotdata_genes["ix"] = np.arange(len(plotdata_genes))

# %%
gene_ids = plotdata_genes.index

# %%
query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
			
	<Dataset name = "hsapiens_gene_ensembl" interface = "default" >
		<Filter name = "ensembl_gene_id" value = "{','.join(gene_ids)}"/>
		<Filter name = "transcript_tsl" excluded = "0"/>
        <Filter name = "transcript_is_canonical" excluded = "0"/>

		<Attribute name = "ensembl_gene_id" />
		<Attribute name = "ensembl_gene_id_version" />
		<Attribute name = "exon_chrom_start" />
		<Attribute name = "exon_chrom_end" />
		<Attribute name = "genomic_coding_start" />
		<Attribute name = "genomic_coding_end" />
		<Attribute name = "ensembl_transcript_id" />
		<Attribute name = "ensembl_transcript_id_version" />
	</Dataset>
</Query>"""
url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace("\t", "").replace("\n", "")
from io import StringIO
import requests
session = requests.Session()
session.headers.update({'User-Agent': 'Custom user agent'})
r = session.get(url)
result = pd.read_table(StringIO(r.content.decode("utf-8")))
# result = result.dropna().copy()
plotdata_exons = result[["Gene stable ID", "Exon region start (bp)", "Exon region end (bp)"]].rename(columns = {"Gene stable ID":"gene", "Exon region start (bp)":"start", "Exon region end (bp)":"end"}).dropna()
plotdata_exons = center(plotdata_exons, promoter)

plotdata_coding = result.rename(columns = {"Gene stable ID":"gene", "Genomic coding start":"start", "Genomic coding end":"end"}).dropna()
plotdata_coding = center(plotdata_coding, promoter)

# %% [markdown]
# ### Empirical density

# %%
fragments.create_cut_data()

# %%
plotdata_empirical = []
for cluster_ix in cluster_info["dimension"]:
    fragments_oi = (latent_torch[fragments.cut_local_cell_ix, cluster_ix] != 0) & (fragments.cut_local_gene_ix == gene_oi)
    cut_coordinates = fragments.cut_coordinates[fragments_oi].cpu().numpy()
    cut_coordinates = cut_coordinates * (window[1] - window[0]) + window[0]
    
    n_bins = 300
    cuts = np.linspace(*window, n_bins+1)
    bincounts, bins = np.histogram(cut_coordinates, bins = cuts, range = window)
    binmids = bins[:-1] + (bins[:-1] - bins[1:]) /2
    bindensity = bincounts/cluster_info["lib"][cluster_ix] * fragments.obs["lib"].mean() * n_bins
    
    plotdata_empirical.append(
        pd.DataFrame({
            "binright":bins[1:],
            "binleft":bins[:-1],
            "count":bincounts,
            "density":bindensity,
            "cluster":cluster_ix
        })
    )
plotdata_empirical_bins = pd.concat(plotdata_empirical, ignore_index = True)

# %%
plotdata_empirical = pd.DataFrame({
    "coord": plotdata_empirical_bins[["binleft", "binright"]].values.flatten(),
    "prob":np.log(plotdata_empirical_bins["density"].values.repeat(2)),
    "cluster":plotdata_empirical_bins["cluster"].values.repeat(2)
})

# %%
baseline = np.log(plotdata_empirical.groupby(["cluster"]).apply(lambda x:np.trapz(np.exp(x["prob"]), x["coord"].astype(float) / (window[1] - window[0]))))

# %%
plotdata_empirical["prob"] = plotdata_empirical["prob"] - baseline[~np.isinf(baseline)].mean()

# %%
cluster_info["n_cells"] = fragments.obs.groupby("cluster").size()[cluster_info["dimension"]].values

# %%
x = plotdata_empirical.query("cluster == 0")["coord"]
y = np.exp(plotdata_empirical.query("cluster == 0")["prob"])
plt.plot(x, y)

x = plotdata_empirical.query("cluster == 1")["coord"]
y = np.exp(plotdata_empirical.query("cluster == 1")["prob"])
plt.plot(x, y)

# %% [markdown]
# ### Extract conservaton

# %%
import chromatinhd.conservation
folder_cons = chd.get_output() / "data" / "cons" / "hs" / "gerp"
conservation = chd.conservation.Conservation(folder_cons/"hg38.phastCons100way.bw")

# %%
plotdata_conservation = pd.DataFrame({
    "conservation":conservation.get_values(promoter.chr, promoter.start, promoter.end),
    "position":np.arange(window[0], window[1])[::promoter["strand"]]
})

# %% [markdown]
# ### Extract GC

# %%
import chromatinhd.conservation

# %%
onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))

# %%
x = onehot_promoters[gene_oi][:, [1, 2]].sum(1)

# %%
size = 50

# %%
import math

# %%
plotdata_gc = pd.DataFrame({
    "gc":torch.nn.functional.pad(x, (math.ceil(size/2), math.floor(size//2)), value = x.mean()).unfold(0, size, 1).mean(1)[:-1],
    "position":np.arange(window[0], window[1])
})

# %% [markdown]
# ### Extract chromatin annot

# %%
raw_folder = chd.get_output() / "data" / "chmm" / "raw"

# %%
metadata = pd.read_csv(raw_folder / "metadata.csv", index_col = 0)

# %%
import pybedtools

# %%
plotdata_chromatin_annot = []
promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame.from_records([promoter])[["chr", "start", "end"]])
for _, metadata_row in metadata.iterrows():
    bed = pybedtools.BedTool(raw_folder / metadata_row["file"])
    intersect = bed.intersect(promoter_bed)
    
    plotdata_ = center_peaks(intersect.to_dataframe(), promoter)
    plotdata_["cluster"] = metadata_row["cluster"]
    plotdata_["name"] = plotdata_["name"].str.split("_").str[1]
    plotdata_chromatin_annot.append(plotdata_)
plotdata_chromatin_annot = pd.concat(plotdata_chromatin_annot)

# %% [markdown]
# ### Extract expression data

# %%
plotdata_expression = sc.get.obs_df(transcriptome.adata, [gene_id, "cluster"]).rename(columns = {gene_id:"expression"})
plotdata_expression_clusters = plotdata_expression.groupby("cluster")["expression"].mean()
plotdata_diffexpression_clusters = plotdata_expression_clusters - plotdata_expression_clusters.mean()

norm_expression = mpl.colors.Normalize(0., plotdata_expression_clusters.max(), clip = True)
cmap_expression = mpl.cm.Reds

# %% [markdown]
# ### Create plotdata

# %%
plotdata_atac = design.query("gene_ix == @gene_oi").copy().rename(columns = {"active_latent":"cluster"}).set_index(["coord", "cluster"]).drop(columns = ["batch", "gene_ix"])
plotdata_atac["prob"] = probs[gene_oi].flatten()
plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

plotdata_atac["prob"] = plotdata_atac["prob"] - np.log(plotdata_atac.reset_index().groupby(["cluster"]).apply(lambda x:np.trapz(np.exp(x["prob"]), x["coord"].astype(float) / (window[1] - window[0])))).mean()
plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

plotdata_genome = plotdata_atac
plotdata_genome_mean = plotdata_atac_mean

# peaks_name = "rolling_50"
# plotdata_genome = plotdata_peaks.loc[peaks_name]
# plotdata_genome_mean = plotdata_peaks_mean.loc[peaks_name]

# %%
norm_atac_diff = mpl.colors.Normalize(-1, 1., clip = True)
cmap_atac_diff = mpl.cm.RdBu_r

# %% [markdown]
# ### Plot

# %% tags=[]
import chromatinhd.grid
main = chd.grid.Grid(3, 3, padding_width = 0.1, padding_height = 0.1)
fig = chd.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005
panel_width = (window[1] - window[0]) * resolution

# gene annotation
ax_gene = main[0, 1] = chd.differential.plot.Genes(
    plotdata_genes,
    plotdata_exons,
    plotdata_coding,
    gene_id,
    promoter, 
    window,
    panel_width
)

panel_height = 0.5

# differential atac
wrap_differential = main[1, 1] = chd.differential.plot.Differential(
    plotdata_genome,
    plotdata_genome_mean,
    cluster_info,
    window,
    panel_width,
    panel_height,
    plotdata_empirical = plotdata_empirical,
    padding_height = padding_height,
    ymax = 20
)
# [artist.set_visible(False) for artist in wrap_differential.get_artists()]
# [artist.set_visible(False) for artist in wrap_differential.title.ax.texts]

# highlight motifs
show_motifs = True
if show_motifs:
    # motifdata = motifdata.query("motif in ['SPI1_HUMAN.H11MO.0.A', 'CEBPB_HUMAN.H11MO.0.A']")
    # motifdata = motifdata.query("motif in ['SPI1_HUMAN.H11MO.0.A']")
    # motifdata = motifdata.query("motif in ['IRF4_HUMAN.H11MO.0.A']")
    chd.differential.plot.MotifsHighlighting(wrap_differential, motifdata, motifs_oi, cluster_info)
    wrap_motiflegend = main[1, 2] = chd.differential.plot.MotifsLegend(
        motifs_oi,
        cluster_info,
        1,
        panel_height,
        padding_height = padding_height
    )

show_expression = True
if show_expression:
    wrap_expression = main[1, 0] = chd.differential.plot.DifferentialExpression(
        plotdata_expression,
        plotdata_expression_clusters,
        cluster_info,
        0.3,
        panel_height,
        padding_height = padding_height
    )

show_peaks = True
if show_peaks:
    ax_peaks = main[2, 1] = chd.differential.plot.Peaks(peaks, peak_methods, window, panel_width)

show_lower = True
if show_lower:
    ax_conservation = main[3, 1] = chd.differential.plot.Conservation(plotdata_conservation, window, panel_width)
    ax_gc = main[4, 1] = chd.differential.plot.GC(plotdata_gc, window, panel_width)
    ax_annot = main[5, 1] = chd.differential.plot.Annot(plotdata_chromatin_annot, window, panel_width, cluster_info)
    main[5, 2] = chd.differential.plot.AnnotLegend(ax_annot, width = 1)

# set x ticks
fig.plot()

# %%
promoter.to_dict()

# %%
gene_ids = transcriptome.gene_id(["MZF1"])

# %%
if transcriptome is not None:
    sc.pl.umap(transcriptome.adata, color = ["cluster", gene_id], legend_loc = "on data")
    gene_ids = transcriptome.gene_id([
        symbol
    ])
    sc.pl.umap(transcriptome.adata, color = gene_ids, title = transcriptome.symbol(gene_ids))

# %% [markdown]
# ### Animated

# %%
fig.set_tight_bounds()

# %%
import matplotlib.animation

# %%
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150  


# %%
def interpolate(x, y, a):
    return x + (y-x) * a


# %%
t = 0
plotdata_genome_keyframes = []
keyframes_genome = []

keyframes_title = []
plotdata_title_keyframes = []

time_standstill = 1500
time_transition = 1500

for peaks_name in ["cellranger", "macs2_improved", "encode_screen", "rolling_50", "rolling_500", "our", "our", "cellranger"]:
# for peaks_name in ["cellranger", "macs2_improved"]:
    if peaks_name == "our":
        plotdata_genome = plotdata_atac
        plotdata_genome_mean = plotdata_atac_mean
        peakcaller_label = "ChromatinHD"
    else:
        plotdata_genome = plotdata_peaks.loc[peaks_name]
        plotdata_genome_mean = plotdata_peaks_mean.loc[peaks_name]
        peakcaller_label = peaks_name
        
    if t == 0:
        plotdata_genome_keyframes.extend([[plotdata_genome, plotdata_genome_mean], [plotdata_genome, plotdata_genome_mean]])
        keyframes_genome.extend([{"t":t}, {"t":t+time_standstill}])
    else:
        plotdata_genome_keyframes.extend([[plotdata_genome, plotdata_genome_mean], [plotdata_genome, plotdata_genome_mean]])
        keyframes_genome.extend([{"t":t}, {"t":t+time_standstill}])
    
    if t == 0:
        keyframes_title.append({"t":t})
        plotdata_title_keyframes.append({"label":peakcaller_label, "alpha":1})
        keyframes_title.append({"t":t+time_standstill})
        plotdata_title_keyframes.append({"label":peakcaller_label, "alpha":1})
    else:
        keyframes_title.append({"t":t-time_transition/2})
        plotdata_title_keyframes.append({"alpha":1, "label":prev_peakcaller_label})
        keyframes_title.append({"t":t-time_transition/4})
        plotdata_title_keyframes.append({"label":peakcaller_label, "alpha":0})
        keyframes_title.append({"t":t})
        plotdata_title_keyframes.append({"alpha":1, "label":peakcaller_label})
    prev_peakcaller_label = peakcaller_label
    
    t += time_standstill + time_transition
keyframes_genome = pd.DataFrame(keyframes_genome)
keyframes_title = pd.DataFrame(keyframes_title)

# %%
total_time = keyframes_genome["t"].max()
fps = 30
delay = 1000/fps
frame_design = pd.DataFrame({
    "t":np.hstack([np.arange(0, total_time, delay), total_time]),
})
n_frames = frame_design.shape[0]


# %%
def get_keyframes(keyframes, t):
    if t == 0:
        keyframe_1 = keyframes.iloc[0]
        keyframe_2 = keyframe_1
        s = 0
    elif t >= keyframes["t"].max():
        keyframe_1 = keyframe_2 = keyframes.iloc[-1]
        s = 0
    else:
        keyframe_1 = keyframes.iloc[np.searchsorted(keyframes["t"], t)-1]
        keyframe_2 = keyframes.iloc[keyframe_1.name+1]
        s = (t - keyframe_1["t"]) / (keyframe_2["t"] - keyframe_1["t"])
    return keyframe_1, keyframe_2, s


# %%
frames = []

for t in frame_design["t"]:
    artists = []
    
    # genome
    keyframe_1, keyframe_2, s = get_keyframes(keyframes_genome, t)
    
    plotdata_genome_1, plotdata_genome_mean_1 = plotdata_genome_keyframes[keyframe_1.name]
    plotdata_genome_2, plotdata_genome_mean_2 = plotdata_genome_keyframes[keyframe_2.name]
    
    plotdata_genome = np.log(interpolate(np.exp(plotdata_genome_1), np.exp(plotdata_genome_2), s))
    plotdata_genome_mean = np.log(interpolate(np.exp(plotdata_genome_mean_1), np.exp(plotdata_genome_mean_2), s))
    
    wrap_differential.draw(plotdata_genome, plotdata_genome_mean)
    artists.extend(wrap_differential.get_artists())
    
    # title
    keyframe_1, keyframe_2, s = get_keyframes(keyframes_title, t)
    
    plotdata_1 = plotdata_title_keyframes[keyframe_1.name]
    plotdata_2 = plotdata_title_keyframes[keyframe_2.name]
    
    alpha = plotdata_1["alpha"] + (plotdata_2["alpha"] - plotdata_1["alpha"]) * s
    
    text = wrap_differential.title.ax.text(0.5, 0.5, plotdata_1["label"], ha="center", va="center", size="large", alpha = alpha)
    artists.append(text)
    
    frames.append(artists)

# %%
anim = mpl.animation.ArtistAnimation(fig, frames, blit=True, interval = total_time/n_frames)

# %%
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import tqdm.auto as tqdm
pbar = tqdm.tqdm(total = n_frames)
anim.save(
    "anim.gif", progress_callback = lambda x, _ :pbar.update(), writer=mpl.animation.FFMpegFileWriter(bitrate = 5000, fps = n_frames/total_time*1000, codec="libx264")
)

# %%
import IPython.display
IPython.display.Image("anim.gif")

# %%
anim.save(
    "anim.mp4", progress_callback = lambda x, _ :pbar.update(), writer=mpl.animation.FFMpegFileWriter(bitrate = 5000, fps = n_frames/total_time*1000, codec="libx264")
)

# %% language="html"
# <video controls loop style="width:100%">
#   <source src="anim.mp4" type="video/mp4">
# </video>

# %%

# %% [markdown]
# ## Old

# %% [markdown]
# ### Temporary look at whatever

# %%
scores_dir = (prediction.path / "scoring" / "significant_up" / motifscan_name)
motifscores = pd.read_pickle(scores_dir / "motifscores.pkl")
scores = pd.read_pickle(scores_dir / "scores.pkl")
print(scores["n_position"])

# %%
peaks_name = "cellranger"
# peaks_name = "macs2"
# peaks_name = "macs2_improved"
# peaks_name = "rolling_500"

# %%
scores_dir = (prediction.path / "scoring" / peaks_name / motifscan_name)
motifscores_all = pd.read_pickle(scores_dir / "motifscores_all.pkl")
scores = pd.read_pickle(scores_dir / "scores.pkl")
# genemotifscores_all = pd.read_pickle(scores_dir / "genemotifscores_all.pkl")
print(scores["n_position"])

# %%
x = motifscores_all.query("cluster == 'pDCs'").sort_values("logodds_peak", ascending = False)

# %%
motifscores_all.sort_values("logodds_region", ascending = False).head(10)

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(motifscores_all["in_peak"], motifscores["in"])
ax.scatter(motifscores_all["in_peak"], motifscores_all["in_region"])
ax.axline([0, 0], slope = 1)

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
# ax.scatter(
#     np.exp(motifscores_all["logodds_peak"]),
#     np.exp(motifscores["logodds"]),
#     s = 1
# )
ax.scatter(
    np.exp(motifscores_all["logodds_peak"]),
    np.exp(motifscores_all["logodds_region"]),
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


# %%
def plot_motifscores(ax, motifscores_all):
    ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
    ax.scatter(
        np.exp(motifscores_all["logodds_peak"]),
        np.exp(motifscores_all["logodds_region"]),
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
    # ax.set_xlabel("Odds-ratio differential peaks")
    # ax.set_ylabel("Odds-ratio\ndifferential\nChromatinHD\nregions", rotation = 0, va = "center", ha = "right")
    
main = chd.grid.Wrap()
fig = chd.grid.Figure(main)

for cluster in cluster_info.index:
    ax_ = main.add(chd.grid.Ax((2, 2)))
    
    ax_.ax.set_title(cluster)
    
    motifscores_cluster = motifscores_all.query("cluster == @cluster")
    plot_motifscores(ax_.ax, motifscores_cluster)
    
    import scipy.stats
    linreg = scipy.stats.linregress(motifscores_cluster["logodds_region"], motifscores_cluster["logodds_peak"])
    linreg = scipy.stats.linregress(motifscores_cluster["logodds_peak"], motifscores_cluster["logodds_region"])
    slope = linreg.slope
    intercept = linreg.intercept
    
    ax_.ax.axline((np.exp(0), np.exp(0)), (np.exp(1), np.exp(slope)), color = "orange")
    
    print(1/slope)
    
    # ax = a
fig.plot()

# %%
motifscores_all.query("cluster == 'leiden_0'").sort_values("logodds_region")

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
# ax.scatter(
#     np.exp(motifscores_all["logodds_peak"]),
#     np.exp(motifscores["logodds"]),
#     s = 1
# )
ax.scatter(
    np.exp(motifscores_all["logodds_peak"]),
    np.exp(motifscores_all["logodds_region"]),
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

# %%
print("same # of positions, all motifs odds:", np.exp((motifscores_all["logodds_region"] - motifscores_all["logodds_peak"]).mean()))
print("significant positions, all motifs odds:", np.exp((motifscores["logodds"] - motifscores_all["logodds_peak"]).mean()))
print("same # of positions, all motifs in region:", (motifscores_all["in_region"]).mean() / (motifscores_all["in_peak"]).mean())
print("significant positions, all motifs in region:", (motifscores["in"]).mean() / (motifscores_all["in_peak"]).mean())

# %%
motifscores_all["in_region"].plot(kind = "hist")
motifscores_all["in_peak"].plot(kind = "hist")

# %%
motifscores_oi = motifs_oi.explode("clusters").rename(columns = {"clusters":"cluster"}).reset_index().set_index(["motif", "cluster"]).join(motifscores_all.reset_index().set_index(["motif", "cluster"]))

# %%
(motifscores_oi["logodds_region"] - motifscores_oi["logodds_peak"]).mean()

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.axline([0, 0], slope = 1)
plt.scatter(motifscores_all["logodds_peak"], motifscores_all["logodds_region"], color = "grey")
plt.scatter(motifscores_oi["logodds_peak"], motifscores_oi["logodds_region"])

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.axline((0, 0), slope = 1)
# ax.scatter(plotdata["logodds_peak"], plotdata["logodds_region"], s = 1, c = plotdata["cluster"].astype("category").cat.codes)
ax.scatter(motifscores_oi["logodds_peak"], motifscores_oi["logodds_region"], s = 3, color = "orange")
for (motif, group), motifscore_oi in motifscores_oi.iterrows():
    ax.text(motifscore_oi["logodds_peak"], motifscore_oi["logodds_region"], motif, color = "orange", ha = "center", va = "center")

# %%
fig, ax = plt.subplots()
logodds = motifscores_all.reset_index().set_index(["motif", "cluster"])["logodds_region"].unstack()
sns.heatmap(logodds)

fig, ax = plt.subplots()
logodds_norm = (logodds - logodds.values.mean(1, keepdims = True)) / logodds.values.std(1, keepdims = True)
sns.heatmap(logodds_norm)

# %%
motifscores_all.query("cluster == 'leiden_4'").sort_values("logodds_region", ascending = False).head(20)

# %% [markdown] tags=[]
# ### Extract peak counts and probs

# %%
probs_peaks_names = ["cellranger", "macs2_improved", "encode_screen", "rolling_500", "rolling_50"]
# probs_peaks_names = ["rolling_500"]

# %%
import chromatinhd.peakcounts

# %%
cluster_labels = latent.idxmax(1)

# %%
peak_cluster_probs = []
cell_to_cluster = np.where(latent)[1]
n_clusters = latent.shape[1]
cut_oi_all = (fragments.cut_local_gene_ix == gene_oi)
for peaks_name, peaks_method in peaks.loc[probs_peaks_names].groupby("method"):
    print(peaks_name)
    peakcounts = chd.peakcounts.FullPeak(folder = chd.get_output() / "peakcounts" / dataset_name / peaks_name)
    
    peak_ixs = peakcounts.var.loc[peaks_method.index.get_level_values("peak")]["ix"]
    
    peak_cluster_counts = []
    peak_cluster_lfcs = []
    peak_cluster_libs = []
    for cluster in latent.columns:
        peak_cluster_counts.append(np.array(peakcounts.counts[:, peak_ixs][cluster_labels == cluster].sum(0))[0])
        peak_cluster_libs.append((cluster_labels == cluster).sum())
        
        # y_out = np.log(np.array(peakcounts.counts[:, peak_ixs].todense()) + 1e-8)
        # x_out = (cluster_labels == cluster).values.astype(float)
        
        # ridge = sklearn.linear_model.Ridge(alpha = 0.01)
        # ridge.fit(x_out[:, None], y_out)
        
        # peak_cluster_lfcs.append(ridge.coef_[:, 0])
        
    peak_cluster_counts = np.vstack(peak_cluster_counts)
    peak_cluster_libs = np.array(peak_cluster_libs)
    peak_cluster_expression = peak_cluster_counts / peak_cluster_libs[:, None]
    peak_cluster_lfcs = np.log(peak_cluster_expression / peak_cluster_expression.mean(0, keepdims = True))
    # peak_cluster_lfcs = np.vstack(peak_cluster_lfcs)
    
    # mean peak counts
    peak_probs = np.log((peak_cluster_counts / peak_cluster_libs[:, None]).mean(0) * (window[1] - window[0]) / peaks_method.loc[[peaks_name]]["size"])
    
    peak_lfc = pd.DataFrame(peak_cluster_lfcs, index = pd.Series(cluster_info["dimension"], name = "cluster"), columns = peak_ixs.index)
    
    peak_cluster_probs_method = peak_lfc.unstack() + peak_probs
    
    peak_cluster_probs.append(peak_cluster_probs_method.to_frame(name = "prob"))
peak_cluster_probs = pd.concat(peak_cluster_probs).join(peaks.drop(columns = ["cluster"], errors = "ignore"))

# %%
import scipy
import scipy.interpolate

# %%
design_coord = design.query("(gene_ix == @gene_oi) & (active_latent == 0)")
pseudocoordinates = design_coord["coord"].values

# %%
plotdata_peaks = []
for (method, cluster), peakcounts_oi in peak_cluster_probs.groupby(["method", "cluster"]):
    xs = []
    ys = []
    
    xs.append(window[0])
    ys.append(-np.inf)
    for _, peak in peakcounts_oi.sort_values("start").iterrows():
        xs.extend([peak["start"], peak["start"],peak["end"], peak["end"]])
        ys.extend([-np.inf, peak["prob"], peak["prob"], -np.inf])
    xs.append(window[1])
    ys.append(-np.inf)
    
    plotdata_peaks.append(
        pd.DataFrame({"method":method, "cluster":cluster, "coord":pseudocoordinates, "prob":scipy.interpolate.interp1d(xs, ys)(pseudocoordinates)})
    )
plotdata_peaks = pd.concat(plotdata_peaks).set_index(["method", "coord", "cluster"])
plotdata_peaks["prob"] = plotdata_peaks["prob"] - np.log(plotdata_peaks.reset_index().groupby(["method", "cluster"]).apply(lambda x:np.trapz(np.exp(x["prob"]), x["coord"].astype(float) / (window[1] - window[0])))).groupby("method").mean()

# %%
plotdata_peaks_mean = plotdata_peaks.replace(-np.inf, np.nan).groupby(["method", "coord"])["prob"].apply(lambda g: g.mean(skipna=True)).replace(np.nan, -np.inf).to_frame()
plotdata_peaks["prob_diff"] = (plotdata_peaks["prob"] - plotdata_peaks_mean["prob"].replace(-np.inf, np.nan)).replace(np.nan, -np.inf)

# %%
peaks_name = "rolling_50"

# %%
plotdata_peaks.loc["rolling_50"].

# %%
np.exp(plotdata_peaks_mean["prob"]).loc[peaks_name].plot()
np.exp(plotdata_peaks.xs(cluster_info.loc["Plasma", "dimension"], level = "cluster")["prob"]).loc[peaks_name].plot()

# %%
np.trapz(np.exp(plotdata_peaks_mean["prob"].loc[peaks_name]), x= np.linspace(0, 1, len(plotdata_peaks_mean["prob"].loc[peaks_name])))

# %%
