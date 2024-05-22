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
# ## Get the dataset

# %%
# dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "pbmc10kx"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
# dataset_name = "hspc"
# dataset_name = "hspc_cycling"
# dataset_name = "hspc_meg_cycling"
# dataset_name = "hspc_gmp_cycling"
# dataset_name = "liver"
regions_name = "100k100k"
# regions_name = "10k10k"
latent = "leiden_0.1"
# latent = "phase"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
cluster_counts = pd.DataFrame(transcriptome.X[:], index = transcriptome.obs.index).groupby(clustering.labels).mean()
# pca
import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=cluster_counts.shape[0], whiten = True)
# pca.fit(cluster_counts)
# components = pca.components_
components = pca.fit_transform(cluster_counts)
a, b = 0, 1
plt.scatter(components[:, a], components[:, b])
# plt.scatter(components[a, :], components[b, :])
plt.xlabel(f"PC{a}")
plt.ylabel(f"PC{b}")
plt.title("PCA of cluster counts")
for i, txt in enumerate(cluster_counts.index):
    # plt.annotate(txt, (components[a, i], components[b, i]))
    plt.annotate(txt, (components[i, a], components[i, b]))

# %%
# fragments2 = fragments.filter_regions(
#     fragments.regions.filter(transcriptome.gene_id(["IRF1", "CCL4"]))
# )
# fragments2.create_regionxcell_indptr()
fragments2 = fragments

# %%
model_params=dict(
    encoder="shared",
    encoder_params=dict(
        binwidths=(5000, 1000, 500, 100, 50, 25),
    ),
)
train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=150, lr=1e-3)

chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31", reset = True)
models = chd.models.diff.model.binary.Models.create(
    path = chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31",
    model_params = model_params,
    train_params = train_params,
    overwrite = True,
    fragments = fragments,
    clustering = clustering,
    folds = folds
)


# %%
# import chromatinhd.models.diff.model.binary
# model = chd.models.diff.model.binary.Model.create(
#     fragments = fragments2,
#     clustering = clustering,
#     folds = folds,
#     # encoder = "shared_lowrank",
#     encoder = "shared",
#     # encoder = "split",
#     encoder_params=dict(
#         delta_regularization=True,
#         delta_p_scale = 1.5,
#         bias_regularization=True,
#         binwidths = (5000, 1000, 500, 100, 50, 25),
#         # transcriptome = transcriptome,
#     ),
#     path = chd.get_output() / "model_test_100k",
#     overwrite = True
# )
# self = model

# %%
# model.encoder.w_0.weight.data[0, 0] = 1
# model.encoder.w_1.weight.data[0, 2] = 1
# model.encoder.w_1.weight.data[0, 10] = 1
# model.encoder.w_1.weight.data[0, 100] = 1
# w, kl = model.encoder._calculate_w(torch.tensor([0]))

# final_w = w.detach()#.repeat_interleave(model.encoder.total_width // model.encoder.n_final_bins, -1).detach()
# (torch.exp(final_w)[0, 0].repeat_interleave(model.encoder.total_width // model.encoder.n_final_bins, -1)).sum()

# fig, ax = plt.subplots(1, 1, figsize=(20, 2))
# plt.plot((final_w)[0, 0])

# assert set(model.parameters_sparse()).__contains__(model.encoder.w_delta_0.weight)
# assert set(model.parameters_sparse()).__contains__(model.overall_delta.weight)

# %% [markdown]
# ## Test

# %%
# loader = chd.loaders.clustering_fragments.ClusteringCuts(fragments, clustering, 50000)

# symbol_oi = "IRF1"
# minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.n_cells), fragments.var.index.get_indexer(transcriptome.gene_id(["IL1B", "CCL4"])))
# data = loader.load(minibatch)

# coords = torch.clamp(data.cuts.coordinates, self.window[0], self.window[1] - 1) - self.window[0]
# bin_ix = coords // self.encoder.binwidths[-1]

# %%
# model.forward(data)

# %%
# len(list(model.encoder.parameters_sparse()))

# %% [markdown]
# ## Train

# %%
# model.train_model(
#     lr = 1e-3,
#     early_stopping = False,
#     n_epochs = 100,
#     n_cells_step = 5000
# )
models.train_models(n_workers_train = 5, n_workers_validation = 2)

# %%
model.trace.plot();

# %%
loader = chd.loaders.clustering_fragments.ClusteringCuts(fragments2, clustering, 500000)

genes_oi = fragments2.var.index[:100]
gene_ixs = fragments2.var.index.get_indexer(genes_oi)
minibatch = chd.loaders.minibatches.Minibatch(fold["cells_validation"], gene_ixs)
data = loader.load(minibatch)

# %%
# multiplier = torch.tensor(1.0, requires_grad=True)
# elbo = model.forward(data, w_delta_multiplier = multiplier)
# elbo.backward()
# multiplier.grad

# %%
# scores = []
# multipliers = np.linspace(0.8, 1.2, 10)
# for multiplier in multipliers:
#     elbo = model.forward(data, w_delta_multiplier = multiplier)
#     scores.append(elbo.item())

# plt.plot(multipliers, scores, marker = ".")

# %% [markdown]
# ## Evaluate

# %%
# for models
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")
# regionpositional.score(
#     models,
#     device="cpu",
# )
regionpositional

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(chd.get_output() / "test_regionpositional2", reset=True)
regionpositional.score(
    models,
    # models[:1],
    # [model],
    # [models["0"]],
    fragments = fragments2,
    clustering = clustering,
    # regions=transcriptome.gene_id([
    #     "CEBPD",
    # ]),
    device="cpu",
    force = True,
)

# %%
# test whether we got a probability
# symbol = "CEBPD"
region_ix = fragments.var.index.get_loc(transcriptome.gene_id(symbol))

ct = "G2M"
# ct = "CD14+ Monocytes"
# ct = "NK"
# ct = "KC"
# ct = "Central Hepatocyte"
# ct = "Immune"
ct_ix = clustering.cluster_info.index.get_loc(ct)

plotdata, plotdata_mean = regionpositional.get_plotdata(transcriptome.gene_id(symbol))
probs = np.exp(plotdata.loc[ct]["prob"])

gene_ix = fragments.var.index.get_loc(transcriptome.gene_id(symbol))
cells_oi = clustering.labels == ct
np.trapz(probs, plotdata.loc[ct].index)* 2/100/100, fragments.counts[cells_oi, gene_ix].mean()

# %% [markdown]
# ### Viz

# %%
transcriptome.var

# %%
symbol = "SPI1"
symbol = "CEBPD"
symbol = "FGFR1"
symbol = "BCL2"
symbol = "CEBPA"
# symbol = "MKI67"
# symbol = "RUNX1"
# symbol = "FLI1"
# symbol = "NFE2"
# symbol = "SLC24A1"
# symbol = "BCL2"
# symbol = "MKI67"
# symbol = "SWI5"
# symbol = "PTPRC"
# symbol = "E2F2"
# symbol = "PCNA"
# symbol = "E2F2"
# symbol = "SPI1"
# symbol = "CCL4"
# symbol = "Dll4"
# symbol = "Glul"
# symbol = "Alb"
# symbol = "Cyp2f2"
# symbol = "Hamp"
# symbol = "Cyp2e1"
# symbol = "Gls2"
# symbol = "Cdh1"
# symbol = "BCL2"
# symbol = "Ptprc"
# symbol = "Clec4f"
# symbol = "Stab2"
# symbol = "Spi1"
# symbol = "GATA1"
# symbol = "C5AR1"
# symbol = "NOL3"
# symbol = "IRF1"
# symbol = "IL1B"
# symbol = "CD74"
# symbol = "TCF4"
# symbol = "CD79A"
# symbol = "QKI"
# symbol = "GZMH"
# symbol = "RRP12"
# symbol = "Neurod1"

gene_id = transcriptome.gene_id(symbol)
# gene_id = "ENSG00000141552"
# gene_id = "ENSG00000145700"

# %%
plotdata, plotdata_mean = regionpositional.get_plotdata(gene_id)

# %%
plotdata["prob_diff"] = (plotdata.unstack() - plotdata.unstack().mean(0).values[None, :]).stack()["prob"].values
fig, ax = plt.subplots()
for ct in plotdata.index.get_level_values("cluster").unique():
    ax.plot(plotdata.loc[ct].index, plotdata.loc[ct]["prob"], label = ct)
ax.set_xlim(-10000, 20000)

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
motifscan.motifs["label"] = motifscan.motifs["HUMAN_gene_symbol"]
clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
regions = regionpositional.select_regions(gene_id)

# %%
# from scipy.ndimage import convolve
# def spread_true(arr, width = 5):
#     kernel = np.ones(width, dtype=bool)
#     result = convolve(arr, kernel, mode='constant', cval=False)
#     result = result != 0
#     return result

# plotdata, plotdata_mean = regionpositional.get_plotdata(gene_id)
# selection = pd.DataFrame(
#     {"chosen":(plotdata["prob"].unstack() > -1.).any()}
# )
# selection["chosen"] = spread_true(selection["chosen"], width = 5)

# # select all contiguous regions where chosen is true
# selection["selection"] = selection["chosen"].cumsum()

# regions = pd.DataFrame(
#     {
#         "start":selection.index[(np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == 1)[:-1]],
#         "end":selection.index[(np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == -1)[1:]]
#     }
# )
# regions["distance_to_next"] = regions["start"].shift(-1) - regions["end"]

# # merge regions that are close to each other
# regions["merge"] = (regions["distance_to_next"] < 1000).fillna(False)
# regions["merge"] = regions["merge"] | regions["merge"].shift(1).fillna(False)
# regions["group"] = (~regions["merge"]).cumsum()
# regions = regions.groupby("group").agg({"start":"min", "end":"max", "distance_to_next":"last"}).reset_index(drop=True)

# regions["length"] = regions["end"] - regions["start"]
# regions = regions[regions["length"] > 200]
# regions

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

window = fragments.regions.window
# window = [-50000, 50000]
# window = [-20000, 20000]
# window = [-20000, 20000]
# window = [-10000, 10000]
# window = [-25000, -15000]
# window = [-25000-10000, -15000+10000]
# window = [-100000, -90000]
# window = [-60000, -50000]
# window = [-50000, -40000]
# window = [-20000, 0]
# window = [0, 100000]
# window = [-20000, 2000]

width = (window[1] - window[0]) / 2000

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10" if dataset_name == "liver" else "GRCh38")
fig.main.add_under(panel_genes)

cluster_info = clustering.cluster_info
# cluster_info = clustering.cluster_info.loc[["Portal Hepatocyte", "Central Hepatocyte", "Mid Hepatocyte"]]
# cluster_info = clustering.cluster_info.loc[["Lymphoma", "Lymphoma cycling", "B"]]
plotdata, plotdata_mean = regionpositional.get_plotdata(gene_id, clusters = cluster_info.index)

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=gene_id, panel_height=0.4, order = True, cluster_info = cluster_info, layer = "normalized",
)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=cluster_info, panel_height=0.4, width=width, window = window, order = panel_expression.order, ymax = 20, relative_to = "G1"
)
# panel_differential[0].ax.axhline(np.exp(0), color = "grey")
# panel_differential[0].ax.axhline(np.exp(-1), color = "grey")

fig.main.add_under(panel_differential)
fig.main.add_right(panel_expression, row=panel_differential)

motifs_oi = motifscan.motifs.loc[[
    motifscan.motifs.index[motifscan.motifs.index.str.contains("E2F2")][0],
    motifscan.motifs.index[motifscan.motifs.index.str.contains("E2F3")][0],
    motifscan.motifs.index[motifscan.motifs.index.str.contains("E2F4")][0],
    "PAX7.H12CORE.1.S.B",
    "NFAC4.H12CORE.1.SM.B",
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("CEBPE")][0],
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("TCF7")][0],
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("ZBT14")][0],
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("ONEC3")][0],
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("HNF6")][0],
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("CUX1")][0]
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("PAX5")][0],
    # motifscan.motifs.index[motifscan.motifs.index.str.contains("PO2")][0],
]]
panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi, width = width, window = window)

fig.main.add_under(panel_motifs)

# import chromatinhd_manuscript as chdm
# panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = window, width = width)
# fig.main.add_under(panel_peaks)

fig.plot()
fig.savefig("test.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### Differential slices

# %%
regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
slices = regionpositional.calculate_slices(-1.5, step = 25)
differential_slices = regionpositional.calculate_differential_slices(slices, fc_cutoff = 1.5)

# slices = regionpositional.get_slices(0.)
# differential_slices = regionpositional.get_differential_slices()

# %%
slicescores = differential_slices.get_slice_scores(regions = fragments.regions, clustering = clustering)
# slicescores.query("cluster == 'Plasma'").groupby("region")["length"].sum().sort_values()

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
n_desired_positions = slicescores.groupby("cluster")["length"].sum()

# %%
pd.DataFrame({
    "chd":slicescores.groupby("cluster")["length"].sum().sort_values(ascending = False),
})

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

motifscan.motifs["label"] = motifscan.motifs["HUMAN_gene_symbol"]
clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
slicescores = differential_slices.get_slice_scores(regions=fragments.regions, clustering=clustering)

slicescores["slice"] = pd.Categorical(
    slicescores["region_ix"].astype(str)
    + ":"
    + slicescores["start"].astype(str)
    + "-"
    + slicescores["end"].astype(str)
)
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

motifscan = chd.data.motifscan.MotifscanView(
    chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name
)

# count motifs in slices
slicecounts = motifscan.count_slices(slices)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores, slicecounts)
enrichment["log_odds"] = np.log(enrichment["odds"])

# %%
enrichment.query("q_value < 0.05").sort_values("odds", ascending = False).loc["G2M"].head(20)

# %%
enrichment.query("q_value < 0.05").sort_values("odds", ascending = False).head(30)

# %%
import pysam
fasta_file = "/data/genome/mm10/mm10.fa"
# fasta_file = "/data/genome/GRCh38/GRCh38.fa"

fasta = pysam.FastaFile(fasta_file)
slicefeatures = []
for _, (chrom, start, end) in slices[["chrom", "start_genome", "end_genome"]].iterrows():
    if start > end:
        start, end = end, start
    seq = fasta.fetch(chrom, start, end)
    slicefeatures.append({"GC":(seq.count("G") + seq.count("C") + seq.count("g") + seq.count("c")), "length":len(seq)})
slicefeatures = pd.DataFrame(slicefeatures, index=slices.index)
slicefeatures["GC_norm"] = slicefeatures["GC"] / slicefeatures["length"]

# %%
slicescores["GC"] = slicefeatures["GC"].loc[slicescores["slice"]].values
slicecounts_norm = slicecounts / (slicefeatures["length"]+1e-3).values[:, None]
slicefeatures_norm = slicefeatures / (slicefeatures["length"]+1e-3).values[:, None]


# %%
def smooth_spline_fit(x, y, x_smooth):
    import rpy2.robjects as robjects

    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)

    r_smooth_spline = robjects.r["smooth.spline"]
    spline1 = r_smooth_spline(x=r_x, y=r_y, nknots=20)
    ySpline = np.array(
        robjects.r["predict"](spline1, robjects.FloatVector(x_smooth)).rx2("y")
    )

    return ySpline


def score_gc_corrected(gc, count, outcome):
    count_smooth = smooth_spline_fit(gc, count, gc)
    residuals = count - count_smooth

    lm2 = scipy.stats.linregress(outcome, residuals)

    return {
        "slope": lm2.slope,
        "r": lm2.rvalue,
        "p": lm2.pvalue,
    }



# %%
gc_normalization = {}
for motif in slicecounts_norm.columns:
    x = np.linspace(0, 1, 21)
    i = (len(x)//2)-1
    y = smooth_spline_fit(slicefeatures["GC_norm"][:1000], slicecounts[motif][:1000]/slicefeatures["length"][:1000], x)
    y = np.clip(y, 1e-3, np.inf)

    correction = np.interp(slicefeatures["GC_norm"], x, y) / y[i]
    gc_normalization[motif] = slicecounts[motif].values / correction

# %%
slicecounts_corrected = pd.DataFrame(gc_normalization, index = slicecounts.index)

# %%
import scipy.stats
motif = "KMT2A.H12CORE.0.P.B"
motif = "KLF9.H12CORE.1.P.B"
# motif = "NDF1.H12CORE.0.P.B"
lm = scipy.stats.linregress(slicefeatures_norm["GC"],slicecounts_norm[motif])

fig, ax = plt.subplots()
ax.scatter(slicefeatures["GC_norm"], slicecounts[motif]/slicefeatures["length"], color = "blue", s = 1)
ax.scatter(slicefeatures["GC_norm"], slicecounts_corrected[motif]/slicefeatures["length"], color = "orange", s = 1)

# %%
slicecounts = motifscan.count_slices(slices)
enrichment2 = enrichment_cluster_vs_clusters(slicescores, slicecounts_corrected)

# %%
slicecounts_peak = motifscan.count_slices(slices_peak)
enrichment_peak = enrichment_cluster_vs_clusters(slicescores_peak, slicecounts_peak)

# %%
pd.DataFrame({
    "chd":(slicecounts.sum() / slicecounts_all.sum()).sort_values(),
    "peak":(slicecounts_peak.sum() / slicecounts_all.sum()).sort_values()
}).T.style.bar()

# %%
enrichment["log_odds"] = np.log(enrichment["odds"]) 
# enrichment2["log_odds"] = np.log(enrichment2["odds"])
enrichment_peak["log_odds"] = np.log(enrichment_peak["odds"])

# %%
fig, ax = plt.subplots()
plt.scatter(enrichment["log_odds"], enrichment_peak["log_odds"])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axvline(0, color = "grey")
ax.axhline(0, color = "grey")
ax.set_aspect(1)

# %%
enrichment.query("q_value < 0.05").sort_values("odds", ascending = False).head(20)

# %%
# enrichment.xs(motifscan.motifs.loc[motifscan.motifs.index.str.contains("NEUROD1")].index[0], level="motif").sort_values("odds")

# %%
enrichment.loc[enrichment.index.get_level_values("motif").isin(motifscan.motifs.loc[motifscan.motifs["HUMAN_gene_symbol"].str.contains("RBPJ")].index)].sort_values("odds").style.bar()

# %%
enrichment.loc[enrichment.index.get_level_values("motif").str.contains("HEY")].sort_values("odds").style.bar()

# %%
# ct = "MAIT"
# ct = "naive B"
# ct = "memory B"
# ct = "CD14+ Monocytes"
# ct = "NK"
# ct = "Plasma"
ct = "Portal Hepatocyte"
# ct = "Cholangiocyte"
# ct = "Central Hepatocyte"
# ct = "Immune"
# ct = "B"
# ct = "Lymphoma"
# ct = "KC"
pd.concat([
    enrichment.loc[ct].query("q_value < 0.05").query("odds > 1").sort_values("odds", ascending=False).head(15),
    enrichment.loc[ct].query("q_value < 0.05").query("odds < 1").sort_values("odds", ascending=False).tail(15)
]).style.bar()

# %%
motifscan.motifs["symbol"] = motifscan.motifs["MOUSE_gene_symbol"]
# motifscan.motifs["symbol"] = motifscan.motifs["HUMAN_gene_symbol"]

# %%
# adata_raw = transcriptome.adata.raw.to_adata()
# sc.pp.normalize_total(adata_raw)
# sc.pp.log1p(adata_raw)
# sc.tl.rank_genes_groups(
#     adata_raw,
#     "cluster",
#     method="t-test",
#     max_iter=500,
# )
# sc.get.rank_genes_groups_df(adata_raw, None).rename(columns={"names": "gene", "scores": "score"}).query("pvals_adj < 0.05").set_index(["group", "gene"])["logfoldchanges"].unstack()

# %%
adata_raw = transcriptome.adata.raw.to_adata()
X = np.array(adata_raw.X.todense())
X = X / X.sum(1, keepdims=True) * X.sum(1).mean()
X = np.log(X + 1)
X = pd.DataFrame(X, index = adata_raw.obs.index, columns = adata_raw.var.index)
cluster_transcriptome = X.groupby(clustering.labels).mean()
diffexp = cluster_transcriptome - cluster_transcriptome.mean(0)

motifs_oi = motifscan.motifs.sort_values("quality").copy().reset_index().groupby("symbol").first().reset_index().set_index("motif")
motifs_oi["gene"] = adata_raw.var.reset_index().set_index("symbol").groupby("symbol").first().reindex(motifs_oi["symbol"])["gene"].values
motifs_oi = motifs_oi.dropna(subset=["gene"])
len(motifs_oi)

# %%
cluster_transcriptome = pd.DataFrame(transcriptome.layers["magic"][:], index = transcriptome.obs.index, columns = transcriptome.var.index).groupby(clustering.labels).mean()
diffexp = cluster_transcriptome - cluster_transcriptome.mean(0)

motifs_oi = motifscan.motifs.sort_values("quality").copy().reset_index().groupby("symbol").first().reset_index().set_index("motif")
motifs_oi["gene"] = [transcriptome.gene_id(symbol) if symbol in transcriptome.var["symbol"].tolist() else None for symbol in motifs_oi["symbol"]]
motifs_oi = motifs_oi.dropna(subset=["gene"])
len(motifs_oi)

# %%
import scipy.stats

def score_diffexp_enrichment(enrichment:pd.DataFrame, diffexp:pd.DataFrame, motifs_oi, fc_cutoff=1.2):
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
        subscore = pd.DataFrame(
            {
                "lfc": diffexp.loc[cluster, motifs_oi["gene"]],
                "odds": enrichment.loc[cluster].loc[motifs_oi.index]["odds"].values,
            }
        )
        subscore["logodds"] = np.log(subscore["odds"])
        subscore = subscore.dropna()
        subscore = subscore.query("abs(lfc) > log(@fc_cutoff)")

        contingency = (
            np.array(
                [
                    [
                        subscore.query("lfc > 0").query("logodds > 0").shape[0],
                        subscore.query("lfc > 0").query("logodds < 0").shape[0],
                    ],
                    [
                        subscore.query("lfc < 0").query("logodds > 0").shape[0],
                        subscore.query("lfc < 0").query("logodds < 0").shape[0],
                    ],
                ]
            )
            + 1
        )
        if len(subscore) > 4:
            odds = (contingency[1, 1] * contingency[0, 0] + 1) / (contingency[1, 0] * contingency[0, 1] + 1)

            if (subscore["lfc"].std() == 0) or (subscore["logodds"].std() == 0):
                cor = 0
                spearman = 0
            else:
                cor = np.corrcoef(subscore["lfc"], subscore["logodds"])[0, 1]
                spearman = scipy.stats.spearmanr(subscore["lfc"], subscore["logodds"])[0]
            log_avg_odds = np.concatenate(
                [subscore.query("lfc > 0")["logodds"], -subscore.query("lfc < 0")["logodds"]]
            ).mean()
        else:
            cor = 0
            spearman = 0
            odds = 1
            log_avg_odds = 0.0

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
                "avg_odds": np.exp(log_avg_odds),
            }
        )
    subscores = pd.concat(subscores).set_index(["cluster", diffexp.columns.name])
    if len(scores):
        scores = pd.DataFrame(scores).set_index("cluster")
    else:
        scores = pd.DataFrame(columns=["cluster", "odds", "log_odds", "cor"]).set_index("cluster")
    return scores, subscores


# %%
fc_cutoffs = np.linspace(1., 2, 20)

cluster_comparison = []

for fc_cutoff in fc_cutoffs:
    scores, _ = score_diffexp_enrichment(enrichment, diffexp, motifs_oi, fc_cutoff = fc_cutoff)
    cluster_comparison.append(scores.assign(method = "chd").assign(fc_cutoff = fc_cutoff).reset_index())
    scores_peak, _ = score_diffexp_enrichment(enrichment_peak, diffexp, motifs_oi, fc_cutoff = fc_cutoff)
    cluster_comparison.append(scores_peak.assign(method = "peak").assign(fc_cutoff = fc_cutoff).reset_index())
    # scores_peak = score_diffexp_enrichment(enrichment2, diffexp, motifs_oi, fc_cutoff = fc_cutoff)
    # cluster_comparison.append(scores_peak.assign(method = "chd2").assign(fc_cutoff = fc_cutoff).reset_index())

cluster_comparison = pd.concat(cluster_comparison, ignore_index = True).set_index(["method", "fc_cutoff", "cluster"])

# %%
comparison = cluster_comparison.groupby(["method", "fc_cutoff"])[["cor", "spearman", "log_odds", "log_avg_odds"]].mean()
comparison["odds"] = np.exp(comparison["log_odds"])
comparison["avg_odds"] = np.exp(comparison["log_avg_odds"])
comparison["contingency"] = cluster_comparison.groupby(["method", "fc_cutoff"])["contingency"].sum()

# %%
overall_contingency = np.stack(comparison["contingency"].values)
comparison["overall_odds"] = (overall_contingency[:, 1, 1] * overall_contingency[:, 0, 0] + 1) / (overall_contingency[:, 1, 0] * overall_contingency[:, 0, 1] + 1)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.8))
panel, ax = fig.main.add_right(chd.grid.Panel((1.5, 1.5)))
comparison["overall_odds"].unstack().T.plot(ax = ax)
ax.set_ylabel("overall_odds")
ax.axvline(1.2)
panel, ax = fig.main.add_right(chd.grid.Panel((1.5, 1.5)))
comparison["odds"].unstack().T.plot(ax = ax)
ax.set_ylabel("odds")
ax.axvline(1.2)
panel, ax = fig.main.add_right(chd.grid.Panel((1.5, 1.5)))
comparison["avg_odds"].unstack().T.plot(ax = ax)
ax.set_ylabel("avg_odds")
ax.axvline(1.2)
panel, ax = fig.main.add_right(chd.grid.Panel((1.5, 1.5)))
comparison["cor"].unstack().T.plot(ax = ax)
ax.set_ylabel("cor")
ax.axvline(1.2)
panel, ax = fig.main.add_right(chd.grid.Panel((1.5, 1.5)))
comparison["spearman"].unstack().T.plot(ax = ax)
ax.set_ylabel("spearman")
ax.axvline(1.2)
fig.plot()

# %%
fc_cutoff = 1.2

scores, subscores = score_diffexp_enrichment(enrichment, diffexp, motifs_oi, fc_cutoff = fc_cutoff)
clustering.cluster_info["n_cells"] = clustering.labels.value_counts()
scores["n_cells"] = clustering.cluster_info["n_cells"]

scores_peak, subscores_peak = score_diffexp_enrichment(enrichment_peak, diffexp, motifs_oi, fc_cutoff = fc_cutoff)

# %%
subscores_peak

# %%
subscores_joined = subscores.query("(lfc > log(@fc_cutoff))").join(subscores_peak, lsuffix = "_chd", rsuffix = "_peak")

# %%
transcriptome.adata.obs["cluster"] = clustering.labels

# %%
subscores_joined.query("(logodds_chd < 0) & (logodds_peak > 0)").sort_values("lfc_chd", ascending = False)["logodds_chd"].plot()
subscores_joined.query("(logodds_chd > 0) & (logodds_peak < 0)").sort_values("lfc_chd", ascending = False)["logodds_peak"].plot()

# %%
fig, ax = plt.subplots()
ax.scatter(subscores_joined["lfc_chd"], subscores_joined["logodds_chd"], s = 1)
ax.scatter(subscores_joined["lfc_peak"], subscores_joined["logodds_peak"], s = 1)

# %%
subscores_joined_oi = subscores_joined.query("(logodds_chd > 0) & (logodds_peak < 0)").sort_values("lfc_chd", ascending = False).iloc[:10]
# subscores_joined_oi = subscores_joined.query("(logodds_chd < 0) & (logodds_peak > 0)").sort_values("lfc_chd", ascending = False).iloc[:10]
sc.pl.umap(
    transcriptome.adata,
    color=["cluster", *subscores_joined_oi.index.get_level_values("gene")],
    size=50,
    legend_loc="on data",
    title=[
        "cluster",
        *(subscores_joined_oi.index.get_level_values(0) + ":" + transcriptome.symbol(subscores_joined_oi.index.get_level_values(1)))
    ],
)

# %%
scores.mean(), scores_peak.mean()

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width=0.2, padding_height = 0.))
import textwrap

main = fig.main.add_under(chd.grid.Grid(padding_width=0.2, margin_height=0.))
for cluster in clustering.cluster_info.index:
    panel, ax = main.add_right(chd.grid.Panel((.5, .5)))
    ax.text(0.5, 0., "\n".join(textwrap.wrap(cluster, width = 10)), ha="center", va="bottom", fontsize=8)
    ax.axis("off")

for method in ["chd", "peak"]:
    if method == "chd":
        scores_oi = scores
        cmap = "Blues"
    else:
        scores_oi = scores_peak
        cmap = "Reds"

    main = fig.main.add_under(chd.grid.Grid(padding_width=0.2))
    
    for cluster, contingency in scores_oi["contingency"].items():
        panel, ax = main.add_right(chd.grid.Panel((.5, .5)))
        ax.matshow(contingency.T, cmap=cmap, vmin=0, vmax=contingency.max())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.text(-0.4, -0.4, contingency[0, 0], ha="left", va="bottom", color="white")
        ax.text(-0.4, 1.4, contingency[0, 1], ha="left", va="top", color="black")
        ax.text(1.4, -0.4, contingency[1, 0], ha="right", va="bottom", color="black")
        ax.text(1.4, 1.4, contingency[1, 1], ha="right", va="top", color="white")
        odds = (contingency[1, 1] * contingency[0, 0] + 1) / (contingency[1, 0] * contingency[0, 1] + 1)
        ax.text(0.5, 0.5, f"{odds:.2f}", ha="center", va="center", color="black",
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
        )
    panel, ax = main.add_right(chd.grid.Panel((.5, .5)))
    ax.axis("off")
    ax.text(0.1, 0.5, "{:.2f}".format(np.exp(scores_oi["log_odds"].mean())), ha="left", va="center", fontsize=10)
fig.plot()

# %%
scores_all = pd.concat([
    scores.assign(method = "chd"),
    scores_peak.assign(method = "peak"),
]).reset_index().set_index(["method", "cluster"])
scores_all["odds"].unstack().plot()

# %%
enrichment_q_value_cutoff = 0.05
enrichment_odds_cutoff = 1.2
expression_fc_cutoff = 1.
n_top_motifs = 10

# %%
founds = []
for ct in tqdm.tqdm(clustering.cluster_info.index):
    motifs_oi = (
        enrichment.loc[ct]
        .query("q_value < @enrichment_q_value_cutoff")
        .query("odds > @enrichment_odds_cutoff")
        .sort_values("odds", ascending=False)
        .head(n_top_motifs)
        .index
    )
    genes_oi = diffexp.columns[(diffexp.idxmax() == ct) & (diffexp.max() > np.log(expression_fc_cutoff))]

    slicescores_foreground = slicescores.query("cluster == @ct").query("region in @genes_oi")
    slicescores_background = slicescores.query("cluster != @ct").query("region in @genes_oi")

    slicecounts_oi = slicecounts.loc[slicescores_foreground["slice"], motifs_oi]

    slicescores_foreground["slice_ix"] = slicecounts_oi.index.get_indexer(slicescores_foreground["slice"])

    print(f"{len(motifs_oi)=} {len(genes_oi)=} {ct=}")

    for gene_oi in genes_oi:
        slicescores_foreground2 = slicescores_foreground.loc[(slicescores_foreground["region"] == gene_oi)]
        found = slicecounts_oi.values[slicescores_foreground2["slice_ix"]]
        n_motifs_found = found.any(1)
        founds.append(
            {
                "gene": gene_oi,
                "found": found.sum(),
                "n_motifs_found": n_motifs_found.sum(),
                "ct": ct,
                "motifs": motifs_oi[slicecounts_oi.values[slicescores_foreground2["slice_ix"]].any(0)].tolist(),
            }
        )
founds = pd.DataFrame(founds).set_index(["ct", "gene"])

# %%
founds_peak = []
for ct in tqdm.tqdm(clustering.cluster_info.index):
    motifs_oi = (
        enrichment.loc[ct]
        .query("q_value < @enrichment_q_value_cutoff")
        .query("odds > @enrichment_odds_cutoff")
        .sort_values("odds", ascending=False)
        .head(n_top_motifs)
        .index
    )
    genes_oi = diffexp.columns[(diffexp.idxmax() == ct) & (diffexp.max() > np.log(expression_fc_cutoff))]

    slicescores_peak_foreground = slicescores_peak.query("cluster == @ct").query("region in @genes_oi")
    slicescores_peak_background = slicescores_peak.query("cluster != @ct").query("region in @genes_oi")

    slicecounts_peak_oi = slicecounts_peak.loc[slicescores_peak_foreground["slice"], motifs_oi]

    slicescores_peak_foreground["slice_ix"] = slicecounts_peak_oi.index.get_indexer(
        slicescores_peak_foreground["slice"]
    )

    print(f"{len(motifs_oi)=} {len(genes_oi)=} {ct=}")

    for gene_oi in genes_oi:
        slicescores_peak_foreground2 = slicescores_peak_foreground.loc[
            (slicescores_peak_foreground["region"] == gene_oi)
        ]
        found = slicecounts_peak_oi.values[slicescores_peak_foreground2["slice_ix"]]
        n_motifs_found = found.any(1)
        founds_peak.append(
            {
                "gene": gene_oi,
                "found": found.sum(),
                "n_motifs_found": n_motifs_found.sum(),
                "ct": ct,
                "motifs": motifs_oi[slicecounts_peak_oi.values[slicescores_peak_foreground2["slice_ix"]].any(0)].tolist(),
            }
        )
founds_peak = pd.DataFrame(founds_peak).set_index(["ct", "gene"])

# %%
founds["n_motifs_found"].mean(), founds_peak["n_motifs_found"].mean()

# %%
founds["detected"] = founds["found"] > 0
founds_peak["detected"] = founds_peak["found"] > 0

# %%
founds.join(founds_peak, rsuffix = "_peak").query("detected & ~detected_peak").sort_values("found", ascending = False)

# %%
scores = pd.DataFrame({
    "Peak differential":founds_peak.groupby("ct")["detected"].mean(),
    "ChromatinHD":founds.groupby("ct")["detected"].mean(),
})
scores = scores.loc[clustering.cluster_info.sort_values("n_cells", ascending = False).index]

fig, ax = plt.subplots()
scores.plot.bar(ax = ax)
ax.set_title("% of genes with at least one differentially accessible binding site", rotation = 0)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_xlabel("Cell type")
ax.set_ylim(0, 1)

# %%
fig, ax = plt.subplots(figsize = (1, 2))
scores.mean().plot.bar()
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

# %%
scores = pd.DataFrame({
    "Peak differential":founds_peak.groupby("ct")["n_motifs_found"].mean(),
    "ChromatinHD":founds.groupby("ct")["n_motifs_found"].mean(),
})
scores = scores.loc[clustering.cluster_info.sort_values("n_cells", ascending = False).index]

fig, ax = plt.subplots()
scores.plot.bar(ax = ax)
ax.set_title("Number of differentially accessible binding sites per gene", rotation = 0)
ax.set_xlabel("Cell type")

# %%
fig, ax = plt.subplots(figsize = (1, 2))
scores.mean().plot.bar()
