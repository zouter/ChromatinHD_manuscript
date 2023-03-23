# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests
import xarray as xr

# %%
fragment_dataset_name = "pbmc10k_leiden_0.1"
dataset_name = fragment_dataset_name + "_gwas"

# %%
genotype_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %%
dataset_folder = chd.get_output() / "datasets" / "genotyped" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Direct eQTL model

# %% [markdown]
# ### Data

# %%
original_transcriptome = chd.data.transcriptome.transcriptome.Transcriptome(chd.get_output() / "data" / "pbmc10k" / "transcriptome")
original_latent = pd.read_pickle(chd.get_output() / "data" / "pbmc10k" / "latent" / "leiden_0.1.pkl")

# %%
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(dataset_folder / "transcriptome")
genes = pd.read_csv(dataset_folder / "genes.csv", index_col = 0)
fragments = chd.data.fragments.ChunkedFragments(dataset_folder / "fragments")
genotype = chd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
gene_variants_mapping = pickle.load((dataset_folder / "gene_variants_mapping.pkl").open("rb"))

# %%
# filter on # of fragments
window_size = 5000

variant_positions = genotype.variants_info["position"].values
chunk_size = fragments.chunk_size

# get cut site coordinates
variant_counts = []

for variant_ix in genotype.variants_info["ix"]:
    position = variant_positions[variant_ix]

    window_start = position - window_size // 2
    window_end = position + window_size // 2

    window_chunks_start = window_start // chunk_size
    window_chunks_end = (window_end // chunk_size) + 1

    chunks_from, chunks_to = (
        fragments.chunkcoords_indptr[window_chunks_start],
        fragments.chunkcoords_indptr[window_chunks_end],
    )

    clusters_oi = fragments.clusters[chunks_from:chunks_to]
    
    variant_counts.append(np.bincount(clusters_oi, minlength = len(fragments.clusters_info)))
variant_counts = np.vstack(variant_counts)
variants_oi = (variant_counts.sum(1) > 200)
variants_oi.mean()

# %%
gene_variants_mapping = [variant_ixs[variants_oi[variant_ixs]] for variant_ixs in gene_variants_mapping]

# %%
import chromatinhd.models.eqtl.mapping.v2 as eqtl_model

# %%
loaders = chd.loaders.LoaderPool(
    eqtl_model.Loader,
    {"transcriptome":transcriptome, "genotype":genotype, "gene_variants_mapping":gene_variants_mapping},
    n_workers=10,
    shuffle_on_iter=False,
)

# %%
minibatches = eqtl_model.create_bins_ordered(transcriptome.var["ix"].values)

# %% [markdown]
# ### Model

# %%
model = eqtl_model.Model.create(transcriptome, genotype, gene_variants_mapping)
model_dummy = eqtl_model.Model.create(transcriptome, genotype, gene_variants_mapping, dummy = True)

# %% [markdown]
# ### Test loader

# %%
# genes_oi = np.array(np.arange(1000).tolist() + transcriptome.var.loc[transcriptome.gene_id(["CTLA4", "BACH2", "BLK"]), "ix"].tolist())
# # genes_oi = np.array([transcriptome.var.loc[transcriptome.gene_id("CTLA4"), "ix"]])

# minibatch = eqtl_model.Minibatch(genes_oi)

# data = loader.load(minibatch)

# %% [markdown]
# ### Train

# %%
loaders.initialize(minibatches)

# %%
optim = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-2)
trainer = eqtl_model.Trainer(model, loaders, optim, checkpoint_every_epoch=50, n_epochs = 300)
trainer.train()

# %%
optim = chd.optim.SparseDenseAdam(model_dummy.parameters_sparse(), model_dummy.parameters_dense(), lr = 1e-2)
trainer = eqtl_model.Trainer(model_dummy, loaders, optim, checkpoint_every_epoch=50, n_epochs = 300)
trainer.train()

# %%
chd.save(model, pathlib.Path("model_direct.pkl").open("wb"))
chd.save(model_dummy, pathlib.Path("model_dummy.pkl").open("wb"))

# %% [markdown]
# ### Inference & interpretion

# %%
model = chd.load(pathlib.Path("model_direct.pkl").open("rb"))
model_dummy = chd.load(pathlib.Path("model_dummy.pkl").open("rb"))

# %%
minibatches_inference = minibatches

# %%
loaders_inference = chd.loaders.LoaderPool(
    eqtl_model.Loader,
    {"transcriptome":transcriptome, "genotype":genotype, "gene_variants_mapping":gene_variants_mapping},
    n_workers=5,
    shuffle_on_iter=False,
)

# %%
variantxgene_index = []
for gene, gene_ix in zip(transcriptome.var.index, transcriptome.var["ix"]):
    variantxgene_index.extend([[gene, genotype.variants_info.index[variant_ix]] for variant_ix in gene_variants_mapping[gene_ix]])
variantxgene_index = pd.MultiIndex.from_frame(pd.DataFrame(variantxgene_index, columns = ["gene", "variant"]))

# %%
device = "cpu"

# %%
loaders_inference.initialize(minibatches_inference)
elbo = np.zeros((len(transcriptome.clusters_info), len(variantxgene_index)))
elbo_dummy = np.zeros((len(transcriptome.clusters_info), len(variantxgene_index)))

model = model.to(device)
model_dummy = model_dummy.to(device)
for data in tqdm.tqdm(loaders_inference):
    data = data.to(device)
    model.forward(data)
    elbo_mb = model.get_full_elbo().sum(0).detach().cpu().numpy()
    elbo[:, data.variantxgene_ixs.cpu().numpy()] += elbo_mb
    
    model_dummy.forward(data)
    elbo_mb = model_dummy.get_full_elbo().sum(0).detach().cpu().numpy()
    elbo_dummy[:, data.variantxgene_ixs.cpu().numpy()] += elbo_mb
    
    loaders_inference.submit_next()
    
bf = xr.DataArray(elbo_dummy - elbo, coords = [transcriptome.clusters_info.index, variantxgene_index])

# %%
fc_log_mu = xr.DataArray(model.fc_log_predictor.variantxgene_cluster_effect.weight.detach().cpu().numpy().T, coords = [transcriptome.clusters_info.index, variantxgene_index])

# %%
scores = fc_log_mu.to_pandas().T.stack().to_frame("fc_log")

# %%
scores["bf"] = bf.to_pandas().T.stack()

# %%
scores.to_pickle("scores.pkl")

# %%
scores.query("cluster == 'pDCs'").sort_values("bf").join(transcriptome.var[["symbol"]])

# %%
scores["significant"] = scores["bf"] > np.log(10)

# %%
scores.groupby("gene")["significant"].sum().sort_values(ascending = False).to_frame().join(transcriptome.var[["symbol", "chr"]]).head(50)

# %%
scores.groupby("cluster")["bf"].sum().to_frame("bf").style.bar()

# %%
variant_id = genotype.variants_info.query("rsid == 'rs3087243'").index[0]

# %%
scores.join(genotype.variants_info[["rsid"]]).xs(variant_id, level = "variant").sort_values("fc_log")

# %%
gene_id = transcriptome.gene_id("SRSF5")
variant_id = "chr14:68793871:C:T"

# %%
scores.join(genotype.variants_info[["rsid"]]).xs(variant_id, level = "variant").sort_values("bf")

# %% [markdown]
# ### Calculate correlation

# %%
# filter on # of fragments
window_size = 2000

variant_positions = genotype.variants_info["position"].values
chunk_size = fragments.chunk_size

# get cut site coordinates
variant_counts = []

for variant_ix in genotype.variants_info["ix"]:
    position = variant_positions[variant_ix]

    window_start = position - window_size // 2
    window_end = position + window_size // 2

    window_chunks_start = window_start // chunk_size
    window_chunks_end = (window_end // chunk_size) + 1

    chunks_from, chunks_to = (
        fragments.chunkcoords_indptr[window_chunks_start],
        fragments.chunkcoords_indptr[window_chunks_end],
    )

    clusters_oi = fragments.clusters[chunks_from:chunks_to]
    
    variant_counts.append(np.bincount(clusters_oi, minlength = len(fragments.clusters_info)))
variant_counts = np.vstack(variant_counts).T
variants_oi = (variant_counts.sum(0) > 500)
variants_oi.mean()

# %%
variant_expression = np.log1p(variant_counts / variant_counts.sum(1, keepdims = True))

# %% [markdown]
# ### Get reference variantxgene effect

# %%
scores["abs_fc_log"] = np.abs(scores["fc_log"])

# %%
variantxgene_effect_sorted = scores.query("significant").sort_values("abs_fc_log", ascending = False).groupby(["gene", "variant"]).first()["fc_log"]
variantxgene_effect_max = scores.sort_values("bf", ascending = False).groupby(["gene", "variant"])["fc_log"].first()

# %%
variantxgene_effect = pd.Series(np.nan, variantxgene_index)
variantxgene_effect[variantxgene_effect_sorted.index] = variantxgene_effect_sorted

missing_reference_fc = variantxgene_effect.index[np.isnan(variantxgene_effect)]
variantxgene_effect[missing_reference_fc] = variantxgene_effect_max[missing_reference_fc]

# %%
chd.save(variantxgene_effect, pathlib.Path("variantxgene_effect.pkl").open("wb"))
chd.save(model.fc_log_predictor.variantxgene_cluster_effect.weight.T.detach(), pathlib.Path("ground_truth_variantxgene_effect.pkl").open("wb"))
chd.save(gene_variants_mapping, pathlib.Path("gene_variants_mapping.pkl").open("wb"))

# %%
chosen = scores.groupby(["gene", "variant"])["significant"].any().to_frame().join(transcriptome.var[["ix"]].rename(columns = {"ix":"gene_ix"})).join(genotype.variants_info[["ix"]].rename(columns = {"ix":"variant_ix"}))
chosen = chosen.loc[variantxgene_index]
chosen["gene_ix"] = pd.Categorical(chosen["gene_ix"], categories=transcriptome.var["ix"])
chosen_variantxgene = chosen["significant"].values
chosen["new_ix"] = np.cumsum(chosen["significant"])
chosen["ix"] = np.arange(len(chosen))

# %%
scores["significant"] = scores["bf"] > np.log(10)
ground_truth_significant = scores["significant"].unstack().loc[variantxgene_index].values.T
ground_truth_bf = scores["bf"].unstack().loc[variantxgene_index].values.T

# %%
ground_truth_variantxgene_effect = model.fc_log_predictor.variantxgene_cluster_effect.weight.T.detach().numpy()


# %%
def paircor(x, y, dim=0, eps=1e-10):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor


# %%
ground_truth_variantxgene_relative = ground_truth_variantxgene_effect/variantxgene_effect.values

# %%
variantxgene_cors = paircor(ground_truth_variantxgene_relative, variant_expression[:,chosen["variant_ix"]])

# %%
chosen["cor"] = variantxgene_cors

# %%
chosen.groupby("significant")["cor"].mean()

# %%
symbol = "BACH2"
gene_id = transcriptome.gene_id(symbol)
gene_ix = transcriptome.gene_ix(symbol)

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (6, 3))
sns.heatmap(ground_truth_variantxgene_effect[:, chosen.loc[(gene_id), "ix"].values], ax = ax0)
sns.heatmap(variant_expression[:, chosen["variant_ix"]][:, chosen.loc[(gene_id), "ix"].values], ax = ax1)

# %%
# chosen["chosen"] = (chosen["cor"] > 0.2) & (chosen["significant"])
chosen["chosen"] = (chosen["significant"])
chosen_variantxgene = chosen.query("chosen")

# %%
new_gene_variants_mapping = []
for gene_ix, variants in chosen_variantxgene.sort_values(["gene_ix", "variant_ix"]).groupby("gene_ix")["variant_ix"]:
    new_gene_variants_mapping.append(variants.values)

# %%
chd.save(variantxgene_effect[chosen["chosen"]], pathlib.Path("variantxgene_effect.pkl").open("wb"))
chd.save(model.fc_log_predictor.variantxgene_cluster_effect.weight.T.detach()[:, chosen["chosen"]], pathlib.Path("ground_truth_variantxgene_effect.pkl").open("wb"))
chd.save(new_gene_variants_mapping, pathlib.Path("gene_variants_mapping.pkl").open("wb"))

# %% [markdown]
# ## Checkout fragment distribution around SNPs

# %%
import pyBigWig
bw = pyBigWig.open(str(chd.get_output() / "data" / "pbmc10k" / "atac_cut_sites.bigwig"))

# %%
window = 10000

# %%
bins = np.linspace(-window, window, 10)

# %%
vals = []
tops = []
for variant_id in chosen.query("chosen").index.get_level_values("variant").unique():
    variant_info = genotype.variants_info.loc[variant_id]
    values = np.array(bw.values(variant_info["chr"], variant_info["start"] - window, variant_info["start"] + window))
    tops.append(np.argmax(np.bincount(
        np.digitize(np.linspace(-window, window, len(values)), bins),
        weights = values
    )))
    vals.append(values)
vals = np.stack(vals)
vals = (vals - vals.mean(1, keepdims = True)) / vals.std(1, keepdims = True)

# %%
sns.histplot(tops)

# %%
sns.heatmap(vals)

# %%
sns.histplot(np.argmax(vals, axis = 1))

# %%
variant_positions = genotype.variants_info["position"].values

# %%
import torch

# %%
relative_coordinates = []
cluster_ixs = []
variant_ixs = []
local_variant_ixs = []
counts = []

tops = []

# for variant_ix in genotype.variants_info.loc[chosen.index.get_level_values("variant").unique()]["ix"]:
for variant_ix in genotype.variants_info["ix"]:
    position = variant_positions[variant_ix]

    window_start = position - window
    window_end = position + window

    window_chunks_start = window_start // fragments.chunk_size
    window_chunks_end = (window_end // fragments.chunk_size) + 1

    chunks_from, chunks_to = (
        fragments.chunkcoords_indptr[window_chunks_start],
        fragments.chunkcoords_indptr[window_chunks_end],
    )

    clusters_oi = fragments.clusters[chunks_from:chunks_to]

    # sorting is necessary here as the original data is not sorted
    # and because sorted data (acording to variantxcluster) will be expected downstream by torch_scatter
    # this is probably a major slow down
    # it might be faster to not sort here, and use torch_scatter scatter operations downstream
    order = np.argsort(clusters_oi)
    
    coord = (            fragments.chunkcoords[chunks_from:chunks_to] * fragments.chunk_size
            + fragments.relcoords[chunks_from:chunks_to]
            - position)

    relative_coordinates.append(coord[order])
    cluster_ixs.append(clusters_oi[order])
    variant_ixs.append(np.repeat(variant_ix, chunks_to - chunks_from))
    local_variant_ixs.append(
        np.repeat(local_variant_ix, chunks_to - chunks_from)
    )
    tops.append(np.argmax(np.bincount(np.digitize(coord, bins), minlength = len(bins)-1)))
    
    counts.append(np.bincount(clusters_oi, minlength = len(fragments.clusters_info)))
relative_coordinates = np.hstack(relative_coordinates)
cluster_ixs = np.hstack(cluster_ixs)
variant_ixs = np.hstack(variant_ixs)
local_variant_ixs = np.hstack(local_variant_ixs)
local_cluster_ixs = cluster_ixs

counts = np.vstack(counts)

# %%
sns.histplot(tops)

# %%
import tabix

# %%
tabix_file = chd.get_output() / "data" / "pbmc10k" / "atac_fragments.tsv.gz"
tabix_ = tabix.open(str(tabix_file))

# %%
tops = []
for _, variant_info in genotype.variants_info.loc[chosen.index.get_level_values("variant").unique()].iterrows():
    coordinates = []
    fragments_ = list(tabix_.query(variant_info["chr"], variant_info["start"] - window, variant_info["start"] + window))
    coordinates.extend([int(fragment[1]) for fragment in fragments_])
    coordinates.extend([int(fragment[2]) for fragment in fragments_])
    coordinates = np.array(coordinates) - variant_info["start"]
    tops.append(np.argmax(np.bincount(np.digitize(coordinates, bins))))

# %%
sns.histplot(tops)

# %%
celltype_1 = "CD4 T"
celltype_2 = "Plasma"

cluster_ix1 = transcriptome.clusters_info.loc[celltype_1, "ix"]
cluster_ix2 = transcriptome.clusters_info.loc[celltype_2, "ix"]

plotdata = pd.concat([
    scores.xs(celltype_1, level = "cluster").loc[variantxgene_index].rename(columns = lambda x:x+"1"),
    scores.xs(celltype_2, level = "cluster").loc[variantxgene_index].rename(columns = lambda x:x+"2"),
    
], axis = 1)

# %%
fragments_exp = np.log1p((counts / counts.mean(0, keepdims = True)).T[:, genotype.variants_info.loc[significant1.index.get_level_values("variant"), "ix"]])
transcriptome_exp = np.log1p((transcriptome.X).sum(0) / (transcriptome.X).sum(0).sum(1, keepdims = True) * 10**6)[:, transcriptome.var.loc[significant1.index.get_level_values("gene"), "ix"]]

# %%
paircor(transcriptome_exp, fragments_exp).mean()

# %%
plotdata["fragments_1"] = fragments_exp[cluster_ix1]
plotdata["fragments_2"] = fragments_exp[cluster_ix2]

plotdata["transcriptome_1"] = transcriptome_exp[cluster_ix1]
plotdata["transcriptome_2"] = transcriptome_exp[cluster_ix2]

# %%
plotdata["ratio"] = (plotdata["fragments_1"] / plotdata["fragments_2"])
plotdata["diff"] = (np.log1p(plotdata["fragments_1"]) - np.log1p(plotdata["fragments_2"]))

# %%
plotdata["significant_any"] = plotdata["significant1"] | plotdata["significant2"]
plotdata["significant_both"] = plotdata["significant1"] & plotdata["significant2"]

# %%
plotdata["protein_coding"] = plotdata.index.get_level_values("gene").isin(transcriptome.var.query("biotype == 'protein_coding'").index)

# %%
plotdata["group"] = plotdata["significant1"].astype(str) + plotdata["significant2"].astype(str)

# %%
plotdata["oi"] = plotdata["significant_any"]
plotdata["oi"] = plotdata["significant_any"]

# %%
plotdata.loc["ENSG00000151702"].query("significant1")

# %%
plotdata.loc["ENSG00000151702"].query("significant2")

# %%
import sklearn.linear_model
import sklearn.preprocessing

# %%
plotdata_oi = plotdata.query("significant_any")

# %%
X = plotdata[["fragments_1", "fragments_2", "transcriptome_1", "transcriptome_2"]].copy()
# X["fragments_1/transcriptome_1"] = X["fragments_1"]-X["transcriptome_1"]
# X["fragments_2/transcriptome_2"] = X["fragments_2"]-X["transcriptome_2"]
# X["full_ratio"] = X["fragments_1/transcriptome_1"]/X["fragments_2/transcriptome_2"]
# X = pd.DataFrame(sklearn.preprocessing.PolynomialFeatures(interaction_only=True,include_bias = False).fit_transform(X), columns = (X.columns.tolist() + ["_".join(x) for x in list(itertools.combinations(X.columns, 2))]))
X = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(X), columns = X.columns)

# y = plotdata["abs_fc_log1"]-plotdata["abs_fc_log2"]
# y = plotdata["significant1"].astype(int)-plotdata["significant2"].astype(int)
y = plotdata["bf1"].astype(float)-plotdata["bf2"].astype(float)

# y = sklearn.preprocessing.StandardScaler().fit_transform(y.values[:, None])[:, 0]

# %%
lm = sklearn.linear_model.LinearRegression().fit(X, y)
pd.Series(lm.coef_, X.columns)

# %%
sns.scatterplot(data = plotdata_oi, x = "fragments_1", y = "bf1")
sns.scatterplot(data = plotdata_oi, x = "fragments_2", y = "bf1")

# %%
