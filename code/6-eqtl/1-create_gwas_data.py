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
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

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

# %%
fragment_dataset_name = "pbmc10k_leiden_0.1"
dataset_name = fragment_dataset_name + "_gwas"

# %%
genotype_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %%
dataset_folder = chd.get_output() / "datasets" / "genotyped" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load data

# %%
transcriptome = chd.data.transcriptome.transcriptome.ClusterTranscriptome(genotype_data_folder / "transcriptome")

# %%
adata2 = transcriptome.adata

# %%
adata2.var["std"] = adata2.X.std(0)
adata2.var["mean"] = adata2.X.mean(0)

# %%
clusters_info = pd.read_pickle(chd.get_output() / "data" / fragment_dataset_name / "clusters_info.pkl") 
clusters_info["ix"] = np.arange(len(clusters_info))

# %%
vcf_file = genotype_data_folder /"final.bcf.gz"

# %%
import cyvcf2

# %%
vcf = cyvcf2.VCF(pathlib.Path(vcf_file))

# %%
donors_info = pd.read_csv(genotype_data_folder / "donors_info.csv", index_col = 0)

# set order to the order in the vcf
donors_info = donors_info.reset_index().set_index("old").loc[vcf.samples].reset_index().set_index("donor")

# %% [markdown]
# ## Select variants

# %%
variants_info = pd.read_pickle(genotype_data_folder / "variants_info.pkl")

# %%
folder_data_preproc = chd.get_output() / "data" / fragment_dataset_name
fragments = chd.data.fragments.ChunkedFragments(folder_data_preproc / "fragments")

# %%
qtl_name = "gwas_immune"

folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
folder_qtl.mkdir(exist_ok = True, parents=True)
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + qtl_name + ".pkl"))

# %%
window_size = 10**6

# %%
variants_info["gwas"] = variants_info["rsid"].isin(qtl_mapped["rsid"])
variants_info = variants_info.loc[variants_info["rsid"].isin(qtl_mapped["rsid"])].copy()

# %%
variants_info["gwas"].mean()

# %%
# # filter on variants not in LD
# lddb_file = pathlib.Path(chd.get_output() / "lddb.pkl")
# lddb = pickle.load(lddb_file.open("rb"))
# variants_info["main"] = [(rsid not in lddb) or (len(lddb[rsid]) == 0) for rsid in variants_info["rsid"]]
# variants_info = variants_info.loc[variants_info["main"]].copy()

# %%
variants_info["ix"] = np.arange(len(variants_info))

# %%
variants_info["position"] = variants_info["pos"] + fragments.chromosomes["position_start"].loc[variants_info["chr"]].values

# %%
len(variants_info)

# %% [markdown]
# ## Genotype

# %%
vcf = cyvcf2.VCF(pathlib.Path(vcf_file))

genotypes = []
genotypes_variant_mapping = []
imputed = []
for (variant_ix, chr, pos) in tqdm.tqdm(zip(variants_info["ix"].values, variants_info["chr"].values, variants_info["pos"].values)):
    window_str = chr + ":" + str(pos-1) + "-" + str(pos)
    variant = next(vcf(window_str))
    
    genotypes.append(variant.gt_types.copy())
    genotypes_variant_mapping.append(variant_ix)
    try:
        variant.INFO["IMPUTED"]
        imputed.append(True)
    except:
        imputed.append(False)
genotypes = np.stack(genotypes, 1)
genotypes[genotypes == 2] = 1
genotypes[genotypes == 3] = 2
genotypes = genotypes - 1

# %%
variants_info["imputed"] = imputed
variants_info["imputed"].mean()

# %%
import chromatinhd.data.genotype.genotype

# %%
genotype = chromatinhd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
genotype.genotypes = genotypes

assert "position" in variants_info.columns
genotype.variants_info = variants_info

# %%
# !ls {genotype.path}

# %% [markdown]
# ## Select genes

# %%
genes = pd.read_csv(genotype_data_folder / "genes.csv", index_col = 0)

# %%
genes["std"] = adata2.var["std"]
genes["mean"] = adata2.var["mean"]

# %%
genes = genes.query("(mean > 0.1) & (std > 1.0)")
len(genes)


# %%
def get_variants(window):
    variant_ids = []
    for variant in vcf(window):
        variant_ids.append(variant.ID)
    return variant_ids


# %%
gene_variants_mapping_all = {}
for chr, genes_oi_chromosome in tqdm.tqdm(genes.groupby("chr")):
    variants_info_chromosome = variants_info.query("chr == @chr")
    for gene, gene_oi in (genes_oi_chromosome.iterrows()):
        window = (gene_oi["tss"] - window_size, gene_oi["tss"] + window_size)
        variant_ixs = variants_info_chromosome["ix"].values[(variants_info_chromosome["pos"].values < window[1]) & (variants_info_chromosome["pos"].values > window[0])]
        
        gene_variants_mapping_all[gene_oi.name] = variant_ixs

# %%
genes_oi = [gene for gene, variants in gene_variants_mapping_all.items() if len(variants) > 0]

# %%
genes = genes.loc[genes_oi]
len(genes)

# %%
genes["tss_position"] = genes["tss"] + fragments.chromosomes["position_start"].loc[genes["chr"]].values

# %%
gene_variants_mapping = [gene_variants_mapping_all[gene] for gene in genes.index]

# %%
genes["ix"] = np.arange(len(genes))

# %%
genes.to_csv(dataset_folder / "genes.csv")

# %%
pickle.dump(gene_variants_mapping, (dataset_folder / "gene_variants_mapping.pkl").open("wb"))

# %% [markdown]
# ## Expression

# %%
donors_info["ix"] = np.arange(len(donors_info))

# %%
adata2.obs["cluster_ix"] = clusters_info["ix"][adata2.obs["cluster"]].values
adata2.obs["donor_ix"] = donors_info["ix"][adata2.obs["donor"]].values

# %%
import xarray as xr

# expression = pd.DataFrame(adata2.X, columns = adata2.var.index)
expression = pd.DataFrame(adata2.raw.X, columns = adata2.var.index)
expression.index = pd.MultiIndex.from_frame(adata2.obs[["cluster", "donor"]])
expression = xr.DataArray(expression).unstack()
expression.values[np.isnan(expression.values)] = 0. # NaNs are caused by the patient having no cells of a particular cell type
expression = expression.sel(gene = genes.index)
expression = expression.transpose("donor", "cluster", "gene")

# %%
assert (expression.coords["gene"] == genes.index).all()
assert (expression.coords["cluster"] == clusters_info.index).all()
assert (expression.coords["donor"] == donors_info.index).all()

# %%
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(dataset_folder / "transcriptome")
transcriptome.X = expression.values
transcriptome.donors_info = donors_info
transcriptome.clusters_info = clusters_info
transcriptome.var = genes

# %% [markdown]
# ## Fragments

# %%
folder_data_preproc = chd.get_output() / "data" / fragment_dataset_name
fragments_original = chd.data.fragments.ChunkedFragments(folder_data_preproc / "fragments")

# %%
if (dataset_folder/"fragments").exists():
    # !rm -r {dataset_folder}/fragments

# %%
(dataset_folder / "fragments").symlink_to(fragments_original.path)

# %%
fragments = chd.data.fragments.ChunkedFragments(dataset_folder / "fragments")

# %% [markdown]
# ## Check loader

# %%
fragments = chd.data.fragments.ChunkedFragments(dataset_folder / "fragments")
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(dataset_folder / "transcriptome")
genotype = chd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
gene_variants_mapping = pickle.load((dataset_folder / "gene_variants_mapping.pkl").open("rb"))

# %%
window_size = 5000

# %%
loader = chd.loaders.chunkfragments.ChunkFragments(
    fragments,
    genotype.variants_info["position"].values,
    window_size = window_size
)

# %%
self = loader

# %%
genes_oi = np.arange(1000) + 3000

# %%
gene_variantxgene_ix_mapping = []
i = 0
for variants in gene_variants_mapping:
    gene_variantxgene_ix_mapping.append(np.arange(i, i + len(variants)))
    i += len(variants)
n_variantxgenes = i

# %%
# this will map a variant_ix to a local_variant_ix
# initially all -1, given that the variant is not (yet) in the variants_oi
variant_ix_to_local_variant_ix = np.zeros(len(genotype.variants_info), dtype = int)
variant_ix_to_local_variant_ix[:] = -1

# this contains all variants that are selected, in order of their local_variant_ix
variants_oi = []

# this maps each variantxgene combination to the (local)_gene_ix
variantxgene_to_gene = []
variantxgene_to_local_gene = []

# contains the variantxgene_ixs
variantxgene_ixs = []

# this will map a local_variant_ix to a local_variantxgene_x
# e.g. if we have calculated something for all local variants, we can then easily reshape to have all variantxgene combinations
# variantxgene_to_gene can then be used to retrieve the exact gene to which this variantxgene maps
local_variant_to_local_variantxgene_selector = []

for local_gene_ix, gene_ix in enumerate(genes_oi):
    gene_variant_ixs = gene_variants_mapping[gene_ix]
    unknown_variant_ixs = gene_variant_ixs[variant_ix_to_local_variant_ix[gene_variant_ixs] == -1]
    variant_ix_to_local_variant_ix[unknown_variant_ixs] = np.arange(len(unknown_variant_ixs)) + len(variants_oi)
    
    variants_oi.extend(unknown_variant_ixs)
    
    local_variant_to_local_variantxgene_selector.extend(variant_ix_to_local_variant_ix[gene_variant_ixs])
    variantxgene_to_gene.extend([gene_ix] * len(gene_variant_ixs))
    variantxgene_to_local_gene.extend([local_gene_ix] * len(gene_variant_ixs))
    variantxgene_ixs.extend(gene_variantxgene_ix_mapping[gene_ix])
    
variants_oi = np.array(variants_oi)
local_variant_to_local_variantxgene_selector = np.array(local_variant_to_local_variantxgene_selector)
variantxgene_to_gene = np.array(variantxgene_to_gene)
variantxgene_to_local_gene = np.array(variantxgene_to_local_gene)
variantxgene_ixs = np.array(variantxgene_ixs)

# %%
relative_coordinates = []
cluster_ixs = []
variant_ixs = []
local_variant_ixs = []
local_cluster_ixs = []

for local_variant_ix, variant_ix in enumerate(variants_oi):
    position = self.variant_positions[variant_ix]
    upper_bound = self.variant_upper_bounds[variant_ix]
    lower_bound = self.variant_lower_bounds[variant_ix]
    
    window_start = max(lower_bound, position - self.window_size // 2)
    window_end = min(upper_bound, position + self.window_size // 2)

    window_chunks_start = window_start // self.chunk_size
    window_chunks_end = (window_end // self.chunk_size) + 1

    chunks_from, chunks_to = self.chunkcoords_indptr[window_chunks_start], self.chunkcoords_indptr[window_chunks_end]
    
    clusters_oi = self.clusters[chunks_from:chunks_to]
    
    # sorting is necessary here as the original data is not sorted
    # and because sorted data (acording to variantxcluster) will expected downstream by torch_scatter
    order = np.argsort(clusters_oi)
    
    relative_coordinates.append(
        (self.chunkcoords[chunks_from:chunks_to] * self.chunk_size + self.relcoords[chunks_from:chunks_to] - position)[order]
    )
    cluster_ixs.append(
        clusters_oi[order]
    )
    variant_ixs.append(np.repeat(variant_ix, chunks_to - chunks_from))
    local_variant_ixs.append(np.repeat(local_variant_ix, chunks_to - chunks_from))
relative_coordinates = np.hstack(relative_coordinates)
cluster_ixs = np.hstack(cluster_ixs)
variant_ixs = np.hstack(variant_ixs)
local_variant_ixs = np.hstack(local_variant_ixs)
local_cluster_ixs = cluster_ixs

# %%
n_variants = len(variants_oi)
n_clusters = len(fragments.clusters_info)
local_clusterxvariant_indptr = chd.utils.indices_to_indptr(local_cluster_ixs * n_variants + local_variant_ixs, n_variants * n_clusters)

# %%
local_clusterxvariant_indptr = chd.utils.indices_to_indptr(local_cluster_ixs * n_variants + local_variant_ixs, n_variants * n_clusters)

# %%
sns.histplot(relative_coordinates, bins = 100, lw = 0)

# %% [markdown]
# Expression

# %%
expression = transcriptome.X[..., genes_oi]

# %% [markdown]
# Genotype

# %%
genotypes = genotype.genotypes[:, variants_oi]

# %% [markdown]
# ### Data

# %%
# device = "cpu"
device = "cuda"

# %%
import torch
import dataclasses

# %%
cluster_cut_lib = self.cluster_cut_lib / 10**6

# %%
clusters_oi = fragments.clusters_info["ix"].values

# %%
data = chd.loaders.chunkfragments.Data(
    torch.from_numpy(relative_coordinates),
    torch.from_numpy(local_cluster_ixs),
    torch.from_numpy(local_clusterxvariant_indptr),
    cluster_cut_lib,
    torch.from_numpy(local_variant_to_local_variantxgene_selector),
    torch.from_numpy(variantxgene_to_gene),
    torch.from_numpy(variantxgene_to_local_gene),
    torch.from_numpy(variantxgene_ixs),
    torch.from_numpy(expression),
    torch.from_numpy(genotypes),
    window_size,
    variants_oi,
    genes_oi,
    clusters_oi
).to(device)

# %% [markdown]
# ## Train

# %%
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(dataset_folder / "transcriptome")
genotype = chd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
gene_variants_mapping = pickle.load((dataset_folder / "gene_variants_mapping.pkl").open("rb"))

# %%
import chromatinhd.models.eqtl.mapping.v2 as eqtl_model

# %%
loader = eqtl_model.Loader(transcriptome, genotype, gene_variants_mapping)

# %%
# genes_oi = np.array(np.arange(1000).tolist() + transcriptome.var.loc[transcriptome.gene_id(["CTLA4", "BACH2", "BLK"]), "ix"].tolist())
# genes_oi = np.array([transcriptome.var.loc[transcriptome.gene_id("CTLA4"), "ix"]])
genes_oi = transcriptome.var["ix"].values

# %%
minibatch = eqtl_model.Minibatch(genes_oi)

# %%
data = loader.load(minibatch)

# %%
model = eqtl_model.Model.create(transcriptome, genotype, gene_variants_mapping)
model_dummy = eqtl_model.Model.create(transcriptome, genotype, gene_variants_mapping, dummy = True)

# %%
loaders = chd.loaders.LoaderPool(
    eqtl_model.Loader,
    {"transcriptome":transcriptome, "genotype":genotype, "gene_variants_mapping":gene_variants_mapping},
    n_workers=10,
    shuffle_on_iter=False,
)

# %%
minibatches = eqtl_model.loader.create_bins_ordered(transcriptome.var["ix"].values)

# %%
minibatches_inference = minibatches

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

# %% [markdown]
# ## Inference & interpretion

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
import xarray as xr

# %%
fc_log_mu = xr.DataArray(model.fc_log_predictor.variantxgene_cluster_effect.weight.detach().cpu().numpy().T, coords = [transcriptome.clusters_info.index, variantxgene_index])

# %%
scores = fc_log_mu.to_pandas().T.stack().to_frame("fc_log")

# %%
scores["bf"] = bf.to_pandas().T.stack()

# %%
scores.sort_values("bf")

# %%
scores.query("cluster == 'CD4 T'").sort_values("bf").join(transcriptome.var[["symbol"]])

# %%
scores.groupby("cluster")["bf"].sum().to_frame("bf").style.bar()

# %%
"rs4987360" in genotype.variants_info["rsid"]

# %%
variant_id = genotype.variants_info.query("rsid == 'rs4987360'").index[0]

# %%
scores.join(genotype.variants_info[["rsid"]]).xs(variant_id, level = "variant").sort_values("fc_log")

# %%
