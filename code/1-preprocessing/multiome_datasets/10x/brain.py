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

# %% [markdown]
# # Preprocess

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

sns.set_style("ticks")

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "brain"; main_url = "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-arc/2.0.0/human_brain_3k/human_brain_3k"; genome = "GRCh38"; organism = "hs"
dataset_name = "alzheimer"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.1/Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote/Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote";  genome = "mm10"; organism = "mm"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

if not (folder_data_preproc / "genome").exists():
    genome_folder = chd.get_output() / "data" / "genomes" / genome
    (folder_data_preproc / "genome").symlink_to(genome_folder)

# %% [markdown]
# ## Download

# %% [markdown]
# For an overview on the output data format, see:
# https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/algorithms/overview

# %%
# ! echo mkdir -p {folder_data_preproc}
# ! echo mkdir -p {folder_data_preproc}/bam

# %%
to_download = [
    "filtered_feature_bc_matrix.h5",
    "atac_fragments.tsv.gz",
    "atac_fragments.tsv.gz.tbi",
    "atac_peaks.bed",
    "atac_peak_annotation.tsv",
]

for filename in to_download:
    if not (folder_data_preproc / filename).exists():
        # !wget {main_url}_{filename} -O {folder_data_preproc}/{filename}
# !cat {folder_data_preproc}/atac_peaks.bed | sed '/^#/d' > {folder_data_preproc}/peaks.tsv

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = sc.read_10x_h5(folder_data_preproc / "filtered_feature_bc_matrix.h5")

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

# %%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts=1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes=200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=100)
print(adata.var.shape[0])

# %%
transcripts = chd.biomart.get_transcripts(chd.biomart.Dataset.from_genome(genome), gene_ids=adata.var.index.unique())
pickle.dump(transcripts, (folder_data_preproc / 'transcripts.pkl').open("wb"))

# %%
# only retain genes that have at least one ensembl transcript
adata = adata[:, adata.var.index.isin(transcripts["ensembl_gene_id"])]

# %%
sc.external.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind="hist")

# %%
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
adata.raw = adata

# %%
sc.pp.normalize_total(adata, size_factor)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.highly_variable_genes(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X

# %%
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %%
import pickle
pickle.dump(adata, (folder_data_preproc / 'adata.pkl').open("wb"))

# %% [markdown]
# ## Interpret and subset

# %%
adata = pickle.load((folder_data_preproc / 'adata.pkl').open("rb"))

# %%
adata.obs

# %%
sc.tl.leiden(adata, resolution=0.1)
sc.pl.umap(adata, color="leiden")

# %%
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", key_added  = "wilcoxon")
# genes_oi = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups').sort_values("scores", ascending = False).groupby("group").head(2)["names"]

# %%
pickle.dump(adata, (folder_data_preproc / 'adata_annotated.pkl').open("wb"))

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / 'adata_annotated.pkl').open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / 'transcripts.pkl').open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(transcripts, fragments_file)

# %%
adata.var["means"].shape

# %%
np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max()).shape

# %%
plt.scatter(adata.var["means"], np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max())[adata.var.index])

# %%
pickle.dump(selected_transcripts, (folder_data_preproc / 'selected_transcripts.pkl').open("wb"))

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %% [markdown]
# ### Create transcriptome

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, adata.var.sort_values("dispersions_norm").tail(5000).index], path=dataset_folder / "transcriptome")

# %%
sc.pl.umap(adata, color = adata.var.index[(adata.var["symbol"] == "Fos")][0], use_raw = False, layer = "magic")

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")

fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    # overwrite = True
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite = True,
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### All

# %%
# import genomepy
# genomepy.install_genome("GRCh38", genomes_dir="/data/genome/")
fasta_file = "/data/genome/GRCh38/GRCh38.fa"
chromosomes_file = "/data/genome/GRCh38/GRCh38.fa.sizes"

regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = dataset_folder / "regions" / "all")

fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments_all = chd.data.Fragments.from_fragments_tsv(
    fragments_file = fragments_file,
    obs = transcriptome.obs,
    regions=regions,
    path=dataset_folder / "fragments" / "all",
    overwrite = True,
    batch_size = 10e7,
)
fragments_all.create_regionxcell_indptr()

# %% [markdown]
# ### 500k

# %%
transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")
fragments_all = chd.flow.Flow.from_path(dataset_folder / "fragments" / "all")

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.adata.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-500000, 500000], dataset_folder / "regions" / "500k500k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.fragments.FragmentsView.from_fragments(
    parent = fragments_all,
    regions = regions,
    overwrite = True,
    path=dataset_folder / "fragments" / "500k500k",
)

# %%
fragments.create_regionxcell_indptr()
