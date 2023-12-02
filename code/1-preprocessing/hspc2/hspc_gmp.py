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
dataset_name = "hspc"
folder_data_preproc_full = folder_data / dataset_name / "MV2"
genome = "GRCh38"

dataset_name = "hspc_gmp"
folder_dataset = chd.get_output() / "datasets" / dataset_name
folder_data_preproc = chd.get_output() / "data" / dataset_name
folder_data_preproc.mkdir(parents = True, exist_ok = True)

# %%
# !ln -s {folder_data_preproc_full}/atac_fragments.tsv.gz {folder_data_preproc}/atac_fragments.tsv.gz
# !ln -s {folder_data_preproc_full}/atac_fragments.tsv.gz.tbi {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")

# %%
adata = pickle.load(open(folder_data_preproc_full / "adata_gmp.pkl", "rb"))

# %%
sc.pp.highly_variable_genes(adata)
adata.var["means"] = adata.X.mean(axis = 0).A1

# %%
genes_oi = adata.var.query("means > 0.02").sort_values("dispersions_norm", ascending = False).index[:1000]
adata.var["highly_variable"] = False
adata.var.loc[genes_oi, "highly_variable"] = True

# %%
sc.pl.umap(adata, color = genes_oi[-10:])

# %%
fig, ax = plt.subplots()
ax.scatter(adata.var["means"], adata.var["dispersions_norm"], c = adata.var["highly_variable"])
ax.set_xscale("log")

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata)


# %%
def get_gene_id(symbols):
    return adata.var.reset_index().set_index("symbol").loc[symbols, "gene"].values


# %%
symbols = ["CD34", "AFF3", "MPO", "HBB", "VWF", "TAL1", "IKZF2", "RUNX1", "STOX2", "AZU1", "MKI67"]
sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

# %%
fig, ax = plt.subplots()
ax.scatter(adata.obsm["X_pca"][:, 0], sc.get.obs_df(adata, keys = get_gene_id(["MSI2"])[0]))

# %%
adata = adata[:, genes_oi]
pickle.dump(adata, open(folder_data_preproc / "adata.pkl", "wb"))

# %% [markdown]
# ## Trajectory

# %%
sc.tl.diffmap(adata)
adata.uns['iroot'] = np.argmax(adata.obsm["X_pca"][:, 0])
sc.tl.dpt(adata, n_branchings=0)

# %%
sc.pl.umap(adata, color = "dpt_pseudotime", title = "dpt_pseudotime")

# %%
import chromatinhd.data.gradient
gradient = chd.data.gradient.Gradient.from_values(adata.obs["dpt_pseudotime"], path = folder_dataset / "gradient")

# %% [markdown]
# ## TSS

# %%
adata = pickle.load(open(folder_data_preproc / "adata.pkl", "rb"))
genes_oi = adata.var.index

# %%
transcripts = pickle.load((folder_data_preproc_full / 'transcripts.pkl').open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(genes_oi)]

# %%
fragments_file = folder_data_preproc_full / "atac_fragments.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(transcripts, fragments_file)

# %%
pickle.dump(selected_transcripts, (folder_data_preproc / 'selected_transcripts.pkl').open("wb"))

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata.pkl").open("rb"))
adata = adata[:, adata.var.index[adata.var.highly_variable]]

# %% [markdown]
# ### Create transcriptome

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata, path = dataset_folder / "transcriptome")

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[adata.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
fragments_file = folder_data_preproc_full / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")
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
# ### 100k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[adata.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
fragments_file = folder_data_preproc_full / "atac_fragments.tsv.gz"
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
# ## Test extracting average expression per time point

# %%
bins = np.array([0] + list(np.linspace(0.1, 0.9, 19)) + [1])

x = gradient.values[:, 0]
y = transcriptome.layers["normalized"][:]

x_binned = np.clip(np.searchsorted(bins, x) - 1, 0, bins.size - 2)
x_onehot = np.zeros((x_binned.size, x_binned.max() + 1))
x_onehot[np.arange(x_binned.size), x_binned] = 1
y_binned = (x_onehot.T @ y) / x_onehot.sum(axis = 0)[:, None]
y_binned = (y_binned - y_binned.min(0)) / (y_binned.max(0) - y_binned.min(0))

# %%
sns.heatmap(y_binned[:, :100])

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.var.index[[100]], layer = "normalized")

# %%
import chromatinhd.loaders.transcriptome_fragments_time

# %%
loader = chromatinhd.loaders.transcriptome_fragments_time.TranscriptomeFragmentsTime(
    fragments = fragments,
    transcriptome=transcriptome,
    gradient=gradient,
    cellxregion_batch_size=10000,
    delta_time = 0.25
)
minibatch = chromatinhd.loaders.minibatches.Minibatch(np.arange(1000), np.arange(1))

result = loader.load(minibatch)

plt.scatter(
    gradient.values[result.minibatch.cells_oi, 0],
    result.transcriptome.value
)

loader = chromatinhd.loaders.transcriptome_fragments_time.TranscriptomeFragmentsTime(
    fragments = fragments,
    transcriptome=transcriptome,
    gradient=gradient,
    cellxregion_batch_size=10000,
    delta_time = 0.5
)
minibatch = chromatinhd.loaders.minibatches.Minibatch(np.arange(1000), np.arange(1))

result = loader.load(minibatch)

plt.scatter(
    gradient.values[result.minibatch.cells_oi, 0],
    result.transcriptome.value
)
