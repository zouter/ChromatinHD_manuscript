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

# %% [markdown]
# ## Download

# %%
import polyptich as pp
pp.setup_ipython()

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
dataset_name1 = "pbmc10k"
folder_data_preproc1 = chd.get_output() / "data" / dataset_name1
folder_data_preproc1.mkdir(exist_ok=True, parents=True)

dataset_name = "pbmc10k_int"
folder_data_preproc = chd.get_output() / "data" / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

genome = "GRCh38"
organism = "hs"

# %%
# # ! wget https://cf.10xgenomics.com/samples/cell-exp/4.0.0/Parent_NGSC3_DI_PBMC/Parent_NGSC3_DI_PBMC_filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/filtered_feature_bc_matrix.h5

if not (folder_data_preproc/"filtered_feature_bc_matrix.h5").exists():
    # ! wget https://cf.10xgenomics.com/samples/cell-exp/6.1.0/20k_PBMC_3p_HT_nextgem_Chromium_X/20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/filtered_feature_bc_matrix.h5

# %% [markdown]
# ## Process transcriptome data of cell reference

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

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
import pickle

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X

# %%
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %%
pickle.dump(adata, (folder_data_preproc / 'adata.pkl').open("wb"))

# %% [markdown]
# ## Merge

# %%
adata2 = pickle.load((folder_data_preproc1 / "adata_annotated.pkl").open("rb"))
folder_data_preproc1

# %%
import anndata as ad

# %%
adata_tot = adata.concatenate(adata2, batch_categories=["cell", "nucleus"])

# %%
adata_tot.X = adata_tot.layers["counts"]

# %%
sc.pp.normalize_per_cell(adata_tot)

# %%
sc.pp.log1p(adata_tot)
sc.pp.pca(adata_tot)
sc.external.pp.harmony_integrate(adata_tot, "batch", lamb = 0.1, nclust = 20)

# %%
sc.pp.neighbors(adata_tot, use_rep="X_pca_harmony")
sc.tl.umap(adata_tot)

# %%
sc.pl.umap(adata_tot, color = ["batch"])

# %%
import annoy

# %%
x_key = adata_tot.obsm["X_pca_harmony"][adata_tot.obs["batch"] == "cell", :]
x_query = adata_tot.obsm["X_pca_harmony"][adata_tot.obs["batch"] == "nucleus", :]

# %%
index = annoy.AnnoyIndex(x_key.shape[1], "euclidean")

for key in tqdm.tqdm(range(x_key.shape[0])):
    index.add_item(key, x_key[key])
index.build(10)

# %%
idxs = []
linked_normalized = []
linked_magic = []
for query in tqdm.tqdm(x_query):
    idx = index.get_nns_by_vector(query, 10)
    linked_normalized.append(np.array(adata.X[idx].mean(0)).flatten())
    linked_magic.append(np.array(adata.layers["magic"][idx].mean(0)).flatten())

# %%
adata3 = ad.AnnData(
    np.stack(linked_normalized),
    layers = {
        "normalized": np.stack(linked_normalized),
        "magic": np.stack(linked_magic),
    },
    obs = adata2.obs,
    var = adata.var,
)

# %%
# sc.pp.pca(adata3)
# sc.pp.neighbors(adata3)
# sc.tl.umap(adata3)
adata3.obsm["X_umap"] = adata2.obsm["X_umap"]

# %%
transcriptome = chd.data.Transcriptome.from_adata(adata3, path = folder_dataset / "transcriptome")

# %%
plt.hist(adata.var["means"], range = (0, 3), alpha = 0.5, bins = 100, label = "cell")
plt.hist(adata2.var["means"], range = (0, 3), alpha = 0.5, bins = 100, label = "nuc")
plt.legend()
""

# %%
fig, ax = plt.subplots(figsize = (2, 2))
ax.hist(adata.obs["n_counts"], range = (0, 20000))
ax.hist(adata2.obs["n_counts"], range = (0, 20000))

fig, ax = plt.subplots(figsize = (2, 2))
ax.hist(adata.obs["n_genes"], range = (0, 5000), label = "cell")
ax.hist(adata2.obs["n_genes"], range = (0, 5000), label = "nuc")
ax.legend()

# %%
genescores = pd.DataFrame({
    "mean_cell":adata.var["means"],
    "mean_nuc":adata2.var["means"],
    "n_cells_cell":adata.var["n_cells"] / adata.obs.shape[0],
    "n_cells_nuc":adata2.var["n_cells"] / adata2.obs.shape[0],
}).fillna(0.)
genescores["symbol"] = adata.var["symbol"]
genescores["diff"] = genescores["mean_cell"] - genescores["mean_nuc"]
genescores["diff"] = genescores["n_cells_cell"] - genescores["n_cells_nuc"]
genescores.sort_values("diff").head(50)
# genescores.sort_values("diff").query("diff > 0").head(20)

# %%
def plot_gene(gene_id):
    fig, axes = plt.subplots(2, 3, figsize = (3*3, 3*2))

    for layer, (ax0, ax1, ax2) in zip(["normalized", "magic"], axes):
        cmap = mpl.colormaps["viridis"]
        plotdata = pd.DataFrame({"UMAP1":adata2.obsm["X_umap"][:, 0], "UMAP2":adata2.obsm["X_umap"][:, 1], "expression":sc.get.obs_df(adata2, gene_id, layer = layer)})
        ax0.scatter("UMAP1", "UMAP2", c = "expression", data = plotdata, s = 1, cmap = cmap)
        ax0.set_title("Nuclear")

        plotdata = pd.DataFrame({"UMAP1":adata.obsm["X_umap"][:, 0], "UMAP2":adata.obsm["X_umap"][:, 1], "expression":sc.get.obs_df(adata, gene_id, layer = layer)})
        ax1.scatter("UMAP1", "UMAP2", c = "expression", data = plotdata, s = 1, cmap = cmap)
        ax1.set_title("Cell")

        plotdata = pd.DataFrame({"UMAP1":adata3.obsm["X_umap"][:, 0], "UMAP2":adata3.obsm["X_umap"][:, 1], "expression":sc.get.obs_df(adata3, gene_id, layer = layer)})
        ax2.scatter("UMAP1", "UMAP2", c = "expression", data = plotdata, s = 1, cmap = cmap)
        ax2.set_title("Integrated")
plot_gene(transcriptome.gene_id("SNX3"))

# %%
fig, ax = plt.subplots(figsize = (2, 2))
ax.axvline(0, color = "black")
sns.ecdfplot(genescores["diff"])
genescores["diff"].mean(), genescores["diff"].median()

# %% [markdown]
# ### Correlation

# %%
common_genes = list(set(adata.var.index) & set(adata2.var.index))
x_cell = np.array(adata3[:, common_genes].layers["magic"])
x_nuc = np.array(adata2[:, common_genes].layers["magic"])

cors = pd.Series(chd.utils.paircor(x_cell, x_nuc), index = common_genes)

# %%
fig, ax = plt.subplots(figsize = (2, 2))
ax.axvline(0, color = "black")
sns.ecdfplot(cors)
ax.axvline(cors.mean())

# %%
genescores["cor"] = cors

# %%
sc.pp.highly_variable_genes(adata3)

# %%
genescores["dispersions_norm_cell"] = adata3.var["dispersions_norm"]

# %%
gene_id = cors.sort_values().index[-10000]
gene_id = adata3.var.sort_values("dispersions_norm", ascending = False).head(1000).index[-20]
plot_gene(gene_id)
genescores.loc[gene_id]

# %%
pickle.dump(adata3, (folder_data_preproc / "adata_annotated.pkl").open("wb"))

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / 'adata_annotated.pkl').open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / 'transcripts.pkl').open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc1 / "atac_fragments.tsv.gz"
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
dataset_folder = chd.get_output() / "datasets" / "pbmc10k_int"
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %% [markdown]
# ### Create transcriptome

# %%
adata = adata[:, adata.var.sort_values("dispersions_norm").tail(5000).index]

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata, path=dataset_folder / "transcriptome")

# %%
sc.pl.umap(adata, color = adata.var.index[(adata.var["symbol"] == "JUNB")][0])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[adata.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
fragments_file = folder_data_preproc1 / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
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
fragments_file = folder_data_preproc1 / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    batch_size = 1000000
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 500k

# %%
transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[adata.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-500000, 500000], dataset_folder / "regions" / "500k500k"
)

# %%
fragments_file = folder_data_preproc1 / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "500k500k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    batch_size = 10000000
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Link peaks

# %%
# !ln -s {folder_data_preproc1}/atac_fragments.tsv.gz {folder_data_preproc}/atac_fragments.tsv.gz
# !ln -s {folder_data_preproc1}/atac_fragments.tsv.gz.tbi {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %%
# !ln -s {chd.get_output()}/peaks/{dataset_name1} {chd.get_output()}/peaks/{dataset_name}
