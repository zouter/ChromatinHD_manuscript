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
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_folder_original = chd.get_output() / "datasets" / "pbmc10k"
transcriptome_original = chd.data.Transcriptome(dataset_folder_original / "transcriptome")
fragments_original = chd.data.Fragments(dataset_folder_original / "fragments" / "10k10k")
fragments_original2 = chd.data.Fragments(dataset_folder_original / "fragments" / "100k100k")
fragments_original3 = chd.flow.Flow.from_path(dataset_folder_original / "fragments" / "500k500k")

# %% [markdown]
# ## Cell type subsets

# %%
transcriptome_original.obs["celltype"].value_counts()

# %%
group_a = ["CD4 naive T", "CD8 naive T", "CD4 memory T", "CD8 memory T", "NK", "MAIT"]
group_b = ["CD14+ Monocytes", "FCGR3A+ Monocytes", "cDCs"]

# %%
n_original_cells_a = transcriptome_original.obs["celltype"].isin(group_a).sum()
n_original_cells_b = transcriptome_original.obs["celltype"].isin(group_b).sum()

n_cells_to_select = min(n_original_cells_a, n_original_cells_b)
n_cells_to_select

# %%
rg = np.random.RandomState(1)
cells_oi_a = rg.choice(transcriptome_original.obs.index[transcriptome_original.obs["celltype"].isin(group_a)], n_cells_to_select, replace=False)
cells_oi_b = rg.choice(transcriptome_original.obs.index[transcriptome_original.obs["celltype"].isin(group_b)], n_cells_to_select, replace=False)
cells_oi_ab = rg.choice(transcriptome_original.obs.index[transcriptome_original.obs["celltype"].isin(group_a + group_b)], n_cells_to_select, replace=False)

# %%
import shutil

# %%
for dataset_name, cells_oi in zip(["mono_t_a", "mono_t_b", "mono_t_ab"], [cells_oi_a, cells_oi_b, cells_oi_ab]):
    dataset_folder = chd.flow.Flow(chd.get_output() / "datasets" / "pbmc10k" / "subsets" / dataset_name, reset = True)

    transcriptome = transcriptome_original.filter_cells(cells_oi, path = dataset_folder / "transcriptome")
    fragments = fragments_original.filter_cells(cells_oi, path = dataset_folder / "fragments" / "10k10k")
    fragments = fragments_original2.filter_cells(cells_oi, path = dataset_folder / "fragments" / "100k100k")
    folds = chd.data.folds.Folds(path = dataset_folder / "folds" / "5x5").sample_cells(fragments, 5, 5)

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata)

# %% [markdown]
# ## Subset top genes for ablation analysis

# %%
genes_oi = transcriptome_original.var.query("n_cells > 500").sort_values("dispersions_norm", ascending = False).index[:250]

# %%
dataset_folder = chd.flow.Flow(
    chd.get_output() / "datasets" / "pbmc10k" / "subsets" / "top250",
    # reset = True
)

transcriptome = transcriptome_original.filter_genes(genes_oi, path = dataset_folder / "transcriptome")
fragments_original = chd.data.Fragments(dataset_folder_original / "fragments" / "all")

# %%
import pickle
folder_data_preproc = chd.get_output() / "data" / "pbmc10k"
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]

# %%
for regions_name, window in [
    ["1k1k", [-1000, 1000]],
    ["2k2k", [-2000, 2000]],
    ["5k5k", [-5000, 5000]],
    # ["10k10k", [-10000, 10000]],
    # ["20k20k", [-20000, 20000]],
    # ["50k50k", [-50000, 50000]],
    # ["100k100k", [-100000, 100000]],
    # ["200k200k", [-200000, 200000]],
    # ["500k500k", [-500000, 500000]],
    # ["1m1m", [-1000000, 1000000]],
]:
    regions = chd.data.regions.Regions.from_transcripts(
        selected_transcripts, window, dataset_folder / "regions" / regions_name
    )
    fragments = chd.data.fragments.FragmentsView.from_fragments(
        fragments_original, regions, path = dataset_folder / "fragments" / regions_name, overwrite=True
    )
    
    fragments.create_regionxcell_indptr()

# %%
folds = chd.data.folds.Folds(path = dataset_folder / "folds" / "5x5").sample_cells(fragments, 5, 5)
folds = chd.data.folds.Folds(path = dataset_folder / "folds" / "5x1").sample_cells(fragments, 5, 1)

# %%
from_folder = chd.get_output() / "peaks" / "pbmc10k"
to_folder = chd.get_output() / "peaks" / "pbmc10k" / "subsets" / "top250"
to_folder.parent.mkdir(parents=True, exist_ok=True)
if not to_folder.exists():
    to_folder.symlink_to(from_folder)

# %%
from_folder = chd.get_output() / "datasets" / "pbmc10k" / "latent"
to_folder = chd.get_output() / "datasets" / "pbmc10k" / "subsets" / "top250" / "latent"
to_folder.parent.mkdir(parents=True, exist_ok=True)
if not to_folder.exists():
    to_folder.symlink_to(from_folder)

# %% [markdown]
# ## Subset 5 genes for test analysis

# %%
genes_oi = transcriptome_original.var.query("n_cells > 500").sort_values("dispersions_norm", ascending = False).index[:5]

# %%
dataset_folder = chd.flow.Flow(chd.get_output() / "datasets" / "pbmc10k" / "subsets" / "top1", reset = True)

transcriptome = transcriptome_original.filter_genes(genes_oi, path = dataset_folder / "transcriptome")

# %%
fragments = fragments_original.filter_regions(fragments_original.regions.filter(genes_oi, path = dataset_folder / "regions" / "10k10k"), path = dataset_folder / "fragments" / "10k10k")
fragments.create_regionxcell_indptr()

# %%
folds = chd.data.folds.Folds(path = dataset_folder / "folds" / "5x5").sample_cells(fragments, 5, 5)
folds = chd.data.folds.Folds(path = dataset_folder / "folds" / "5x1").sample_cells(fragments, 5, 1)

# %%
fragments = fragments_original2.filter_regions(fragments_original2.regions.filter(genes_oi, path = dataset_folder / "regions" / "100k100k"), path = dataset_folder / "fragments" / "100k100k")
fragments.create_regionxcell_indptr()

# %%
fragments.regions.coordinates

# %%
from_folder = chd.get_output() / "peaks" / "pbmc10k"
to_folder = chd.get_output() / "peaks" / "pbmc10k" / "subsets" / "top1"
to_folder.parent.mkdir(parents=True, exist_ok=True)
if not to_folder.exists():
    to_folder.symlink_to(from_folder)

# %%
from_folder = chd.get_output() / "datasets" / "pbmc10k" / "latent"
to_folder = chd.get_output() / "datasets" / "pbmc10k" / "subsets" / "top1" / "latent"
to_folder.parent.mkdir(parents=True, exist_ok=True)
if not to_folder.exists():
    to_folder.symlink_to(from_folder)

# %% [markdown]
# # Test

# %%
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / "pbmc10k" / "subsets" / "top250" / "fragments" / "500k500k")
fragments2 = chd.flow.Flow.from_path(chd.get_output() / "datasets" / "pbmc10k" / "subsets" / "top250" / "fragments" / "100k100k")

# %%
region_oi = transcriptome.gene_id("CCL4")
region_ix = fragments.var.index.get_loc(region_oi)

# %%
minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.obs.shape[0]), np.array([region_ix]))

# %%
regionxcell_ixs = (minibatch.regions_oi * fragments.n_cells + minibatch.cells_oi[:, None]).flatten()

indptrs = fragments.regionxcell_indptr[regionxcell_ixs]

coordinates_reader = fragments.parent.coordinates.open_reader()
coordinates = np.zeros((500000, 2), dtype = np.int64)

# %%
loader = chd.loaders.Fragments(fragments, 50000)
data = loader.load(minibatch)

# %%
loader2 = chd.loaders.Fragments(fragments2, 50000)
data2 = loader2.load(minibatch)

# %%
loader3 = chd.loaders.FragmentsRegional(fragments, 50000, region_oi)
data3 = loader.load(minibatch)

# %%
(
    torch.bincount(data.local_cellxregion_ix, minlength= minibatch.n_cells * minibatch.n_regions) >= 
    torch.bincount(data2.local_cellxregion_ix, minlength=minibatch.n_cells * minibatch.n_regions)
).all()

# %%
(
    torch.bincount(data3.local_cellxregion_ix, minlength= minibatch.n_cells * minibatch.n_regions) ==
    torch.bincount(data.local_cellxregion_ix, minlength=minibatch.n_cells * minibatch.n_regions)
).all()

# %%
plt.hist(data.coordinates[:, 0], bins = np.arange(*fragments.regions.window, step=10000), lw = 0, alpha = 0.5)
plt.hist(data2.coordinates[:, 0], bins = np.arange(*fragments2.regions.window, step=10000), lw = 0, alpha = 0.5)
plt.hist(data3.coordinates[:, 0], bins = np.arange(*fragments.regions.window, step=10000), lw = 0, alpha = 0.5)
""
