# ---
# jupyter:
#   jupytext:
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

dataset_name = "hepatocytes"
genome = "mm10"
organism = "mm"

folder_data_preproc = folder_data / "liver"
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
adata.obs["cell_original"] = adata.obs.index
adata = adata[adata.obs["celltype"].isin(["Portal Hepatocyte", "Mid Hepatocyte", "Central Hepatocyte"])].copy()
adata.obs.index = adata.obs.index.str.split("-").str[0] + "-1"

# %%
sc.pp.neighbors(adata, n_neighbors=10)
sc.tl.umap(adata)

# %%
sc.pp.pca(adata)

# %%
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.packages

# %%
# ro.r("install.packages('princurve')")

# %%
data = adata.obsm["X_pca"][:, [0, 1]]
princurve = ro.packages.importr("princurve", on_conflict="warn")

# %%
curve = princurve.principal_curve(data, stretch = 10.)

# %%
plt.hist(curve[2])

# %%
pcurve = pd.DataFrame({
    "ix":np.arange(len(curve[0])),
    "order":curve[1]-1,
})
pcurve["x"] = adata.obsm["X_pca"][pcurve["ix"], 0]
pcurve["y"] = adata.obsm["X_pca"][pcurve["ix"], 1]
pcurve["time"] = curve[2]
pcurve["time"] = np.clip((pcurve["time"] - pcurve["time"].quantile(0.01)) / (pcurve["time"].quantile(0.99) - pcurve["time"].quantile(0.01)), 0, 1)

# %%
fig, ax = plt.subplots()
mp = ax.scatter(pcurve["x"], pcurve["y"], c = pcurve["time"], cmap = "viridis")
fig.colorbar(mp)
# ax.scatter(data[:, 0], data[:, 1], c = projection[2])
# ax.scatter(curve[0][:, 0], curve[0][:, 1])

# %%
adata.obs["celltype"] = pd.Categorical(pd.cut(pcurve["time"], np.linspace(-0.0001, 1, 6), labels = ["1", "2", "3", "4", "5"]).values)

# %%
sc.pl.pca(adata, color = ["celltype", "leiden"])

# %%
sc.pp.highly_variable_genes(adata)

# %% [markdown]
# ### Create transcriptome

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, adata.var.query("means > 0.1").sort_values("dispersions_norm", ascending = False).head(2000).index], path=dataset_folder / "transcriptome")

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
    overwrite = True
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

# %%

# %%
