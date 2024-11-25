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
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "lsecs_kia"
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
adata = adata[adata.obs["celltype"].isin(["LSEC"])].copy()
adata.obs.index = adata.obs.index.str.split("-").str[0] + "-1"

# %%
folder_data_preproc2 = folder_data / "liverkia" / "liver_control_JVG28"
adata2 = pickle.load((folder_data_preproc2 / "adata_annotated.pkl").open("rb"))
adata2 = adata2[adata2.obs["leiden"].isin(["0"])].copy()

# %%
adata = sc.AnnData.concatenate(adata, adata2, batch_categories = ["liveraerts", "liverkia"])

# %%
adata.var["symbol"] = adata.var["symbol-liveraerts"]

# %%
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color = ["celltype", "leiden", "batch"])
sc.pl.umap(adata, color = adata.var.query("symbol in ['Dll4', 'Vwf', 'Lyve1', 'Wnt2']").index)

# %%
import eyck
eyck.modalities.transcriptome.plot_umap(adata, ["leiden", "Stab2", "Lyve1", "Wnt2", "Wnt9b", "Glul", "Vwf"]).display()

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
