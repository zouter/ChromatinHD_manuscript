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

import pickle

import scanpy as sc
import eyck

import tqdm.auto as tqdm
import io

# %%
import crispyKC as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "liverkia_lsecs"
genome = "mm10"
organism = "mm"

folder_data_preproc = folder_data / "liverkia" / "liver_control_JVG28"
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
eyck.modalities.transcriptome.plot_umap(
    adata, ["leiden", "Stab2", "Lyve1", "Vwf", "Wnt2"]
).display()

# %%
adata = adata[adata.obs["leiden"].isin(["0", "23"])].copy()

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
eyck.modalities.transcriptome.plot_umap(
    adata,
    ["leiden", "Stab2", "Lyve1", "Vwf", "Dll4", "Wnt9b", "Wnt2", "Glul"],
    layer="magic",
    panel_size = 1
).display()

# %%
sc.tl.leiden(adata, resolution=0.5)

# %%
eyck.modalities.transcriptome.plot_umap(
    adata,
    ["leiden"],
    layer="magic",
    panel_size = 1
).display()

# %%
adata.obs["celltype2"] = pd.Series({"3":"EC_central", "1":"LSEC_central", "0":"LSEC_portal", "2":"EC_portal"})[adata.obs["leiden"]].values

# %%
adata = adata[adata.obs["celltype2"].isin(["LSEC_portal", "LSEC_central"])].copy()

# %%
sc.pp.highly_variable_genes(adata)

# %%
# add counts
adata2 = sc.read_10x_h5(folder_data_preproc / "liver_naive_JVG28_feature_bc_matrix.h5")
adata2.var.index = adata2.var["gene_ids"]
adata.layers["counts"] = adata2[adata.obs.index, :][:, adata.var.index].X


# %% [markdown]
# ### Create transcriptome

# %%
adata.var["selected"] = (adata.var.index.isin(
    adata.var.query("means > 0.1")
        .sort_values("dispersions_norm", ascending=False)
        .head(8000)
        .index
) | adata.var.symbol.isin(["Stab2", "Lyve1", "Vwf", "Dll4", "Wnt9b", "Wnt2"]))
transcriptome = eyck.modalities.transcriptome.Transcriptome.from_adata(
    adata[
        :,
        adata.var.query("selected").index
    ],
    path=dataset_folder / "transcriptome",
    overwrite=True,
)

# %%
pd.DataFrame({"exp":sc.get.obs_df(transcriptome.adata, transcriptome.gene_id("Rspo3"), layer = "counts"), "celltype":transcriptome.obs["celltype2"], "n_counts":transcriptome.layers["counts"].sum(1)}).groupby("celltype").sum()

# %%
transcriptome.obs["n_counts"] = transcriptome.layers["counts"].sum(1)

# %%


# %% [markdown]
# ### 2-8

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = eyck.modalities.regions.Regions.from_transcripts(
    selected_transcripts, [-2**17, +2**17], dataset_folder / "regions" / "2-8", overwrite = True
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = eyck.modalities.Fragments(dataset_folder / "fragments" / "2-8")
fragments.regions = regions
fragments = eyck.modalities.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
)

# %%
fragments.create_regionxcell_indptr()

# %%
fragments.obs
# %%
fragments.coordinates.shape[0] / fragments.regions.var.shape[0] / np.diff(fragments.regions.window)[0] * 1000
# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = eyck.modalities.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k", overwrite = True
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = eyck.modalities.Fragments(dataset_folder / "fragments" / "100k100k")
fragments.regions = regions
fragments = eyck.modalities.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
)

# %%
fragments.create_regionxcell_indptr()

# %%
fragments.obs
# %%
fragments.coordinates.shape[0] / fragments.regions.var.shape[0] / np.diff(fragments.regions.window)[0] * 1000
# %%
