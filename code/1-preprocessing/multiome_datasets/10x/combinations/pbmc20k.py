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
dataset_name1 = "pbmc10k"
folder_data_preproc1 = chd.get_output() / "data" / dataset_name1
folder_data_preproc1.mkdir(exist_ok=True, parents=True)
folder_dataset1 = chd.get_output() / "datasets" / dataset_name1

dataset_name2 = "pbmc10k_gran"
folder_data_preproc2 = chd.get_output() / "data" / dataset_name2
folder_data_preproc2.mkdir(exist_ok=True, parents=True)
folder_dataset2 = chd.get_output() / "datasets" / dataset_name2

dataset_name = "pbmc20k"
folder_data_preproc = chd.get_output() / "data" / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

genome = "GRCh38"
organism = "hs"

# %% [markdown]
# ## Merge fragments

# %%
# import gzip
# comments = True
# cur_chromosome = None
# comments = [True, True]
# lastlines = [None, None]
# indices = [0, 1]
# replaces = [None, ("-1", "-2")]
# files = [gzip.open(folder_data_preproc1 / "atac_fragments.tsv.gz", "rb").__iter__(), gzip.open(folder_data_preproc2 / "atac_fragments.tsv.gz", "rb").__iter__()]
# keep = True

# fragments = open(folder_data_preproc / "atac_fragments.tsv", "w")

# while keep:
#     starts = []
#     ends = []
#     cells = []
#     keep = False
#     for i,file, comment, lastline, replace in zip(indices, files, comments, lastlines, replaces):
#         for line in tqdm.tqdm(file, mininterval=1):
#             keep = True
#             if comment:
#                 if line.decode().startswith("#"):
#                     continue
#                 else:
#                     comment = False
#                     comments[i] = False
#             fragment = line.decode().split("\t")
#             if cur_chromosome is None:
#                 cur_chromosome = fragment[0]
#                 print(f"starting {cur_chromosome}")
#             starts.append(int(fragment[1]))
#             ends.append(int(fragment[2]))

#             cell = fragment[3]
#             if replace is not None:
#                 cell = cell.replace(replace[0], replace[1])
#             cells.append(cell)

#             if cur_chromosome != fragment[0]:
#                 lastlines[i] = line
#                 break
#     # sort the fragments
#     print(f"sorting {cur_chromosome}")
#     stars = np.array(starts)
#     ends = np.array(ends)
#     order = np.argsort(starts)

#     # write
#     print(f"writing {cur_chromosome}")
#     for i in order:
#         fragments.write(f"{cur_chromosome}\t{starts[i]}\t{ends[i]}\t{cells[i]}\n")

#     cur_chromosome = None

# %%
# bgzip
if not (folder_data_preproc / "atac_fragments.tsv.gz").exists():
    # !bgzip -c {folder_data_preproc / "atac_fragments.tsv"} > {folder_data_preproc / "atac_fragments.tsv.gz"}

# %%
# tabix index
if not (folder_data_preproc / "atac_fragments.tsv.gz.tbi").exists():
    # !tabix -s 1 -b 2 -e 3 {folder_data_preproc / "atac_fragments.tsv.gz"}

# %% [markdown]
# ## Transcriptome

# %%
transcriptome1 = chd.data.Transcriptome(folder_dataset1 / "transcriptome")
transcriptome2 = chd.data.Transcriptome(folder_dataset2 / "transcriptome")

# %%
adata1 = transcriptome1.adata.raw.to_adata()
adata1.obs["cell_original"] = adata1.obs.index
adata1.obs["batch"] = 0
adata2 = transcriptome2.adata.raw.to_adata()
adata2.obs["cell_original"] = adata2.obs.index
adata2.obs.index = adata2.obs.index.str.split("-").str[0] + "-2"
adata2.obs["batch"] = 1

adata = sc.concat(
    [
        adata1, adata2
    ]
)
adata.var["symbol"] = adata1.var["symbol"].reindex(adata.var.index).fillna(adata2.var["symbol"].reindex(adata.var.index))
adata.raw = adata

# %%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color = ["celltype", "batch"])

# %%
pickle.dump(adata, open(folder_data_preproc / "adata_annotated.pkl", "wb"))

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / 'adata_annotated.pkl').open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc1 / 'transcripts.pkl').open("rb"))
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
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %% [markdown]
# ### Create transcriptome

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, adata.var.sort_values("dispersions_norm").tail(5000).index], path=dataset_folder / "transcriptome", overwrite = True)

# %%
sc.pl.umap(adata, color = adata.var.index[(adata.var["symbol"] == "CCL4")][0])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=dataset_folder / "fragments" / "10k10k",
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
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=dataset_folder / "fragments" / "100k100k",
    overwrite = True
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Peaks

# %%
# !ls {chd.get_output()}/peaks

# %%
if not (chd.get_output() / "peaks" / dataset_name).exists():
    # !ln -s {chd.get_output() / "peaks" / "pbmc10k"} {chd.get_output() / "peaks" / dataset_name}
