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
#     display_name: Python 3
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

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd

# %%
device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# splitter = "random_5fold"
# promoter_name, window = "10k10k", np.array([-10000, 10000])
# prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# create design to run
from design import get_design, get_folds_inference


class Prediction(chd.flow.Flow):
    pass


# folds & minibatching
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxregion_batch_size = get_folds_inference(fragments, folds, n_cells_step=2000)
folds = folds  # [:1]

# design
from design import get_design, get_folds_training

design = get_design(transcriptome, fragments)

# %%
Scorer = chd.scoring.prediction.Scorer

# %%
design_row = design[prediction_name]

# %%
fragments.window = window

# %%
design_row["loader_parameters"]["cellxregion_batch_size"] = cellxregion_batch_size

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output() / "prediction_positional" / dataset_name / promoter_name / splitter / prediction_name
)

# %% [markdown]
# ## Neighbors

# %%
import faiss

# X = np.array(transcriptome.adata.X.todense())
X = transcriptome.adata.obsm["X_pca"]

index = faiss.index_factory(X.shape[1], "Flat")
index.train(X)
index.add(X)
distances, neighbors = index.search(X, 50)
neighbors = neighbors[:, 1:]

# %% [markdown]
# ## Subset

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

# symbol = "SELL"
# symbol = "BACH2"
# symbol = "CTLA4"
# symbol = "IL1B"
# symbol = "SPI1"
# symbol = "IL1B"
symbol = "TCF3"
# symbol = "CCL4"
genes_oi = transcriptome.var["symbol"] == symbol
gene = transcriptome.var.index[genes_oi][0]
folds, cellxregion_batch_size = get_folds_inference(fragments, folds, n_cells_step=2000, genes_oi=genes_oi)
folds = folds  # [:1]

gene_ix = transcriptome.gene_ix(symbol)
gene = transcriptome.var.iloc[gene_ix].name

# %%
sc.pl.umap(transcriptome.adata, color=gene, use_raw=False, show=False)

# %% [markdown]
# ## Window + size

# %%
scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
deltacor = windowsize_scoring.genescores.mean("model").sel(phase="validation", gene=gene)["deltacor"]
lost = windowsize_scoring.genescores.mean("model").sel(phase="validation", gene=gene)["lost"]
reldeltacor = windowsize_scoring.genescores.mean("model").sel(phase="validation", gene=gene)["deltacor"] / (
    1 - windowsize_scoring.genescores.mean("model").sel(phase="validation", gene=gene)["retained"]
)

# %%
deltacor.coords["window_size"] = pd.MultiIndex.from_frame(windowsize_scoring.design[["window", "size"]])
lost.coords["window_size"] = pd.MultiIndex.from_frame(windowsize_scoring.design[["window", "size"]])

# %%
sns.heatmap(deltacor.to_pandas().unstack())

# %%
fig, ax = plt.subplots(figsize=(30, 5))
ax.plot(lost.to_pandas().unstack().index, np.log1p(lost).to_pandas().unstack(), marker=".")
# ax.set_xlim([-20000, -8000])
# ax.set_xlim([3000, 6000])
ax.set_xlim(*window)

fig, ax = plt.subplots(figsize=(30, 5))
plotdata = deltacor.to_pandas().unstack()
for size, plotdata_size in plotdata.items():
    ax.plot(plotdata_size.index, plotdata_size, marker=".", label=size)
ax.legend()
# ax.set_xlim([-20000, -8000])
# ax.set_xlim([3000, 6000])
ax.set_xlim(*window)

# %%
chd.utils.paircor(deltacor.unstack().values, np.log(0.1 + lost.unstack().values))

# %%
sns.heatmap(np.corrcoef(deltacor.to_pandas().unstack().T.loc[:, (lost.unstack().sum("size") > 10).values]))

# %%
deltacor

# %%
sns.heatmap(np.corrcoef(deltacor.to_pandas().unstack().T))

# %%

# %%
