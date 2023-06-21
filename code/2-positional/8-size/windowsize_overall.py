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
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"
outcome_source = "counts"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "10k10k", np.array([-10000, 10000])
# outcome_source = "magic"
# prediction_name = "v20"
# prediction_name = "v21"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"
outcome_source = "magic"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %%
genes_oi = transcriptome.var.index
# genes_oi = transcriptome.gene_id(["TCF3"])


# %% [markdown]
# ## Window + size

# %%
scores = []
for gene in tqdm.tqdm(genes_oi):
    try:
        scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
        windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

        scores_folder = prediction.path / "scoring" / "window_gene" / gene
        window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

    except FileNotFoundError:
        continue

    score = (
        windowsize_scoring.genescores.mean("model")
        .sel(phase=["validation", "test"], gene=gene)
        .mean("phase")
        .to_pandas()
    ).reset_index()
    score.loc[:, ["window", "size"]] = windowsize_scoring.design[
        ["window", "size"]
    ].values
    score["deltacor_gene"] = (
        window_scoring.genescores["deltacor"]
        .sel(phase=["validation", "test"])
        .mean("model")
        .mean("phase")
        .sel(gene=gene)
        .to_pandas()[score["window"]]
    ).values
    score = score.loc[score["deltacor_gene"] < -0.01]
    print(len(score))
    scores.append(score)
scores = pd.concat(scores)

# %%
x = scores.set_index(["gene", "window", "size"])["deltacor"].unstack()
# scores["reldeltacor"] = scores["deltacor"] / (scores["lost"] + 1)
# x = scores.set_index(["gene", "window", "size"])["reldeltacor"].unstack()

# %%
cors = []
for gene, x_ in x.groupby("gene"):
    cor = np.corrcoef(x_.T)
    cors.append(cor)
cors = np.stack(cors)
cor = pd.DataFrame(np.nanmean(cors, 0), index=x.columns, columns=x.columns)

# %%
plt.scatter(x.iloc[:, 0], x.iloc[:, 5])

# %%
design_windows = windowsize_scoring.design.groupby("window").first()
design_size = windowsize_scoring.design.groupby("size").first()
design_size["label"] = [
    "footprint",
    "submono",
    "mono",
    "supermono",
    "di",
    "superdi",
    "tri",
    "supertri",
    "multi",
]

# %%
fig, ax = plt.subplots(figsize=(2, 2))

ax.matshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cor)))
ax.set_yticks(range(len(cor)))
ax.set_xticklabels(design_size.loc[cor.columns, "label"], rotation=90)
ax.set_yticklabels(design_size.loc[cor.columns, "label"])

# %%
