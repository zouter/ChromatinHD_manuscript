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
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import torch_scatter
import torch

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import peakfreeatac as pfa

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# transcriptome
dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = pfa.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# fragments
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

fragments = pfa.data.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# create design to run
from design import get_design, get_folds_inference

class Prediction(pfa.flow.Flow):
    pass


# %%
prediction_name = "v14"
prediction_name = "v14_50freq_sum_sigmoid"
prediction_name = "v14_50freq_sum_sigmoid_initdefault"

# %%
prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)
scores_dir = (prediction.path / "scoring" / "lengths")

# %%
design = pd.read_pickle(scores_dir / "design.pkl")
scores = pd.read_pickle(scores_dir / "scores.pkl")
genescores = pd.read_pickle(scores_dir / "genescores.pkl")

# %%
scores_dir_overall = (prediction.path / "scoring" / "overall")

scores_overall = pd.read_pickle(scores_dir_overall / "scores.pkl")
genescores_overall = pd.read_pickle(scores_dir_overall / "genescores.pkl")

# %%
deltacor_window_cutoff = -0.001

scores["deltacor"] = scores["cor"] - scores_overall["cor"]

genescores["deltacor"] = genescores["cor"] - genescores_overall["cor"]
genescores["deltacor_mask"] = genescores["deltacor"] < deltacor_window_cutoff

# %% [markdown]
# ### Global view

# %%
retained = scores["retained"].unstack().T
effect = scores["effect"].unstack().T
score = scores["cor"].unstack().T


# %%
def zscore(x, dim = 0):
    return (x - x.values.mean(dim, keepdims = True))/x.values.std(dim, keepdims = True)
def minmax(x, dim = 0):
    return (x - x.values.min(dim, keepdims = True))/(x.max(dim, keepdims = True) - x.min(dim, keepdims = True))


# %%
fig, ax_perc = plt.subplots(figsize = (4, 4))

ax_perc.axhline(0, dashes = (2, 2), color = "grey")

ax_mse = ax_perc.twinx()
ax_mse.plot(score.index, scores.loc["validation", "deltacor"], color = pfa.plotting.colors[0])
ax_mse.set_ylabel(r"$\Delta$ cor", rotation = 0, ha = "left", va = "center", color = pfa.plotting.colors[0])
ax_mse.invert_yaxis()
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["validation"]) 
ax_perc.set_xlabel("Fragment length")

ax_perc.plot(retained.index, 1-retained["validation"], color = pfa.plotting.colors[1])
ax_perc.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center", color = pfa.plotting.colors[1])
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax_perc.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_mse.yaxis.set_major_locator(plt.MaxNLocator(3))

sns.despine(ax = ax_mse)
sns.despine(ax = ax_perc, right = False)

# %%
fig, ax = plt.subplots(figsize = (4, 4))
scores["reldeltacor"] = ((zscore(score) - zscore(retained))).unstack()
scores["reldeltacor"].loc["validation"].plot()
ax.set_ylabel("Relative\n$\\Delta$ cor", rotation = 0, ha = "right", va = "center")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.axhline(0, dashes = (2, 2), color = "#333333")

# %%
scores.to_pickle(scores_dir / "scores.pkl")

# %% [markdown]
# Do (mono/di/...)"nucleosome" fragments still have an overall positive or negative effect on gene expression?

# %%
fig, ax_perc = plt.subplots(figsize = (4, 4))

ax_perc.axhline(0, dashes = (2, 2), color = "grey")

ax_mse = ax_perc.twinx()
ax_mse.plot(effect.index, scores.loc["validation", "effect"], color = pfa.plotting.colors[0])
ax_mse.set_ylabel(r"effect", rotation = 0, ha = "left", va = "center", color = pfa.plotting.colors[0])
ax_mse.invert_yaxis()
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["validation"]) 
ax_perc.set_xlabel("Fragment length")

ax_perc.plot(retained.index, 1-retained["validation"], color = pfa.plotting.colors[1])
ax_perc.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center", color = pfa.plotting.colors[1])
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax_perc.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_mse.yaxis.set_major_locator(plt.MaxNLocator(3))

sns.despine(ax = ax_mse)
sns.despine(ax = ax_perc, right = False)

# %% [markdown]
# ## Across datasets

# %%
dataset_names = [
    "pbmc10k",
    "e18brain",
    "lymphoma"
]

# %%
datasets = pd.DataFrame({"dataset":dataset_names}).set_index("dataset")
datasets["color"] = sns.color_palette("Set1", datasets.shape[0])

# %%
relative_deltacor = {}

fig, (ax, ax1) = plt.subplots(1, 2, figsize = (8, 4), gridspec_kw={"wspace":0.6})

for dataset_name in dataset_names:
    prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)
    scores_dir = (prediction.path / "scoring" / "lengths")
    
    scores = pd.read_pickle(scores_dir / "scores.pkl")
    
    ax.plot(scores.loc["validation"].index, -scores.loc["validation"]["reldeltacor"], label = dataset_name, color = datasets.loc[dataset_name, "color"])
    ax1.plot(scores.loc["validation"].index, 1-scores.loc["validation"]["retained"], label = dataset_name, color = datasets.loc[dataset_name, "color"])
ax.axhline(0, dashes = (2, 2), color = "#333333")
ax.set_xlabel("Fragment length")
ax.set_ylabel("Relative\n$\\Delta$ cor", rotation = 0, ha = "right", va = "center")
ax.legend(title = "dataset")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax1.set_xlabel("Fragment length")
ax1.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center")
sns.despine()

# %%
for 
relative_deltacor

# %%

# %%

# %%
