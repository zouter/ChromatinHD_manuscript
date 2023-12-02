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
class Prediction(pfa.flow.Flow):
    pass


# %%
promoter_name = "10k10k"
prediction_name = "v14_50freq_sum_sigmoid_initdefault"

# %%
dataset_names = [
    "pbmc10k",
    "e18brain",
    "lymphoma"
]
peak_names = [
    "macs2",
    "cellranger"
]

# %% [markdown]
# ### Is the most predictive window inside a peak?

# %%
fig, ax = plt.subplots(figsize = (5, 3))

for dataset_name in dataset_names:
    for peaks_name in peak_names:
        prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)
        scores_dir = (prediction.path / "scoring" / "windows")

        try:
            gene_best_windows = pd.read_pickle(scores_dir / ("gene_best_windows_" + peaks_name + ".pkl"))
        except BaseException as e:
            print(e)

        # ax.plot(scores.loc["validation"].index, scores.loc["validation"]["reldeltacor"], label = dataset_name)
        ax.plot(
            gene_best_windows["perc"],
            1-gene_best_windows["cum_matched"]
        )
        
    # ax.axhline(0, dashes = (2, 2), color = "#333333")
ax.set_xlabel("Top genes (acording to cor)")
ax.set_title("% genes where most predictive locus is not contained in a peak", rotation = 0, loc='left')
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_ylim(0, 1)
    # ax.legend(title = "dataset")
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
    # ax1.set_xlabel("Fragment length")
    # ax1.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center")
    # sns.despine()

# %% [markdown]
# ### Are all predictive windows within a peak?

# %%
fig, ax = plt.subplots(figsize = (5, 3))

perc_cutoff = 0.05

for dataset_name in dataset_names:
    for peaks_name in peak_names:
        prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)
        scores_dir = (prediction.path / "scoring" / "windows")

        try:
            genescores_matched_oi = pd.read_pickle(scores_dir / ("genescores_matched_" + peaks_name + ".pkl"))
            genescores_matched_oi = genescores_matched_oi.iloc[:int(genescores_matched_oi.shape[0] * perc_cutoff)]
            ax.plot(
                genescores_matched_oi["perc"],
                1-genescores_matched_oi["cum_matched"],
                label = (dataset_name, peaks_name)
            )
        except BaseException as e:
            print(e)

        # ax.plot(scores.loc["validation"].index, scores.loc["validation"]["reldeltacor"], label = dataset_name)

        
    # ax.axhline(0, dashes = (2, 2), color = "#333333")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol = 3, fontsize = 8)
ax.set_xlabel("Top loci (acording to $\Delta$ cor)")
ax.set_title("% of most predictive loci not contained in a peak", rotation = 0, loc='left')
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_ylim(0, 1)
    # ax.legend(title = "dataset")
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
    # ax1.set_xlabel("Fragment length")
    # ax1.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center")
    # sns.despine()

# %%
