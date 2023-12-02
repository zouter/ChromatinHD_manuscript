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
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = pfa.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# fragments
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_name, window = "20kpromoter", np.array([-10000, 0])
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
# prediction_name = "counter"
# prediction_name = "v14"
# prediction_name = "cellranger_linear"
prediction_name = "v14_50freq_sum_sigmoid"
prediction_name = "v14_50freq_sum_sigmoid_initdefault"

# %%
baseline_prediction_name = "counter"
# baseline_prediction_name = "cellranger_linear"

# %%
prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)

scores_dir = (prediction.path / "scoring" / "overall")

scores = pd.read_pickle(scores_dir / "scores.pkl")
genescores = pd.read_pickle(scores_dir / "genescores.pkl")

baseline_prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / baseline_prediction_name)
baseline_scores_dir = (baseline_prediction.path / "scoring" / "overall")
scores_baseline = pd.read_pickle(baseline_scores_dir / "scores.pkl")
genescores_baseline = pd.read_pickle(baseline_scores_dir / "genescores.pkl")

# %%
diffscores = (scores - scores_baseline).rename(columns = lambda x:x + "_diff")
diffscores["mse_diff"] *= -1
scores[diffscores.columns] = diffscores

diffgenescores = (genescores - genescores_baseline).rename(columns = lambda x:x + "_diff")
diffgenescores["mse_diff"] *= -1
genescores[diffgenescores.columns] = diffgenescores

# %%
# genescores.loc["validation"]["mse"]["ENSG00000025708"]

# %%
genescores.to_pickle(scores_dir / "genescores.pkl")

# %% [markdown]
# ### Global view

# %%
phases = pfa.plotting.phases

# %%
cor_cutoff = 0.005
mse_cutoff = 0.05

# %%
(genescores.sort_values("cor_diff")["cor_diff"] > 0).mean()

# %%
sumscores = {
    "same":((genescores["cor_diff"] > -cor_cutoff) & (genescores["cor_diff"] < cor_cutoff)).mean(),
    "higher":((genescores["cor_diff"] > cor_cutoff)).mean(),
    "lower":((genescores["cor_diff"] < -cor_cutoff)).mean()
}

# %%
fig, ax = plt.subplots(figsize = (3, 2))
sns.ecdfplot(genescores.sort_values("cor_diff")["cor_diff"].loc["validation"])
mean = genescores["cor_diff"].loc["validation"].mean()
ax.axvline(mean, color = "red")
ax.set_xlim(-0.1, 0.1)
ticks = ax.set_xticks([-.1, 0, .1])
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax.xaxis.tick_top()
ax.axvline(0, color = "#333333", dashes = (2, 2))
ax.annotate(f"{mean:.2f}", (mean, -0.05), xycoords = "data", ha = "center", va = "top", clip_on = False, color = "red", annotation_clip=False)
ax.set_xlabel("Δcor")
ax.xaxis.set_label_position('top')

# %%
fig, ax = plt.subplots(figsize = (3, 2))
sns.ecdfplot(genescores.sort_values("cor_diff")["cor_diff"].loc["validation"])
ax.axvspan(-0.1, -cor_cutoff, color = "#333333", alpha = 0.1)
ax.axvspan(+0.1, cor_cutoff, color = "green", alpha = 0.1)
mean = genescores["cor_diff"].loc["validation"].mean()
ax.axvline(cor_cutoff, color = "#333333", dashes = (2, 2))
ax.axvline(-cor_cutoff, color = "#333333", dashes = (2, 2))
ax.set_xlim(-0.1, 0.1)
ticks = ax.set_xticks([-.1, cor_cutoff, .1])
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax.xaxis.tick_top()
ax.annotate(f"{sumscores['lower']:.0%} genes\n worse prediction", (0.25, 0.95), xycoords = "axes fraction", ha = "center", va = "top", bbox = bbox_props, color = "#333333")
# ax.annotate(f"{sumscores['same']:.0%} genes\n same prediction", (0.5, -0.45), xycoords = "axes fraction", ha = "center", va = "top", bbox = bbox_props, color = "grey")
ax.annotate(f"{sumscores['higher']:.0%} genes\n better prediction", (0.75, 0.05), xycoords = "axes fraction", ha = "center", va = "bottom", bbox = bbox_props, color = "green")
ax.set_xlabel("Δcor")
ax.xaxis.set_label_position('top') 

# %%

# %%
