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

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
import itertools

# %% [markdown]
# ### Method info

# %%
from chromatinhd_manuscript.designs import dataset_latent_peakcaller_diffexp_method_qtl_enricher_combinations as design

# %%
promoter_name = "10k10k"


# %%
def get_score_folder(x):
    return chd.get_output() / "prediction_likelihood" / x.dataset / promoter_name / x.latent / x.method / "scoring" / x.peakcaller / x.diffexp / x.motifscan / x.enricher
design["score_folder"] = design.apply(get_score_folder, axis = 1)

# %%
import scipy.stats

# %%
design_row = (
    design
    # .query("dataset == 'pbmc10k'")
    # .query("dataset == 'lymphoma'")
    # .query("dataset == 'e18brain'")
    # .query("dataset == 'alzheimer'")
    .query("dataset == 'brain'")
    
    .query("peakcaller == 'rolling_50'")
    .query("enricher == 'cluster_vs_all'")
    # .query("motifscan == 'gtex_immune'")
    # .query("motifscan == 'gwas_cns'")
    # .query("motifscan == 'gwas_immune'")
    # .query("motifscan == 'gwas_lymphoma'")
    .query("motifscan == 'gtex_cerebellum'")
    # .query("motifscan == 'onek1k_0.2'")
    # .query("enricher == 'cluster_vs_clusters'")
    .iloc[0]
)
score_folder = design_row["score_folder"]

# %%
print(score_folder)
scores_peaks = pd.read_pickle(
    score_folder / "scores_peaks.pkl"
)
scores_regions = pd.read_pickle(
    score_folder / "scores_regions.pkl"
)

# scores[ix] = scores_peaks
motifscores = pd.merge(scores_peaks, scores_regions, on = scores_peaks.index.names, suffixes = ("_peak", "_region"), how = "outer")

# %%
(motifscores.query("n_peak > 100")["in_region"]).mean(), (motifscores.query("n_region > 100")["in_peak"]).mean()

# %%
motifscores["in_region"].mean(), motifscores["in_peak"].mean()

# %%
sns.histplot(motifscores.query("n_peak > 0")["pval_region"], bins = 10)
sns.histplot(motifscores.query("n_peak > 0")["pval_peak"], bins = 10)

# %%
# motifscores_oi = motifscores.loc["Monocytes"].xs("monoc", level = "cell_id")
motifscores_oi = motifscores
(motifscores_oi.query("n_region > 0")["pval_region"] < 0.05).sum() / (motifscores_oi.query("n_peak > 0")["pval_peak"] < 0.05).sum()

# %%
(motifscores_oi.query("n_region > 0")["in_region"]).mean() / (motifscores_oi.query("n_region > 0")["in_peak"]).mean()

# %%
(motifscores_oi.query("n_peak > 0")["pval_peak"] < 0.05).sum(), (motifscores_oi.query("n_region > 0")["pval_region"] < 0.05).sum()
