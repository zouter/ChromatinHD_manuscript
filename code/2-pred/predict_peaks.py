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

sns.set_style("ticks")

import pickle

import scanpy as sc
import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
promoter_name = "10k10k"

# %% [markdown]
# ## Predict (temporarily here 👷)

# %%
import chromatinhd.prediction

# %%
peaks_names = [
    "cellranger",
    "macs2",
    "macs2_improved",
    "rolling_500"
]
design_peaks = pd.DataFrame({"peaks": peaks_names})
methods = [
    # ["_xgboost", chromatinhd.prediction.PeaksGene],
    ["_linear", chromatinhd.prediction.PeaksGeneLinear],
    # ["_polynomial", chromatinhd.prediction.PeaksGenePolynomial],
    # ["_lasso", chromatinhd.prediction.PeaksGeneLasso]
]
design_methods = pd.DataFrame(methods, columns=["method_suffix", "method_class"])
dataset_names = [
    "pbmc10k",
    # "lymphoma",
    # "e18brain",
]
design_datasets = pd.DataFrame({"dataset": dataset_names})

# %%
design = chd.utils.crossing(design_peaks, design_methods, design_datasets)

# %%
for _, design_row in design.iterrows():
    print(design_row)
    dataset_name = design_row["dataset"]
    peaks_name = design_row["peaks"]
    transcriptome = chd.data.Transcriptome(
        chd.get_output() / "data" / dataset_name / "transcriptome"
    )
    peakcounts = chd.peakcounts.FullPeak(
        folder=chd.get_output() / "peakcounts" / dataset_name / peaks_name
    )

    peaks = peakcounts.peaks

    gene_peak_links = peaks.reset_index()
    gene_peak_links["gene"] = pd.Categorical(
        gene_peak_links["gene"], categories=transcriptome.adata.var.index
    )

    fragments = chromatinhd.data.Fragments(
        chd.get_output() / "data" / dataset_name / "fragments" / promoter_name
    )
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))

    method_class = design_row["method_class"]
    method_suffix = design_row["method_suffix"]
    prediction = method_class(
        chd.get_output()
        / "prediction_positional"
        / dataset_name
        / promoter_name
        / (peaks_name + method_suffix),
        transcriptome,
        peakcounts,
    )

    prediction.score(gene_peak_links, folds)

    prediction.scores = prediction.scores
    # prediction.models = prediction.models

# %%
# !ls {prediction.path}/scoring/overall

# %%
