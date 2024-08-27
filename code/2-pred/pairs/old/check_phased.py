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

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")


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

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"
prediction_name = "v21"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %% [markdown]
# ## Subset

# %%
scores_all = pd.read_csv("~/NAS2/wsaelens/scores_all.csv", index_col=0)

# %% [markdown]
# ## Pairwindow

# %%
cors = []
for gene, gene_scores in scores_all.groupby("gene"):
    print(gene)
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if not interaction_file.exists():
        continue
    interaction = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()

    design = pd.read_pickle(scores_folder / "design.pkl")

    gene_scores["window_1"] = design.index[
        np.clip(
            np.searchsorted(design.index, gene_scores["position_1"]), 0, len(design) - 1
        )
    ]
    gene_scores["window_2"] = design.index[
        np.clip(
            np.searchsorted(design.index, gene_scores["position_2"]), 0, len(design) - 1
        )
    ]

    interaction_scores = interaction.set_index(["window1", "window2"]).reindex(
        pd.MultiIndex.from_frame(gene_scores[["window_1", "window_2"]])
    )
    interaction_scores["score"] = gene_scores["score"].values
    print(pd.isnull(interaction_scores["cor"]).mean())
    # interaction_scores["cor"] = interaction_scores["cor"].fillna(0)
    interaction_scores = interaction_scores.loc[~pd.isnull(interaction_scores["cor"])]

    if len(interaction_scores) > 2:
        cors.append(
            np.corrcoef(interaction_scores["score"], interaction_scores["cor"])[0, 1]
        )
    if gene == transcriptome.gene_id("CCL4"):
        break
np.mean(cors)

# %%

# %%
