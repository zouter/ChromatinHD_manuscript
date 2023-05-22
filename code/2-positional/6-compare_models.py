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
# dataset_name = "pbmc10k_gran"
# dataset_name = "pbmc3k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
from chromatinhd_manuscript.designs import (
    dataset_splitter_method_combinations as design,
)

design = design.loc[
    (design["dataset"] == "pbmc10k")
    & (design["splitter"] == "permutations_5fold5repeat")
    & (design["promoter"] == "10k10k")
    & (design["method"].isin(["v20", "v21"]))
]

# %%
# load nothing scoring

scores = []
for _, design_row in design.iterrows():
    prediction = chd.flow.Flow(
        chd.get_output()
        / "prediction_positional"
        / design_row.dataset
        / design_row.promoter
        / design_row.splitter
        / design_row.method
    )

    scorer_folder = prediction.path / "scoring" / "nothing"
    nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

    scores.append(
        pd.DataFrame(
            {
                "cor": nothing_scoring.genescores["cor"]
                .mean("model")
                .sel(phase="test")
                .sel(i=0)
                .to_pandas(),
                "method": design_row.method,
            }
        )
    )
scores = pd.concat(scores).reset_index()

# %%
plotdata = scores.set_index(["method", "gene"])["cor"].unstack("method")


# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))

plotdata = scores.set_index(["method", "gene"])["cor"].unstack("method")
plotdata["diff"] = plotdata["v21"] - plotdata["v20"]
plotdata["gene"] = transcriptome.var["symbol"]
norm = mpl.colors.CenteredNorm(halfrange=0.1)
cmap = mpl.cm.get_cmap("RdBu_r")
ax.scatter(plotdata["v20"], plotdata["v21"], c=cmap(norm(plotdata["diff"])), s=1)

symbols_oi = [
    "CD74",
    "LYN",
    "TNFAIP2",
]
genes_oi = transcriptome.gene_id(symbols_oi)
texts = []
for gene_oi in genes_oi:
    texts.append(
        ax.annotate(
            transcriptome.symbol(gene_oi),
            (plotdata.loc[gene_oi, "v20"], plotdata.loc[gene_oi, "v21"]),
            (-35.0, 0.0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="black"),
            # bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.5),
        )
    )
    text = texts[-1]
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFFAA"),
            mpl.patheffects.Normal(),
        ]
    )
import adjustText

adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black"))

cutoff = 0.05
percs = (
    (plotdata["diff"] > cutoff).mean(),
    (plotdata["diff"] < -cutoff).mean(),
    1 - (np.abs(plotdata["diff"]) > cutoff).mean(),
)
ax.axline((cutoff, 0), (0.80 + cutoff, 0.8), color="black", linestyle="--")
ax.axline((-cutoff, 0), (0.8 - cutoff, 0.8), color="black", linestyle="--")
bbox = dict(boxstyle="round", fc="white", ec="black", lw=0.5)
ax.annotate(
    f"{percs[0]:.1%}",
    (0.0, 0.8),
    (2.0, -2.0),
    textcoords="offset points",
    ha="left",
    va="top",
    bbox=bbox,
)
ax.annotate(
    f"{percs[1]:.1%}",
    (0.8, 0.0),
    (2.0, 0.0),
    textcoords="offset points",
    ha="right",
    va="center",
    bbox=bbox,
)
text = ax.annotate(
    f"{percs[2]:.1%}",
    (0.8, 0.8),
    (5.0, 0.0),
    textcoords="offset points",
    ha="center",
    va="center",
    bbox=bbox,
)
text.set_path_effects(
    [mpl.patheffects.Stroke(linewidth=3, foreground="white"), mpl.patheffects.Normal()]
)
ax.set_xlabel("cor additive model")
ax.set_ylabel("cor\nnon-additive\nmodel", rotation=0, ha="right", va="center")
manuscript.save_figure(fig, "5", "positional_additive_vs_nonadditive", dpi=300)


# %%
plotdata.query("v20 > 0.4").sort_values("diff", ascending=False).head(10)
# %%
cutoff = 0.05
(plotdata["diff"] > cutoff).mean(), (plotdata["diff"] < -cutoff).mean(), 1 - (
    np.abs(plotdata["diff"]) > cutoff
).mean()
# %%
