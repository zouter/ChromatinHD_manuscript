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
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

# symbol = "SELL"
# symbol = "BACH2"
# symbol = "CTLA4"
# symbol = "IL1B"
# symbol = "SPI1"
# symbol = "IL1B"
# symbol = "TCF3"
# symbol = "CCL4"
# symbol = "CD74"
symbol = "BCL2"
symbol = "ZNF652"
symbol = "IPP"
symbol = "LYN"
# symbol = "TNFAIP2"
# symbol = "CD74"
# symbol = "TCF3"
# symbol = "JCHAIN"
# symbol = "BCL11B"
symbol = "TCF3"
genes_oi = transcriptome.var["symbol"] == symbol
gene = transcriptome.var.index[genes_oi][0]

gene = "ENSG00000231389"


# %% [markdown]
# ## Window + size

# %%
scores_folder = prediction.path / "scoring" / "window_gene" / gene
window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
scores = (
    windowsize_scoring.genescores.mean("model")
    .sel(phase=["validation", "test"], gene=gene)
    .mean("phase")
    .to_pandas()
).reset_index()
scores.loc[:, ["window", "size"]] = windowsize_scoring.design[["window", "size"]].values
scores["deltacor_gene"] = (
    window_scoring.genescores["deltacor"]
    .sel(phase=["validation", "test"])
    .mean("model")
    .mean("phase")
    .sel(gene=gene)
    .to_pandas()[scores["window"]]
).values
# scores = scores.loc[scores["deltacor_gene"] < -0.01]

# %%
deltacor = (
    windowsize_scoring.genescores.mean("model")
    .sel(phase=["validation", "test"], gene=gene)
    .mean("phase")["deltacor"]
)
lost = (
    windowsize_scoring.genescores.mean("model")
    .sel(phase=["validation", "test"], gene=gene)
    .mean("phase")["lost"]
)
retained = (
    windowsize_scoring.genescores.mean("model")
    .sel(phase=["validation", "test"], gene=gene)
    .mean("phase")["retained"]
)
reldeltacor = deltacor / retained

# %%
deltacor.coords["window_size"] = pd.MultiIndex.from_frame(
    windowsize_scoring.design[["window", "size"]]
)
lost.coords["window_size"] = pd.MultiIndex.from_frame(
    windowsize_scoring.design[["window", "size"]]
)

# %%
sns.heatmap(deltacor.to_pandas().unstack())

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
plt.plot(scores["window"], scores["deltacor_gene"], marker=".")

# %%
fig, ax = plt.subplots(figsize=(30, 5))

plotdata = scores
bottom = np.zeros(len(plotdata["window"].unique()))
for size, plotdata_size in plotdata.groupby("size"):
    x = np.clip(plotdata_size["deltacor"], -np.inf, 0)
    # x = np.clip(plotdata_size["lost"], -np.inf, np.inf)
    ax.bar(
        plotdata_size["window"],
        x,
        width=100,
        label=design_size.loc[size, "label"],
        lw=0,
    )
    bottom += x
# ax.set_xlim([90000, 100000])
# ax.set_xlim([-60000, -40000])
# ax.set_xlim([-10000, 10000])
ax.set_xlim([50000, 60000])
# ax.set_xlim([80000, 90000])
ax.legend()


# %%
fig, ax = plt.subplots(figsize=(30, 5))
ax.plot(
    lost.to_pandas().unstack().index, np.log1p(lost).to_pandas().unstack(), marker="."
)
ax.set_xlim([-20000, -8000])
# ax.set_xlim([3000, 6000])

fig, ax = plt.subplots(figsize=(30, 5))
plotdata = deltacor.to_pandas().unstack()
for size, plotdata_size in plotdata.items():
    ax.plot(plotdata_size.index, plotdata_size, marker=".", label=size)
ax.legend()
ax.set_xlim([-20000, -8000])
# ax.set_xlim([3000, 6000])

# %%
chd.utils.paircor(deltacor.unstack().values, np.log(0.1 + lost.unstack().values))

# %%
fig, ax = plt.subplots()
cor = np.corrcoef(deltacor.to_pandas().unstack().T)
cor = pd.DataFrame(
    cor,
    index=deltacor.to_pandas().unstack().columns,
    columns=deltacor.to_pandas().unstack().columns,
)
ax.matshow(cor)
ax.set_xticks(range(len(cor)))
ax.set_yticks(range(len(cor)))
ax.set_xticklabels(cor.columns)
ax.set_yticklabels(cor.columns)

# %%
x = deltacor.to_pandas().unstack().T.loc[design_size.index, design_windows.index].values
w = (
    window_scoring.genescores["deltacor"]
    .sel(phase=["validation", "test"])
    .sum("model")
    .sum("phase")
    .sel(gene=gene)
    .to_pandas()[design_windows.index]
).values

# %%
mx = (x * w).sum(1) / w.sum(0)
sx = (w * (x - mx[:, None]) ** 2).sum(1) / w.sum(0)
sxy = (((x - mx[:, None])[:, None, :] * (x - mx[:, None])[None, :, :]) * w).sum(
    -1
) / w.sum(0)

wcor = sxy / np.sqrt(sx[:, None] * sx[None, :])
wcor = pd.DataFrame(wcor, index=design_size.index, columns=design_size.index)

# %%
cor = pd.DataFrame(
    np.corrcoef(x),
    index=deltacor.to_pandas().unstack().columns,
    columns=deltacor.to_pandas().unstack().columns,
)

# %%
x.shape

# %%
fig, ax = plt.subplots(figsize=(2, 2))

ax.matshow(wcor)
ax.set_xticks(range(len(cor)))
ax.set_yticks(range(len(cor)))
ax.set_xticklabels(design_size.loc[cor.columns, "label"], rotation=90)
ax.set_yticklabels(design_size.loc[cor.columns, "label"])

# %%
