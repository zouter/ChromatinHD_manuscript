# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
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
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %% [markdown]
# ### Download CRISPRi data

# %%
file = chd.get_output() / "data" / "crispri" / "reilly_2021" / "41588_2021_900_MOESM5_ESM.xlsx"
file.parent.mkdir(parents=True, exist_ok=True)

import requests
if not file.exists():
    response = requests.get("https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-021-00900-4/MediaObjects/41588_2021_900_MOESM5_ESM.xlsx")
    with open(file, "wb") as f:
        f.write(response.content)

# %% [markdown]
# ### Load interpretation

# %%
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / "pbmc10k" / "fragments" / "100k100k")

# %%
symbols = ["GATA1", "HDAC", "FADS1", "FADS2", "FADS3", "FEN1", "LMO2", "ERP29", "CD164", "NMU", "MEF2C", "MYC", "PVT1", "CAT", "CAPRIN1"]
symbols = ["CD2"]
[symbol for symbol in symbols if (symbol in transcriptome.var["symbol"].tolist())]

# %%
models = [chd.models.pred.model.additive.Model(chd.get_output() / "pred" / "pbmc10k" / "100k100k" / "5x5" / "normalized" / "v20" / str(model_ix)) for model_ix in range(5)]
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / "pbmc10k" / "folds" / "5x5")[:5]

symbols_oi = ["MEF2C", "FADS3", "MYC"]
genes_oi = transcriptome.gene_id(symbols_oi)
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, (200, ))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
regionmultiwindow.score(fragments, transcriptome, models, folds, censorer, regions = genes_oi)
regionmultiwindow.interpolate()

# %% [markdown]
# ### Check which symbols overlap

# %%
gene_oi = transcriptome.gene_id("MYC")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
data = pd.read_excel(file, sheet_name=symbol_oi + "_rep2")
data["HS_LS_logratio"] = np.log1p(data["HS_reads"]) - np.log1p(data["LS_reads"])
data["chrom"] = data["Coordinates"].str.split(":", expand=True)[0]
data["start"] = data["Coordinates"].str.split(":", expand=True)[1].str.split("-", expand=True)[0]
data["end"] = data["Coordinates"].str.split(":", expand=True)[1].str.split("-", expand=True)[1]

# %%
data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
data_oi["start"] = data_oi["start"].astype(int)
data_oi["end"] = data_oi["end"].astype(int)
data_oi = data_oi.loc[data_oi["start"] > region["start"]]
data_oi = data_oi.loc[data_oi["end"] < region["end"]]
data_centered = chd.plot.genome.genes.center(data_oi, region)

# %%
data_centered["bin"] = regionmultiwindow.design.index[np.digitize(data_centered["start"], regionmultiwindow.design["window_start"])-1]
data_binned = data_centered.groupby("bin").mean()

# %%
joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

# %%
fig, ax = plt.subplots()
# ax.scatter(joined["HS_LS_logratio"], joined["deltacor"])
ax.scatter(joined["window_mid"], joined["HS_LS_logratio"])
ax2 = ax.twinx()
ax2.scatter(joined["window_mid"], joined["deltacor"], color = "orange")
# ax.set_xlim(-1000, 1000)

# %%
fig, ax = plt.subplots()
ax.scatter(-joined["deltacor"], joined["HS_LS_logratio"])
ax.set_xlabel("deltacor")
ax.set_ylabel("HS_LS_logratio")

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = gene_oi)

# %%
fig, ax = plt.subplots()
ax.scatter(data_centered["start"], data_centered["HS_LS_logratio"], s=30)
(regionmultiwindow.interpolated[genes_oi[0]]["deltacor"]*10).plot(ax = ax, color = "green")
ax.set_xlim(-100000, 100000)

# %%
regionmultiwindow.interpolated[genes_oi[0]]["deltacor"].plot()

# %%
fig, ax = plt.subplots()
ax.scatter(data_oi["start"], data_oi["HS_LS_logratio"], s=1)

# %%
data["HS_LS_logratio"].plot()
