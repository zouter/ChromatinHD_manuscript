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

# %%
chd.set_default_device("cuda:1")

# %% [markdown]
# ### Download CRISPRi data

# %%
file = chd.get_output() / "data" / "crispri" / "nasser_2021" / "41586_2021_3446_MOESM6_ESM.txt"
file.parent.mkdir(parents=True, exist_ok=True)

import requests
if not file.exists():
    response = requests.get("https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03446-x/MediaObjects/41586_2021_3446_MOESM6_ESM.txt")
    with open(file, "wb") as f:
        f.write(response.content)

# %%
import liftover
converter = liftover.get_lifter("hg19", "hg38")

# %%
data_orig = pd.read_table(file, comment = "#")
data_orig["chrom"] = data_orig["chrPerturbationTarget"]
data_orig["HS_LS_logratio"] = data_orig["EffectSize"]

data = data_orig.copy()
data["start_orig"] = data["startPerturbationTarget"]
data["end_orig"] = data["endPerturbationTarget"]

data["start"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["start_orig"])]
data["end"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["end_orig"])]
data["Gene"] = data["GeneSymbol"]
data["Significant"] = data["padj"] < 0.05

# %%
# data = data.loc[data["CellType"] == "GM12878"]
# data = data.loc[data["CellType"].isin(["Jurkat", "Jurkat_anti-CD3_PMA_4hr"])]
data

# %% [markdown]
# ### Load interpretation

# %%
dataset_name = "pbmc10k"
# dataset_name = "lymphoma"

# %%
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")

# %%
models = [chd.models.pred.model.additive.Model(chd.get_output() / "pred" /dataset_name / "100k100k" / "5x5" / "normalized" / "v20" / str(model_ix)) for model_ix in range(10)]
# models = [chd.models.pred.model.additive.Model(chd.get_output() / "pred" /dataset_name / "100k100k" / "5x5" / "magic" / "v20" / str(model_ix)) for model_ix in range(3)]
folds = chd.data.folds.Folds(chd.get_output() / "datasets" /dataset_name / "folds" / "5x5")

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, (1000, ))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
regionmultiwindow.score(fragments, transcriptome, models, folds, censorer, regions = genes_oi)
regionmultiwindow.interpolate()

# %% [markdown]
# ### Check which symbols overlap

# %%
joined_all = []
for gene_oi in tqdm.tqdm(genes_oi):
# for gene_oi in tqdm.tqdm(transcriptome.gene_id(["JUNB"])):
    region = fragments.regions.coordinates.loc[gene_oi]
    symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]
    
    data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
    data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
    data_oi["start"] = data_oi["start"].astype(int)
    data_oi["end"] = data_oi["end"].astype(int)

    data_oi["z"] = (data["HS_LS_logratio"]) / data["HS_LS_logratio"].std()

    data_oi = data_oi.loc[data_oi["start"] > region["start"]]
    data_oi = data_oi.loc[data_oi["end"] < region["end"]]

    if data_oi.shape[0] > 0:
        data_centered = chd.plot.genome.genes.center(data_oi, region)
        data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

        data_centered["bin"] = regionmultiwindow.design.index[np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"])-1]
        data_binned = data_centered.groupby("bin").mean(numeric_only = True)

        joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
        joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

        joined_shared = joined.loc[joined["HS_LS_logratio"].notnull()].copy()
        joined_shared["gene"] = gene_oi
        joined_all.append(joined_shared)
joined_all = pd.concat(joined_all)

# %%
fig, ax =plt.subplots(figsize = (5, 2))
ax.scatter(joined_all["deltacor"], joined_all["HS_LS_logratio"], c = joined_all["Significant"], cmap = "viridis", s = 10, vmin = 0, vmax = 1)

fig, ax =plt.subplots(figsize = (5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_all["gene"].unique())), index = joined_all["gene"].unique())
ax.scatter(joined_all["deltacor"], joined_all["HS_LS_logratio"], c = colors[joined_all["gene"]])
for gene, color in colors.items():
    ax.scatter([], [], c = color, label = transcriptome.symbol(gene))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axvline(-0.00)

# %%
joined_all["significant_expression"] = (joined_all["Significant"] > 0)
joined_all["significant_chd"] = joined_all["deltacor"] < -0.005

# %%
confirmed = (joined_all["significant_expression"] & joined_all["significant_chd"]).sum() / joined_all["significant_chd"].sum()

randoms = []
for i in range(1000):
    randoms.append((joined_all.iloc[np.random.permutation(np.arange(len(joined_all)))]["significant_expression"].values & joined_all["significant_chd"]).sum() / joined_all["significant_chd"].sum())
randoms = np.array(randoms)

fig, ax =plt.subplots(figsize = (5, 2))
ax.hist(randoms, bins = np.linspace(0, 1, 20), density = True)
ax.axvline(confirmed, color = "red")
(randoms >= confirmed).mean()

# %%
import fisher
contingency = pd.crosstab(joined_all["significant_expression"], joined_all["significant_chd"])
fisher.pvalue(*contingency.values.flatten())
odds = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / (contingency.iloc[1, 0] * contingency.iloc[0, 1])
odds

# %%
gene_oi = transcriptome.gene_id("CD40")
gene_oi = transcriptome.gene_id("BLK")
gene_oi = transcriptome.gene_id("ITGAL")
gene_oi = transcriptome.gene_id("ETS1")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
data_oi["start"] = data_oi["start"].astype(int)
data_oi["end"] = data_oi["end"].astype(int)

data_oi = data_oi.loc[data_oi["start"] > region["start"]]
data_oi = data_oi.loc[data_oi["end"] < region["end"]]

data_centered = chd.plot.genome.genes.center(data_oi, region)
data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

data_centered["bin"] = regionmultiwindow.design.index[np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"])-1]
data_binned = data_centered.groupby("bin").mean(numeric_only = True)

# %%
joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height = 0))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 20000]
# window = [-10000, 10000]

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width = 10, window = window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(chd.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["HS_LS_logratio"], lw = 0, width = binwidth)
ax.set_xlim(*window)
ax.set_ylabel("CRISPRi score", rotation = 0, ha = "right", va = "center")

panel, ax = fig.main.add_under(chd.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["deltacor"], lw = 0, width = binwidth)
ax.set_xlim(*window)
ax.set_ylabel("$\Delta cor$", rotation = 0, ha = "right", va = "center")

panel, ax = fig.main.add_under(chd.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["lost"], lw = 0, width = binwidth)
ax.set_xlim(*window)
ax.set_ylabel("# fragments", rotation = 0, ha = "right", va = "center")

panel, ax = fig.main.add_under(chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = fragments.regions.window, width = 10))
ax.set_xlim(*window)

fig.plot()

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
