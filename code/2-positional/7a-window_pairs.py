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

# ##
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)
genes_all_oi = transcriptome.var.index[
    (nothing_scoring.genescores.sel(phase="test").mean("model").mean("i")["cor"] > 0.1)
]
transcriptome.var.loc[genes_all_oi].head(30)

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

# symbol = "SELL"
# symbol = "BACH2"
# symbol = "CTLA4"
# symbol = "SPI1"
# symbol = "IL1B"
# symbol = "TCF3"
# symbol = "PCDH9"
# symbol = "EBF1"
# symbol = "QKI"
# symbol = "NKG7"
# symbol = "CCL4"
# symbol = "TCF4"
# symbol = "TSHZ2"
# symbol = "IL1B"
# symbol = "PAX5"
# symbol = "CUX2"
# symbol = "CD79A"
# symbol = "RALGPS2"
# symbol = "RHEX"
# symbol = "PTPRS"
# symbol = "RGS7"
# symbol = "CD74"
# symbol = "PLXNA4"
# symbol = "TNFRSF21"
# symbol = "MEF2C"
# symbol = "BCL2"
# symbol = "CCL4"
# symbol = "EBF1"
# symbol = "LYN"
# symbol = "CD74"
symbol = "TNFRSF13C"
symbol = "CCR6"
symbol = "BCL2"
# symbol = transcriptome.symbol("ENSG00000170345")
print(symbol)
genes_oi = transcriptome.var["symbol"] == symbol
gene = transcriptome.var.index[genes_oi][0]

gene_ix = transcriptome.gene_ix(symbol)
gene = transcriptome.var.iloc[gene_ix].name

# %%
sc.pl.umap(transcriptome.adata, color=gene, use_raw=False, show=False)

# %% [markdown]
# ## Window

# %% [markdown]
# ### Load

# %%
scores_folder = prediction.path / "scoring" / "window_gene" / gene
window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
sns.scatterplot(
    x=window_scoring.genescores["deltacor"]
    .sel(phase="test")
    .mean("model")
    .values.flatten(),
    y=window_scoring.genescores["deltacor"]
    .sel(phase="validation")
    .mean("model")
    .values.flatten(),
    hue=np.log1p(
        window_scoring.genescores["lost"]
        .sel(phase="validation")
        .mean("model")
        .values.flatten()
    ),
)

# %%
# genescores["cor"].mean("model").sel(phase = "train").sel(gene = transcriptome.gene("IL1B")).plot()
# genescores["cor"].mean("model").sel(phase = "validation").sel(gene = transcriptome.gene("CTLA4")).plot()
fig, ax = plt.subplots()
window_scoring.genescores["deltacor"].sel(phase="validation").sel(gene=gene).mean(
    "model"
).plot(ax=ax)
window_scoring.genescores["deltacor"].sel(phase="test").sel(gene=gene).mean(
    "model"
).plot(ax=ax, color="blue")
ax.yaxis_inverted()
ax2 = ax.twinx()
window_scoring.genescores["retained"].sel(phase="test").mean("gene").mean("model").plot(
    ax=ax2, color="red"
)
window_scoring.genescores["retained"].sel(phase="validation").mean("gene").mean(
    "model"
).plot(ax=ax2, color="orange")
ax2.yaxis_inverted()

# %% [markdown]
# ## Pairwindow

scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
interaction_file = scores_folder / "interaction.pkl"

interaction = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
interaction = interaction.rename(columns={0: "cor"})
assert len(interaction) > 0

# %%
promoter = promoters.loc[gene]
genome_folder = folder_data_preproc / "genome"

# %%
radius = (
    (window_scoring.design["window_end"] - window_scoring.design["window_start"])
).iloc[0] / 2

# %%
plotdata_interaction = interaction.copy()
plotdata_interaction = plotdata_interaction.loc[
    (plotdata_interaction["lost1"] > 10)
    & (plotdata_interaction["lost2"] > 10)
    & (plotdata_interaction["distance"] > 1000)
]
print(len(plotdata_interaction))

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

panel_width = 8

plotdata_predictive = (
    window_scoring.genescores.sel(gene=gene).sel(phase="test").mean("model").to_pandas()
)
plotdata_predictive["position"] = plotdata_predictive.index

panel_genes = chdm.plotting.genes.Genes(
    promoter, genome_folder, window, width=panel_width
)
panel_genes = main.add_under(panel_genes)

panel_predictive = chd.predictive.plot.Predictive(
    plotdata_predictive, window, panel_width
)
panel_predictive = main.add_under(panel_predictive, padding=0)

panel_interaction = main.add_under(chd.grid.Panel((panel_width, panel_width / 2)))
ax = panel_interaction.ax

# norm = mpl.colors.Normalize(0, plotdata_interaction["cor"].max())
# norm = mpl.colors.Normalize(-0.001, 0.001)
norm = mpl.colors.CenteredNorm(0, np.abs(plotdata_interaction["cor"]).max())

cmap = mpl.cm.RdBu_r

offsets = []
colors = []

ax.set_ylim((window[1] - window[0]) / 2)
ax.set_xlim(*window)

data_to_pixels = ax.transData.get_matrix()[0, 0]
pixels_to_points = 1 / fig.get_dpi() * 72.0
size = np.pi * (data_to_pixels * pixels_to_points * radius) ** 2

for windowpair, plotdata_row in plotdata_interaction.iterrows():
    window1 = plotdata_row["window1"]
    window2 = plotdata_row["window2"]

    center = np.array(
        [
            window1 + (window2 - window1) / 2,
            (window2 - window1) / 2,
        ]
    )
    offsets.append(center)
    colors.append(cmap(norm(plotdata_row["cor"])))

    # if len(offsets) > 10000:
    #     break

collection = mpl.collections.RegularPolyCollection(
    4,
    sizes=(size,),
    offsets=offsets,
    transOffset=ax.transData,
    ec=None,
    lw=0,
    fc=colors,
)
ax.add_collection(collection)

for x in np.linspace(*window, 16):
    x2 = x
    x1 = x2 + (window[0] - x2) / 2
    y2 = 0
    y1 = x2 - x1

    if np.isclose(x1, window[1]) or (np.isclose(x2, window[1])):
        color = "black"
        lw = 1
        zorder = 10
    else:
        color = "#eee"
        lw = 0.5
        zorder = -1
    ax.plot(
        [x1, x2],
        [y1, y2],
        zorder=zorder,
        color=color,
        lw=lw,
    )
    ax.plot(
        [-x1, -x2],
        [y1, y2],
        zorder=zorder,
        color=color,
        lw=lw,
    )
ax.axis("off")

ax.axvline(66148)

fig.plot()

# %%
fig.savefig("fig.png", dpi=300)

# %% [markdown]
# ## Plot with HiC
hic, bins_hic = chdm.hic.extract_hic(promoter)
import itertools

hic = (
    pd.DataFrame(
        index=pd.MultiIndex.from_frame(
            pd.DataFrame(
                itertools.product(bins_hic.index, bins_hic.index),
                columns=["window1", "window2"],
            )
        )
    )
    .join(hic, how="left")
    .fillna({"balanced": 0.0})
)

# %%
radius_hic = ((bins_hic["start"] - bins_hic["end"])).iloc[0]

plotdata_hic = hic.copy().reset_index()
plotdata_hic["balanced_log"] = np.log1p(plotdata_hic["balanced"])
plotdata_hic["distance"] = np.abs(plotdata_hic["window1"] - plotdata_hic["window2"])
plotdata_hic = plotdata_hic.loc[plotdata_hic["distance"] > 1000]

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

panel_width = 8

plotdata_predictive = (
    window_scoring.genescores.sel(gene=gene).sel(phase="test").mean("model").to_pandas()
)
plotdata_predictive["position"] = plotdata_predictive.index

##
panel_interaction = main.add_under(chd.grid.Panel((panel_width, panel_width / 2)))
ax = panel_interaction.ax

# norm = mpl.colors.Normalize(0, plotdata_interaction["cor"].max())
# norm = mpl.colors.Normalize(-0.001, 0.001)
norm = mpl.colors.CenteredNorm(0, np.abs(plotdata_interaction["cor"]).max())

cmap = mpl.cm.RdBu_r

offsets = []
colors = []

ax.set_ylim(0, (window[1] - window[0]) / 2)
ax.set_xlim(*window)

data_to_pixels = ax.transData.get_matrix()[0, 0]
pixels_to_points = 1 / fig.get_dpi() * 72.0
size = np.pi * (data_to_pixels * pixels_to_points * radius) ** 2

for windowpair, plotdata_row in plotdata_interaction.iterrows():
    window1 = plotdata_row["window1"]
    window2 = plotdata_row["window2"]

    center = np.array(
        [
            window1 + (window2 - window1) / 2,
            (window2 - window1) / 2,
        ]
    )
    offsets.append(center)
    colors.append(cmap(norm(plotdata_row["cor"])))

    # if len(offsets) > 10000:
    #     break

collection = mpl.collections.RegularPolyCollection(
    4,
    sizes=(size,),
    offsets=offsets,
    transOffset=ax.transData,
    ec=None,
    lw=0,
    fc=colors,
)
ax.add_collection(collection)

##

peaks_folder = chd.get_output() / "peaks" / dataset_name
peaks_panel = main.add_under(
    chdm.plotting.Peaks(
        promoter,
        peaks_folder,
        window=window,
        width=panel_width,
        row_height=0.8,
    )
)

panel_genes = chdm.plotting.genes.Genes(
    promoter, genome_folder, window, width=panel_width
)
panel_genes = main.add_under(panel_genes)

panel_predictive = chd.predictive.plot.Predictive(
    plotdata_predictive, window, panel_width
)
panel_predictive = main.add_under(panel_predictive, padding=0)

panel_interaction = main.add_under(chd.grid.Panel((panel_width, panel_width / 2)))
ax = panel_interaction.ax

norm = mpl.colors.Normalize(0, np.abs(plotdata_hic["balanced_log"]).max())
cmap = mpl.cm.magma

offsets = []
colors = []

ax.set_ylim((window[1] - window[0]) / 2)
ax.set_xlim(*window)

data_to_pixels = ax.transData.get_matrix()[0, 0]
pixels_to_points = 1 / fig.get_dpi() * 72.0
size = np.pi * (data_to_pixels * pixels_to_points * radius_hic) ** 2

for windowpair, plotdata_row in plotdata_hic.iterrows():
    window1 = plotdata_row["window1"]
    window2 = plotdata_row["window2"]

    center = np.array(
        [
            window1 + (window2 - window1) / 2,
            (window2 - window1) / 2,
        ]
    )
    offsets.append(center)
    colors.append(cmap(norm(plotdata_row["balanced_log"])))

collection = mpl.collections.RegularPolyCollection(
    4,
    sizes=(size,),
    offsets=offsets,
    transOffset=ax.transData,
    ec=None,
    lw=0,
    fc=colors,
)
ax.add_collection(collection)
ax.axis("off")

fig.plot()

# %%
fig.savefig("fig.png", dpi=1000)

# %% [markdown]
# ## Focus on a single window

position_oi = 66148  # rs17758695, https://www.nature.com/articles/s41588-019-0362-6, https://www.sciencedirect.com/science/article/pii/S0002929721003037?via%3Dihub#app3
# position_oi = (
#     promoter["tss"] - 63235095
# )  # rs954954, https://www.sciencedirect.com/science/article/pii/S0002929721003037?via%3Dihub#app3
# position_oi = promoter["tss"] - 63248351  # rs9967405, random search
# position_oi = promoter["tss"] - 63239451  # rs9967405, random search
# position_oi = promoter["tss"] - 63121512  # rs9967405, random search

windows_all = np.unique(interaction[["window1", "window2"]].values.flatten())
window_oi = windows_all[np.argmin(np.abs(windows_all - position_oi))]

interaction_single = interaction.query("(window1 == @window_oi)").set_index("window2")
interaction_single = interaction_single.loc[
    (interaction_single["lost1"] > 10)
    & (interaction_single["lost2"] > 10)
    # & (interaction_single["distance"] > 1000)
    & True
]

# %%
fig, ax = plt.subplots()
ax.scatter(interaction_single.index, interaction_single["cor"], s=5, color="red")
ax.axhline(0)

# ax2 = ax.twinx()
plotdata_predictive["deltacor"].plot(ax=ax)
ax.axvline(position_oi)
# ax.set_xlim(-10000, 10000)
# ax.set_xlim(position_oi - 2000, position_oi + 2000)


# %% [markdown]
# ## Plot with HiC

# %%
norm = mpl.colors.CenteredNorm(0, np.abs(plotdata_interaction["cor"]).max())
cmap = mpl.cm.RdBu_r
fig, ax = plt.subplots()
ax.matshow(
    norm(plotdata_interaction.set_index(["window1", "window2"])["cor"].unstack()),
    cmap=cmap,
)
# %%
plotdata_interaction["deltacor_prod"] = (
    plotdata_interaction["deltacor1"] * plotdata_interaction["deltacor2"]
)
plotdata_interaction["deltacor_geom"] = np.sqrt(
    plotdata_interaction["deltacor1"] * plotdata_interaction["deltacor2"]
)
# %%
fig, ax = plt.subplots()
ax.scatter(plotdata_interaction["deltacor_geom"], plotdata_interaction["cor"])
# %%
import scipy.stats

lm = scipy.stats.linregress(
    plotdata_interaction.loc[~pd.isnull(plotdata_interaction["deltacor_geom"])][
        "deltacor_geom"
    ],
    plotdata_interaction.loc[~pd.isnull(plotdata_interaction["deltacor_geom"])]["cor"],
)
# %%
fig, ax = plt.subplots()
ax.scatter(np.sqrt(plotdata_interaction["deltacor_prod"]), plotdata_interaction["cor"])
ax.axline((0, lm.intercept), slope=lm.slope, color="red")
# %%
sc.pl.umap(transcriptome.adata, color=[gene])
# %%
