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
#
# ##
# scorer_folder = prediction.path / "scoring" / "nothing"
# nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)
# genes_all_oi = transcriptome.var.index[
#     (nothing_scoring.genescores.sel(phase="test").mean("model").mean("i")["cor"] > 0.1)
# ]
# transcriptome.var.loc[genes_all_oi].head(30)

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

symbol = "BCL2"
# symbol = "TNFAIP2"
# symbol = "KLF12"
# symbol = "CD14"
# symbol = "AXIN2"
symbol = "FLYWCH1"
# symbol = "PPP2R3C"
symbol = "BCL2"
symbol = "CCL4"
symbol = "CCL5"
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

# %%
window_scoring.genescores.sel(phase="validation").mean("gene").mean(
    "model"
).to_pandas().sort_values("deltacor").head(20)

# %% [markdown]
# ## Pairwindow
# %%
scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
interaction_file = scores_folder / "interaction.pkl"

interaction = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
interaction = interaction.rename(columns={0: "cor"})
assert len(interaction) > 0

# %%
interaction["effect_prod"] = interaction["effect1"] * interaction["effect2"]
fig, ax = plt.subplots()
ax.scatter(
    interaction.query("distance > 1000")["effect_prod"],
    interaction.query("distance > 1000")["cor"],
)


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

panel_predictive = chd.models.pred.plot.Predictivity(
    plotdata_predictive, window, panel_width
)
panel_predictive = main.add_under(panel_predictive, padding=0)

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

panel_interaction = main.add_under(chd.grid.Panel((panel_width, panel_width / 2)))
ax = panel_interaction.ax

# norm = mpl.colors.Normalize(0, plotdata_interaction["cor"].max())
# norm = mpl.colors.Normalize(-0.001, 0.001)
norm = mpl.colors.CenteredNorm(0, np.abs(plotdata_interaction["cor"]).max())

cmap = mpl.cm.RdBu_r

offsets = []
colors = []

plotdata_interaction["dist"] = (
    plotdata_interaction["window2"] - plotdata_interaction["window1"]
)
chd.plot.matshow45(
    ax,
    plotdata_interaction.query("dist < 0").set_index(["window1", "window2"])["cor"],
    cmap=cmap,
    norm=norm,
    radius=np.diff(plotdata_predictive.index)[0],
)

ax.set_ylim(-(window[1] - window[0]) / 2, 0)
ax.set_xlim(*window)

for x in np.linspace(*window, 16):
    x2 = x
    x1 = x2 + (window[0] - x2) / 2
    y2 = 0
    y1 = -(x2 - x1)

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

fig.plot()

# %%
sns.heatmap(
    plotdata_interaction.set_index(["window1", "window2"])["cor"].unstack(),
    vmin=-0.1,
    vmax=0.1,
    cmap="RdBu_r",
)

# %% [markdown]
# ## Focus on specific regions in a broken axis
#
# ### Get SNPs

# %%
motifscan_name = "gwas_immune"

motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
if "motifscan" not in globals():
    motifscan = chd.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
motifscan.n_motifs = len(motifs)
motifs["ix"] = np.arange(motifs.shape[0])

association = pd.read_pickle(motifscan_folder / "association.pkl")

# %%
gene_oi = transcriptome.gene_ix(symbol)

indptr_start = gene_oi * (window[1] - window[0])
indptr_end = (gene_oi + 1) * (window[1] - window[0])

indptr = motifscan.indptr[indptr_start:indptr_end]
motif_indices = motifscan.indices[indptr[0] : indptr[-1]]
position_indices = chd.utils.indptr_to_indices(indptr - indptr[0]) + window[0]

plotdata_snps = pd.DataFrame(
    {
        "position": position_indices,
        "motif": motifs.iloc[motif_indices].index.values,
    }
)
plotdata_snps = plotdata_snps.groupby("position").agg({"motif": list}).reset_index()

# %%
promoter = promoters.loc[gene]
for position in plotdata_snps["position"]:
    genome_position = (
        promoter["tss"] + position * promoter["strand"] + 1 * (promoter["strand"] == -1)
    )
    assoc = association.loc[
        (association["chr"] == promoter.chr) & (association["start"] == genome_position)
    ]
    if len(assoc) > 0:
        plotdata_snps.loc[plotdata_snps["position"] == position, "rsid"] = assoc[
            "snp"
        ].values[0]
        plotdata_snps.loc[plotdata_snps["position"] == position, "snp_main"] = assoc[
            "snp_main"
        ].values[0]
    # if True:  # len(assoc) > 0:
    #     print(
    #         (assoc["chr"].iloc[0] + ":" + str(int(assoc["start"].iloc[0])))
    #         + ">".join(assoc["alleles"].iloc[0].decode("utf-8").split(",")[:2][::-1]),
    #         assoc["snp"].iloc[0],
    #         assoc["snp_main"].iloc[0],
    #         assoc["alleles"].iloc[0],
    #         ", ".join(assoc["disease/trait"]),
    #     )

if len(plotdata_snps):
    plotdata_snps = plotdata_snps.loc[
        ~plotdata_snps["rsid"].isin(["rs142827445"])
    ].copy()

# %%
fig, ax = plt.subplots()
position_oi = (
    plotdata_interaction.groupby(["window1"])["cor"]
    .mean()
    .sort_values(ascending=False)
    .index[0]
)
plotdata_interaction.set_index(["window1", "window2"]).loc[position_oi]["cor"].plot(
    ax=ax
)
ax.set_xlim(-1000, 1000)


# %% [markdown]
# ### Plot

# %%
class TransformBroken:
    def __init__(self, regions, gap, resolution=None, width=None):
        """
        Transforms from data coordinates to (broken) data coordinates

        Parameters
        ----------
        regions : pd.DataFrame
            Regions to break
        resolution : float
            Resolution of the data to go from data to points
        gap : float
            Gap between the regions in points

        """

        regions["width"] = regions["end"] - regions["start"]
        regions["ix"] = np.arange(len(regions))

        if width is not None:
            resolution = (width - (gap * (len(regions) - 1))) / regions["width"].sum()
        regions["cumstart"] = (
            np.pad(np.cumsum(regions["width"])[:-1], (1, 0))
        ) + regions["ix"] * gap / resolution
        regions["cumend"] = (
            np.cumsum(regions["width"]) + regions["ix"] * gap / resolution
        )

        self.regions = regions
        self.resolution = resolution
        self.gap = gap

    def __call__(self, x):
        """
        Transform from data coordinates to (broken) data coordinates

        Parameters
        ----------
        x : float
            Position in data coordinates

        Returns
        -------
        float
            Position in (broken) data coordinates

        """

        assert isinstance(x, (int, float, np.ndarray, np.float64, np.int64))

        if isinstance(x, (int, float, np.float64, np.int64)):
            x = np.array([x])

        match = (x[:, None] >= self.regions["start"].values) & (
            x[:, None] <= self.regions["end"].values
        )

        argmax = np.argmax(
            match,
            axis=1,
        )
        allzero = (match == False).all(axis=1)

        # argmax[allzero] = np.nan

        y = self.regions.iloc[argmax]["cumstart"].values + (
            x - self.regions.iloc[argmax]["start"].values
        )
        y[allzero] = np.nan

        return y


regions = pd.DataFrame(
    {"start": [-52000, -10000, 50000], "end": [-40000, 10000, 80000]}
)
resolution = 1 / 1000
transform = TransformBroken(regions, 1, resolution)

y = transform(np.array([1000, 5000]))
assert np.allclose(y, np.array([24000, 28000]))

y = transform(np.array([-12000, 5000]))
assert np.isnan(y[0])

# %%
window_size = 100
windows = pd.DataFrame(
    {
        "window": plotdata_interaction["window1"].unique(),
        "start": plotdata_interaction["window1"].unique() - window_size / 2,
        "end": plotdata_interaction["window1"].unique() + window_size / 2,
    }
).set_index("window")
windows["region"] = np.pad(
    np.cumsum((~np.isclose(windows["end"].values[:-1], windows["start"].values[1:]))),
    (1, 0),
)

# select consecutive windows as regions
regions = windows.groupby("region").agg({"start": min, "end": max})

# combine regions that are close together
# regions["distance_til_next"] = np.pad(
#     regions["start"].values[1:] - regions["end"].values[:-1], (0, 1)
# )
# regions["joins_with_next"] = regions["distance_til_next"] < 500
# regions["new_region"] = np.pad(np.cumsum(~regions["joins_with_next"])[:-1], (1, 0))
# regions = regions.groupby("new_region").agg({"start": min, "end": max})

# remove regions that are too small
# regions["width"] = regions["end"] - regions["start"]
# regions = regions.loc[regions["width"] > 200]

# finalize regions
regions.index = pd.Series(range(len(regions)), name="region")
regions["width"] = regions["end"] - regions["start"]

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

# resolution = 1 / 4500
resolution = 1 / 1000
gap = 0.025
panel_width = regions["width"].sum() * resolution + len(regions) * gap

transform = TransformBroken(regions, width=panel_width, gap=gap)

plotdata_predictive = (
    window_scoring.genescores.sel(gene=gene).sel(phase="test").mean("model").to_pandas()
)
if "effect1" not in interaction.columns:
    interaction["effect1"] = 1.0
plotdata_predictive = (
    interaction.groupby("window1")[["deltacor1", "effect1"]]
    .first()
    .reset_index()
    .rename(columns={"deltacor1": "deltacor", "window1": "window", "effect1": "effect"})
    .set_index("window")
)
plotdata_predictive["position"] = plotdata_predictive.index

# labels of broken segments
panel_label = main.add_under(
    chd.predictive.plot.LabelBroken(regions, width=panel_width, gap=gap), padding=0
)

# gwas snps
if len(plotdata_snps):
    panel_snps = chdm.plotting.gwas.SNPsBroken(
        plotdata=plotdata_snps,
        regions=regions,
        width=panel_width,
        transform=transform,
    )
    panel_snps = main.add_under(panel_snps)

# genes
panel_genes = chdm.plotting.genes.GenesBroken(
    promoter, genome_folder, window, width=panel_width, gap=gap, regions=regions
)
panel_genes = main.add_under(panel_genes)

# predictive (deltacor)

panel_predictive = chd.predictive.plot.PredictiveBroken(
    plotdata=plotdata_predictive,
    width=panel_width,
    regions=regions,
    gap=gap,
    break_size=0,
    height=0.4,
)
panel_predictive = main.add_under(panel_predictive, padding=0)

##
panel_interaction = main.add_under(
    chd.grid.Panel((panel_width, panel_width / 2)), padding=0.0
)
ax = panel_interaction.ax

norm = mpl.colors.CenteredNorm(0, np.abs(plotdata_interaction["cor"]).max())
cmap = mpl.cm.RdBu_r

plotdata = plotdata_interaction.copy()
import itertools

# make plotdata, making sure we have all window combinations, otherwise nan
plotdata = (
    pd.DataFrame(
        itertools.combinations(windows.index, 2), columns=["window1", "window2"]
    )
    .set_index(["window1", "window2"])
    .join(plotdata_interaction.set_index(["window1", "window2"]))
)
plotdata.loc[np.isnan(plotdata["cor"]), "cor"] = 0.0
plotdata["dist"] = plotdata.index.get_level_values(
    "window2"
) - plotdata.index.get_level_values("window1")
plotdata["window1_broken"] = transform(
    plotdata.index.get_level_values("window1").values
)
plotdata["window2_broken"] = transform(
    plotdata.index.get_level_values("window2").values
)

plotdata = plotdata.loc[
    ~pd.isnull(plotdata["window1_broken"]) & ~pd.isnull(plotdata["window2_broken"])
]

chd.plot.matshow45(
    ax,
    plotdata.query("dist > 0").set_index(["window1_broken", "window2_broken"])["cor"],
    cmap=cmap,
    norm=norm,
    radius=50,
)
ax.invert_yaxis()

if symbol in ["BCL2"]:
    panel_interaction_legend = panel_interaction.add_inset(chd.grid.Panel((0.05, 0.8)))
    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=panel_interaction_legend.ax,
        orientation="vertical",
    )
    panel_interaction_legend.ax.set_ylabel(
        "Co-predictivity\n(cor $\\Delta$cor)",
        rotation=0,
        ha="left",
        va="center",
    )

ax.set_xlim([transform(regions["start"].min()), transform(regions["end"].max())])
fig.plot()


# %%
if symbol in ["BCL2", "TNFAIP2", "KLF12"]:
    manuscript.save_figure(fig, "5", "copredictivity_example_" + symbol)

# %%
for position in position_indices:
    if not np.isnan(transform(position)):
        genome_position = (
            promoter["tss"]
            + position * promoter["strand"]
            + 1 * (promoter["strand"] == -1)
        )
        assoc = association.loc[
            (association["chr"] == promoter.chr)
            & (association["start"] == genome_position)
        ]
        print(
            (assoc["chr"].iloc[0] + ":" + str(int(assoc["start"].iloc[0])))
            + ">".join(assoc["alleles"].iloc[0].decode("utf-8").split(",")[:2][::-1]),
            assoc["snp"].iloc[0],
            assoc["snp_main"].iloc[0],
            assoc["alleles"].iloc[0],
            ", ".join(assoc["disease/trait"]),
        )

# %%
# https://adastra.autosome.org/bill-cipher/snps/rs8086404
# https://adastra.autosome.org/bill-cipher/snps/rs3826622
# https://adastra.autosome.org/bill-cipher/snps/rs17758695

# %% [markdown]
# ## Cluster

# %%

import scipy.cluster.hierarchy as sch

x = plotdata_interaction.set_index(["window1", "window2"])["cor"].unstack()
x.values[np.isnan(x.values)] = 0
d = 1 - np.corrcoef(x)
L = sch.linkage(d, method="average")
ind = sch.fcluster(L, 0.5 * d.max(), "distance")
columns = [x.columns.tolist()[i] for i in list((np.argsort(ind)))]
x = x.reindex(columns, axis=1)
x = x.reindex(columns, axis=0)
sns.heatmap(x, vmin=-0.1, vmax=0.1, cmap="RdBu_r")


# %% [markdown]
# ## Plot with HiC
# import cooler
#
# cool_name = "rao_2014_1kb"
# step = 1000
# c = cooler.Cooler(str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000")
#
# promoter = promoters.loc[gene].copy()
# promoter["start"] = promoter["start"]  # - 200000
# promoter["end"] = promoter["end"]  # + 200000
#
# hic, bins_hic = chdm.hic.extract_hic(promoter, c=c)
# import itertools
#
# hic = (
#     pd.DataFrame(
#         index=pd.MultiIndex.from_frame(
#             pd.DataFrame(
#                 itertools.product(bins_hic.index, bins_hic.index),
#                 columns=["window1", "window2"],
#             )
#         )
#     )
#     .join(hic, how="left")
#     .fillna({"balanced": 0.0})
# )
# hic["distance"] = np.abs(
#     hic.index.get_level_values("window1") - hic.index.get_level_values("window2")
# )

# %% [markdown]
# Power law
#
# fig, ax = plt.subplots()
# plotdata = hic.query("distance > 1000")
# ax.scatter(np.log(plotdata["distance"]), np.log(plotdata["balanced"]))
# fig, ax = plt.subplots()
# plotdata = interaction.query("distance > 1000")
# ax.scatter(np.log(plotdata["distance"]), plotdata["cor"].abs())

# %%
fig, ax = plt.subplots()
bins_oi = bins_hic.query("start > 0000").query("end < 100000").index
# ax.matshow(np.log1p(hic.query("distance > 1000")["balanced"]).unstack())
ax.matshow(
    np.log1p(hic.query("distance > 1000")["balanced"])
    .unstack()
    .reindex(index=bins_oi, columns=bins_oi)
)
ax.set_yticks(np.arange(len(bins_oi)))
ax.set_yticklabels(bins_oi)
""


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

panel_predictive = chd.models.pred.plot.Predictivity(
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


# %% [markdown]
# ## Focus on a single window
#
# position_oi = 66148  # rs17758695, https://www.nature.com/articles/s41588-019-0362-6, https://www.sciencedirect.com/science/article/pii/S0002929721003037?via%3Dihub#app3
# position_oi = (
#     promoter["tss"] - 63235095
# )  # rs954954, https://www.sciencedirect.com/science/article/pii/S0002929721003037?via%3Dihub#app3
# position_oi = promoter["tss"] - 63248351  # rs9967405, random search
# position_oi = promoter["tss"] - 63239451  # rs9967405, random search
# position_oi = promoter["tss"] - 63121512  # rs9967405, random search
#
# windows_all = np.unique(interaction[["window1", "window2"]].values.flatten())
# window_oi = windows_all[np.argmin(np.abs(windows_all - position_oi))]
#
# interaction_single = interaction.query("(window1 == @window_oi)").set_index("window2")
# interaction_single = interaction_single.loc[
#     (interaction_single["lost1"] > 10)
#     & (interaction_single["lost2"] > 10)
#     # & (interaction_single["distance"] > 1000)
#     & True
# ]

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
