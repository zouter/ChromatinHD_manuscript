# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import polyptich as pp
pp.setup_ipython()

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

# fragments
promoter_name = "100k100k"
window = np.array([-1000000, 1000000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype",
    method="wilcoxon",
)
diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, None)
    .groupby("names")["logfoldchanges"]
    .max()
    .sort_values()
)

# %%
from chromatinhd_manuscript.designs import (
    dataset_splitter_method_combinations as design,
)

additive_method = "v20_initdefault"
additive_method = "v20"
nonadditive_method = "v22"

design = design.loc[
    (design["dataset"] == "pbmc10k")
    & (design["splitter"] == "permutations_5fold5repeat")
    & (design["promoter"] == promoter_name)
    & (design["method"].isin([additive_method, nonadditive_method]))
]

# %%
# load scoring

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
                .median("model")
                .sel(phase=["test"])
                .mean("phase")
                .sel(i=0)
                .to_pandas(),
                "method": design_row.method,
            }
        )
    )
scores = pd.concat(scores).reset_index()

# %%
plotdata = scores.set_index(["method", "gene"])["cor"].unstack("method")
plotdata["diff"] = plotdata[nonadditive_method] - plotdata[additive_method]
plotdata["gene"] = transcriptome.var["symbol"]

# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))

norm = mpl.colors.CenteredNorm(halfrange=0.1)
cmap = mpl.cm.get_cmap("RdBu_r")
ax.scatter(
    plotdata[additive_method],
    plotdata[nonadditive_method],
    c=cmap(norm(plotdata["diff"])),
    s=0.1,
)

symbols_oi = [
    "CD74",
    "TNFAIP2",
    "KLF12",
    "BCL2",
]
offsets = {
    "CD74": (-0.1, 0.05),
    "TNFAIP2": (-0.05, 0.1),
    "KLF12": (-0.1, 0.0),
    "BCL2": (-0.1, 0.1),
}
genes_oi = transcriptome.gene_id(symbols_oi)
texts = []
for symbol_oi, gene_oi in zip(symbols_oi, genes_oi):
    x, y = (
        plotdata.loc[gene_oi, additive_method],
        plotdata.loc[gene_oi, nonadditive_method],
    )
    text = ax.annotate(
        symbol_oi,
        (x, y),
        xytext=(x + offsets[symbol_oi][0], y + offsets[symbol_oi][1]),
        ha="right",
        va="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="black"),
        # bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.5),
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFFAA"),
            mpl.patheffects.Normal(),
        ]
    )
    texts.append(text)

cutoff = 0.05
percs = (
    (plotdata["diff"] > cutoff).mean(),
    (plotdata["diff"] < -cutoff).mean(),
    1 - (np.abs(plotdata["diff"]) > cutoff).mean(),
)
ax.axline((cutoff, 0), (0.80 + cutoff, 0.8), color="black", linestyle="--", lw=0.5)
ax.axline((-cutoff, 0), (0.8 - cutoff, 0.8), color="black", linestyle="--", lw=0.5)
bbox = dict(boxstyle="round", fc="white", ec="black", lw=0.5)
ax.annotate(
    f"{percs[0]:.1%}",
    (0.03, 0.97),
    (0.0, 0.0),
    textcoords="offset points",
    ha="left",
    va="top",
)
ax.annotate(
    f"{percs[1]:.1%}",
    (0.97, 0.03),
    (0.0, 0.0),
    textcoords="offset points",
    ha="right",
    va="bottom",
)
text = ax.annotate(
    f"{percs[2]:.1%}",
    (0.97, 0.97),
    (0.0, 0.0),
    textcoords="offset points",
    ha="right",
    va="top",
)
text.set_path_effects(
    [mpl.patheffects.Stroke(linewidth=3, foreground="white"), mpl.patheffects.Normal()]
)

polygon = mpl.patches.Polygon(
    [
        (cutoff, 0),
        (1.0 + cutoff, 1.0),
        (1.0 - cutoff, 1.0),
        (-cutoff, 0),
        (cutoff, 0),
    ],
    closed=True,
    fill=True,
    edgecolor="black",
    lw=0.5,
    facecolor="#00000033",
    zorder=0,
)
# ax.add_patch(polygon)
polygon = mpl.patches.Polygon(
    [
        (1.0, 1.0 + cutoff),
        (0, 1),
        (0, cutoff),
    ],
    fill=True,
    facecolor="#00000022",
    lw=0.5,
    zorder=0,
)
ax.add_patch(polygon)

ax.set_xlabel("cor additive model")
ax.set_ylabel("cor\nnon-additive\nmodel", rotation=0, ha="right", va="center")
ax.set_aspect(1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
manuscript.save_figure(fig, "5", "positional_additive_vs_nonadditive", dpi=300)

# %%
# plotdata.query("v20 > 0.4").sort_values("diff", ascending=False).head(10)
plotdata.sort_values("diff", ascending=False).head(10)

# %%
sc.pl.umap(
    transcriptome.adata,
    color=plotdata.sort_values("diff", ascending=False).head(10).index,
    ncols=5,
    cmap="viridis",
    use_raw=False,
)


# %%
cutoff = 0.05
(plotdata["diff"] > cutoff).mean(), (plotdata["diff"] < -cutoff).mean(), 1 - (
    np.abs(plotdata["diff"]) > cutoff
).mean()
# %%
plotdata.to_pickle(chd.get_output() / "additive_vs_nonadditive_gene_scores.pkl")

# %%
transcriptome.var["n_fragments"] = torch.bincount(fragments.mapping[:, 1])
transcriptome.var["n_expressed"] = (transcriptome.X.dense() > 0).sum(0)

# %%
plotdata["n_fragments"] = transcriptome.var["n_fragments"]
plotdata["n_expressed"] = transcriptome.var["n_expressed"]
plotdata["dispersions_norm"] = transcriptome.var["dispersions_norm"]
plotdata["diffexp"] = diffexp

# %%
def smooth_spline_fit(x, y, x_smooth):
    import rpy2.robjects as robjects

    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)

    r_smooth_spline = robjects.r[
        "smooth.spline"
    ]  # extract R function# run smoothing function
    spline1 = r_smooth_spline(x=r_x, y=r_y)
    ySpline = np.array(
        robjects.r["predict"](spline1, robjects.FloatVector(x_smooth)).rx2("y")
    )

    return ySpline


# %%
plotdata_oi = plotdata
plotdata_oi["means"] = np.log(transcriptome.var["means"])
plotdata_oi["log1p_n_fragments"] = np.log1p(plotdata_oi["n_fragments"])
plotdata_oi["log1p_n_expressed"] = np.log1p(plotdata_oi["n_expressed"])
plotdata_oi["log_dispersions_norm"] = np.log1p(plotdata_oi["dispersions_norm"])

plotdata_oi["total_rank"] = (
    plotdata_oi["means"].rank()
    + plotdata_oi["log_dispersions_norm"].rank()
    + plotdata_oi["log1p_n_fragments"].rank()
    + plotdata_oi["log1p_n_expressed"].rank()
    + plotdata_oi["diffexp"].rank()
    + 0
    + 0
    + 0
    + 0
).rank()

# variable = "log1p_n_expressed"
# variable = "means"
# variable = additive_method
# variable = "log1p_n_fragments"
variable = "dispersions_norm"
variable = "log_dispersions_norm"
variable = "diffexp"
# variable = "important"
variable = "total_rank"
x_smooth = np.linspace(plotdata_oi[variable].min(), plotdata_oi[variable].max(), 1000)
x = plotdata_oi[variable].values
# y_smooth = smooth_spline_fit(x, plotdata_oi[additive_method].values, x_smooth)

np.corrcoef(
    plotdata_oi["v20"].values,
    plotdata_oi["total_rank"].values,
)

# %%
fig, ax = plt.subplots(figsize=(2.0, 2.0))

ax.scatter(
    x,
    plotdata_oi["diff"],
    s=1,
    c=cmap(norm(plotdata_oi["diff"])),
    alpha=0.5,
)
# ax.plot(
#     x_smooth,
#     y_smooth,
# )
# ax.set_xscale("log")
ax.set_xlabel(variable)
ax.set_ylabel("cor\nnon-additive\nmodel", rotation=0, ha="right", va="center")

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / design_row.dataset
    / design_row.promoter
    / design_row.splitter
    / "v20_initdefault"
)
window_scores = pd.read_pickle(
    prediction.path / "scoring" / "windowsize_gene" / "window_scores.pkl"
)

# %%
window_scores["important"] = window_scores["deltacor"] < -0.00

plotdata["important"] = window_scores.groupby("gene")["important"].mean()

# %%
