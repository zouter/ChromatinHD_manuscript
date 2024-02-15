# ---
# jupyter:
#   jupytext:
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
import IPython

if IPython.get_ipython():
    IPython.get_ipython().magic("load_ext autoreload")
    IPython.get_ipython().magic("autoreload 2")
    IPython.get_ipython().magic("config InlineBackend.figure_format='retina'")

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

import tqdm.auto as tqdmpbmc10k_eqtl

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    default="pbmc10k",
    # default = "GSE198467_H3K27ac",
    # default = "brain",
    # default = "pbmc10k"
)
parser.add_argument("--promoter_name", default="10k10k")
# parser.add_argument("--latent_name", default = "celltype")
parser.add_argument("--latent_name", default="leiden_0.1")
parser.add_argument("--method_name", default="v9_128-64-32")

try:
    get_ipython().__class__.__name__
    in_jupyter = True
except:
    in_jupyter = False
parameters = vars(parser.parse_args("" if in_jupyter else None))
dataset_name = parameters["dataset_name"]
promoter_name = parameters["promoter_name"]
latent_name = parameters["latent_name"]
method_name = parameters["method_name"]

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / dataset_name


# %% [markdown]
# ### Load data

# %%
class Prediction(chd.flow.Flow):
    pass


prediction = Prediction(
    chd.get_output()
    / "prediction_likelihood"
    / dataset_name
    / promoter_name
    / latent_name
    / method_name
)
# model = chd.load((prediction.path / "model_0.pkl").open("rb"))

# %%
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

probs_diff = probs - probs.mean(1, keepdims=True)

# %%
design["gene_ix"] = design["gene_ix"]

# %%
window = {"10k10k": np.array([-10000, 10000])}[promoter_name]
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])
transcriptome.adata.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# Interpolate probs for individual positions

# %%
x = (
    (design["coord"].values)
    .astype(int)
    .reshape(
        (
            len(design["gene_ix"].cat.categories),
            len(design["active_latent"].cat.categories),
            len(design["coord"].cat.categories),
        )
    )
)
desired_x = torch.arange(*window)
probs_interpolated = chd.utils.interpolate_1d(
    desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
).numpy()

# %% [markdown]
# ## Load slices

# %%
prob_cutoff = np.log(1.0)

# %%
scores_dir = prediction.path / "scoring" / "significant_up"
pureregionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %% [markdown]
# ### Relative to peaks

# %%
import chromatinhd.peakcounts

# peakcaller = "cellranger"; peakcaller_label = "Cellranger"
# peakcaller = "macs2"
# peakcaller = "macs2_improved";peakcaller_label = "MACS2"
# peakcaller = "encode_screen";peakcaller_label = "ENCODE SCREEN"

peakcaller = "macs2_leiden_0.1_merged"

# peakcaller = "rolling_500";peakcaller_label = "Sliding window 500"

# peakcaller = "rolling_100"

# diffexp = "signac"
diffexp = "scanpy"
# diffexp = "scanpy_wilcoxon"

# %%
scores_dir = (
    chd.get_output()
    / "prediction_differential"
    / dataset_name
    / promoter_name
    / latent_name
    / diffexp
    / peakcaller
)
peakresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %%
peakresult.get_slicescores()

# %% [markdown]
# ## Classification

# %%
import chromatinhd.slicetypes

chromatinhd.slicetypes.types_info

# %% [markdown]
# ### Classify

# %%
slicetopologies = pd.concat(
    [
        pureregionresult.get_slicetopologies(probs_interpolated),
        pureregionresult.get_sliceaverages(probs_interpolated),
        pureregionresult.get_slicescores(),
    ],
    axis=1,
)

# %%
sns.histplot(slicetopologies["balances_raw"], label="balances_raw")
# sns.histplot(slicetopologies["prominence"], label="prominence")
# sns.histplot(slicetopologies["dominance"], label="dominance")
# sns.histplot(slicetopologies["shadow"], label="shadow")

# %%
slicetopologies["flank"] = slicetopologies["prominence"] <= 0.5
slicetopologies["flank_rank"] = slicetopologies["prominence"].rank()
slicetopologies["hill"] = slicetopologies["dominance"] <= 0.5
slicetopologies["hill_rank"] = slicetopologies["dominance"].rank()
slicetopologies["chain"] = (slicetopologies["length"] > 800) & (
    slicetopologies["std"] > np.log(1.5)
)
slicetopologies["chain_rank"] = (-slicetopologies["n_subpeaks"]).rank()
slicetopologies["canyon"] = slicetopologies["balances_raw"] < 2 / 3
slicetopologies["canyon_rank"] = (-slicetopologies["balances_raw"]).rank()
slicetopologies["ridge"] = (slicetopologies["length"] > 800) & (
    slicetopologies["std"] < np.log(1.5)
)
slicetopologies["ridge_rank"] = slicetopologies["shadow"].rank()
slicetopologies["volcano"] = slicetopologies["max_baseline"] < np.log(1.0)
slicetopologies["volcano_rank"] = slicetopologies["max_baseline"].rank()

slicetopologies["peak_rank"] = 1.0

# %%
slicetopologies["type"] = "peak"
slicetopologies.loc[slicetopologies["volcano"], "type"] = "volcano"
slicetopologies.loc[slicetopologies["hill"], "type"] = "hill"
slicetopologies.loc[slicetopologies["canyon"], "type"] = "canyon"
slicetopologies.loc[slicetopologies["flank"], "type"] = "flank"
slicetopologies.loc[slicetopologies["ridge"], "type"] = "ridge"
slicetopologies.loc[slicetopologies["chain"], "type"] = "chain"
slicetopologies["type"] = pd.Categorical(
    slicetopologies["type"], categories=chd.slicetypes.types_info.index
)

# %%
slicetopologies["type"].value_counts()

# %%
slicetopologies["loglength"] = np.log(slicetopologies["length"])

# %% [markdown]
# ### Store for David

# %%
slicetopologies_mapped = slicetopologies.copy()
slicetopologies_mapped["gene"] = promoters.index[slicetopologies_mapped["gene_ix"]]
slicetopologies_mapped["cluster"] = cluster_info.index[
    slicetopologies_mapped["cluster_ix"]
]

# %%
slicetopologies_mapped["start"] = (
    promoters.loc[slicetopologies_mapped.gene, "tss"].values
    + (slicetopologies["start"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == 1).values
    - (slicetopologies["end"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == -1).values
)

slicetopologies_mapped["end"] = (
    promoters.loc[slicetopologies_mapped.gene, "tss"].values
    + (slicetopologies["end"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == 1).values
    - (slicetopologies["start"] + window[0])
    * (promoters.loc[slicetopologies_mapped.gene, "strand"] == -1).values
)
slicetopologies_mapped["chr"] = promoters.loc[slicetopologies_mapped.gene, "chr"].values

# %%
scores_dir = prediction.path / "scoring" / "significant_up"
slicetopologies_mapped.to_csv(scores_dir / "slicetopologies.csv")

# %%
slicetopologies_mapped[["chr", "start", "end", "cluster", "type"]]

# %%
# from_file = scores_dir / "slicetopologies.csv"
# to_output = pathlib.Path("/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output")
# to_file = to_output / from_file.relative_to(chd.get_output())

# to_file.parent.mkdir(parents = True, exist_ok = True)

# import shutil
# shutil.copy(from_file, to_file)

# %% [markdown]
# ### 2D visualization of slices

# %%
features = [
    "prominence",
    "dominance",
    "max",
    "std",
    "loglength",
    "average",
    "max_lfc",
    "average_lfc",
    "max_baseline",
    "average_baseline",
    "log1p_n_subpeaks",
    "shadow",
    "balances_raw",
    "balance",
]

# %%
X = slicetopologies[features].values
X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True))

# %%
import sklearn.decomposition

pca = sklearn.decomposition.PCA(5)
X_pca = pca.fit_transform(X)

# %%
fig, ax = plt.subplots()
ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=chd.slicetypes.types_info.loc[slicetopologies["type"], "color"],
    s=1,
)

# %%
wrap = chd.grid.Wrap()
fig = chd.grid.Figure(wrap)

for feature in features:
    ax = wrap.add(chd.grid.Ax((3, 3)))
    ax = ax.ax
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=slicetopologies[feature], s=1)
    ax.set_title(feature)
fig.plot()

# %%
import umap

umap = umap.UMAP(n_neighbors=30)

# %%
X_umap = umap.fit_transform(X_pca)

# %%
wrap = chd.grid.Wrap()
fig = chd.grid.Figure(wrap)

for feature in features:
    ax = wrap.add(chd.grid.Ax((3, 3)))
    ax = ax.ax
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=slicetopologies[feature], s=1)
    ax.set_title(feature)
fig.plot()

# %%
fig, ax = plt.subplots()
ax.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=chd.slicetypes.types_info.loc[slicetopologies["type"], "color"],
    s=1,
)

# %% [markdown]
# ### Examples

# %%
# window_oi = np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
# window_oi = np.array([500, 1000])

# %%
types_info = chd.slicetypes.types_info
types_info["ix"] = np.arange(len(types_info))

# %%
import chromatinhd.grid

main = chd.grid.Grid(len(types_info), 2, padding_width=0.1, padding_height=0.1)
fig = chd.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005

panel_height = 0.5

total_width_cutoff = 10

for slicetype, slicetype_info in types_info.iterrows():
    slicetype_ix = slicetype_info["ix"]
    ax_row_title = main[slicetype_ix, 0] = chd.grid.Ax((panel_height, panel_height))
    ax = ax_row_title.ax
    ax.axis("off")
    ax.text(
        -0.2,
        0.5,
        chd.slicetypes.types_info.loc[slicetype, "label"],
        va="center",
        ha="right",
    )
    chd.slicetypes.plot_type(ax, slicetype)

    wrap = main[slicetype_ix, 1] = chd.grid.Wrap(padding_width=0.1, ncol=10)

    width_so_far = 0

    slices_oi_ranked = slicetopologies.query(
        "(type == @slicetype) & (length > 100)"
    ).sort_values(slicetype + "_rank")
    slices_oi_ranked = slicetopologies.query("(type == @slicetype) & (length > 100)")

    for i in range(10):
        slice_oi = slices_oi_ranked.iloc[i]

        expanded_slice_oi = slice_oi.copy()
        expanded_slice_oi["start"] = np.clip(
            slice_oi["start"] - 800, *(window - window[0])
        )
        expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 800, *(window - window[0]))

        window_oi = (
            np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
        )

        gene_oi = expanded_slice_oi["gene_ix"]
        cluster_info_oi = cluster_info.iloc[[expanded_slice_oi["cluster_ix"]]]

        plotdata_atac = (
            design.query("gene_ix == @gene_oi")
            .copy()
            .rename(columns={"active_latent": "cluster"})
            .set_index(["coord", "cluster"])
            .drop(columns=["batch", "gene_ix"])
        )
        plotdata_atac["prob"] = probs[gene_oi].flatten()
        plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

        plotdata_atac["prob"] = (
            plotdata_atac["prob"]
            - np.log(
                plotdata_atac.reset_index()
                .groupby(["cluster"])
                .apply(
                    lambda x: np.trapz(
                        np.exp(x["prob"]),
                        x["coord"].astype(float) / (window[1] - window[0]),
                    )
                )
            ).mean()
        )
        plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

        resolution = 0.0005
        panel_width = (window_oi[1] - window_oi[0]) * resolution

        width_so_far += panel_width

        if width_so_far >= total_width_cutoff:
            break

        # differential atac
        wrap_differential = chd.models.diff.plot.Differential(
            plotdata_atac,
            plotdata_atac_mean,
            cluster_info_oi,
            window_oi,
            panel_width,
            panel_height,
            padding_height=padding_height,
            title=False,
        )
        wrap.add(wrap_differential)

        ax = wrap_differential.elements[0].ax

        start = slice_oi["start"] + window[0]
        end = slice_oi["end"] + window[0]
        ax.axvspan(start, end, fc="#0003", lw=0)

        # gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
        gene_label = gene_oi
        cluster_label = cluster_info.query("dimension == @slice_oi.cluster_ix")[
            "label"
        ][0]
        position_label = str(
            int(slice_oi["start"] + slice_oi["length"] / 2) + window[0]
        )
        extra = str(slice_oi.name)
        text = ax.annotate(
            f"$\\it{{{gene_label}}}$ $\\bf{{{cluster_label}}}$ {extra}",
            (0, 1),
            (2, 2),
            va="bottom",
            ha="left",
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )

        trans = mpl.transforms.blended_transform_factory(
            y_transform=ax.transAxes, x_transform=ax.transData
        )
        text = ax.annotate(
            f"{start:+}",
            (start, 1),
            (-2, -2),
            va="top",
            ha="right",
            xycoords=trans,
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )
        text = ax.annotate(
            f"{end:+}",
            (end, 1),
            (2, -2),
            va="top",
            ha="left",
            xycoords=trans,
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )

fig.plot()

# %%
wrap = chd.grid.WrapAutobreak(padding_width=0.1, max_width=5, padding_height=0.25)

main = wrap
fig = chd.grid.Figure(main)

slicetype = "canyon"

slicetype_ix = slicetype_info["ix"]

for i in range(30):
    main.align()
    if main.nrow >= 7:
        break

    slice_oi = slicetopologies.query("(type == @slicetype) & (length > 100)").iloc[i]

    expanded_slice_oi = slice_oi.copy()
    expanded_slice_oi["start"] = np.clip(slice_oi["start"] - 800, *(window - window[0]))
    expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 800, *(window - window[0]))

    window_oi = (
        np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]
    )

    gene_oi = expanded_slice_oi["gene_ix"]
    cluster_info_oi = cluster_info.iloc[[expanded_slice_oi["cluster_ix"]]]

    plotdata_atac = (
        design.query("gene_ix == @gene_oi")
        .copy()
        .rename(columns={"active_latent": "cluster"})
        .set_index(["coord", "cluster"])
        .drop(columns=["batch", "gene_ix"])
    )
    plotdata_atac["prob"] = probs[gene_oi].flatten()
    plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

    plotdata_atac["prob"] = (
        plotdata_atac["prob"]
        - np.log(
            plotdata_atac.reset_index()
            .groupby(["cluster"])
            .apply(
                lambda x: np.trapz(
                    np.exp(x["prob"]),
                    x["coord"].astype(float) / (window[1] - window[0]),
                )
            )
        ).mean()
    )
    plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

    resolution = 0.0005
    panel_width = (window_oi[1] - window_oi[0]) * resolution

    # differential atac
    wrap_differential = chd.models.diff.plot.Differential(
        plotdata_atac,
        plotdata_atac_mean,
        cluster_info_oi,
        window_oi,
        panel_width,
        panel_height,
        padding_height=padding_height,
        title=False,
    )
    wrap.add(wrap_differential)

    ax = wrap_differential.elements[0].ax

    start = slice_oi["start"] + window[0]
    end = slice_oi["end"] + window[0]
    ax.axvspan(start, end, fc="#0003", lw=0)

    gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
    cluster_label = cluster_info.query("dimension == @slice_oi.cluster_ix")["label"][0]
    position_label = str(int(slice_oi["start"] + slice_oi["length"] / 2) + window[0])
    extra = str(slice_oi.name)
    text = ax.annotate(
        f"$\\it{{{gene_label}}}$ $\\bf{{{cluster_label}}}$ {extra}",
        (0, 1),
        (2, 2),
        va="bottom",
        ha="left",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=6,
        color="#999",
        zorder=200,
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(foreground="white", linewidth=2),
            mpl.patheffects.Normal(),
        ]
    )

    trans = mpl.transforms.blended_transform_factory(
        y_transform=ax.transAxes, x_transform=ax.transData
    )
    text = ax.annotate(
        f"{start:+}",
        (start, 1),
        (-2, -2),
        va="top",
        ha="right",
        xycoords=trans,
        textcoords="offset points",
        fontsize=6,
        color="#999",
        zorder=200,
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(foreground="white", linewidth=2),
            mpl.patheffects.Normal(),
        ]
    )
    text = ax.annotate(
        f"{end:+}",
        (end, 1),
        (2, -2),
        va="top",
        ha="left",
        xycoords=trans,
        textcoords="offset points",
        fontsize=6,
        color="#999",
        zorder=200,
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(foreground="white", linewidth=2),
            mpl.patheffects.Normal(),
        ]
    )

fig.plot()

# %% [markdown]
# ### Positions

# %%
fig, axes = plt.subplots(
    chd.slicetypes.types_info.shape[0],
    1,
    figsize=(5, chd.slicetypes.types_info.shape[0] * 0.5),
    sharex=True,
    gridspec_kw={"hspace": 0},
)
nbins = 100
density_lim = 1 / ((window[1] - window[0]) / nbins) / 25
for ax, (type, plotdata) in zip(axes, slicetopologies.groupby("type")):
    color = chd.slicetypes.types_info.loc[type, "color"]
    sns.histplot(
        plotdata["mid"],
        bins=nbins,
        stat="density",
        label=type,
        lw=0,
        ax=ax,
        color=color,
    )
    # ax.text(0.02, 0.95, type, color = color, transform = ax.transAxes, va = "top", ha = "left")
    ax.set_yticks([])
    ax.set_xlim(*window)

    w, h = fig.transFigure.inverted().transform([[1, 1]])[0] * 20
    x, y = fig.transFigure.inverted().transform(ax.transAxes.transform([0.1, 0.9]))
    y -= h

    l = ax.yaxis.get_label()
    inset = chd.plot.replace_patch(ax, l, points=25, ha="right")
    l.set_visible(False)
    inset.axis("off")
    chd.slicetypes.plot_type(inset, type)
    ax.set_ylim(0, density_lim)
    ax.axvline(0, dashes=(2, 2), lw=1, color="#333")
axes[-1].set_xlabel("Distance from TSS")
axes[-1].set_xlabel("    ← upstream    TSS    downstream →")

# %% [markdown]
# ### Frequencies

# %%
fig, ax = plt.subplots(figsize=(1.5, 3))

plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby("type").size(),
        "n_positions": slicetopologies.groupby("type")["length"].sum(),
    }
)
plotdata["rel_n_regions"] = plotdata["n_regions"] / plotdata["n_regions"].sum()
plotdata["cum_n_regions"] = (
    np.cumsum(plotdata["rel_n_regions"]) - plotdata["rel_n_regions"]
)
plotdata["rel_n_positions"] = plotdata["n_positions"] / plotdata["n_positions"].sum()
plotdata["cum_n_positions"] = (
    np.cumsum(plotdata["rel_n_positions"]) - plotdata["rel_n_positions"]
)

ax.bar(
    0,
    plotdata["rel_n_regions"],
    bottom=plotdata["cum_n_regions"],
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
    lw=0,
)
ax.bar(
    1,
    plotdata["rel_n_positions"],
    bottom=plotdata["cum_n_positions"],
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
    lw=0,
)

texts = []
for type, plotdata_type in plotdata.iterrows():
    color = chd.slicetypes.types_info.loc[type, "color"]
    text = ax.text(
        -0,
        plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"] / 2,
        f"{plotdata_type['rel_n_regions']:.1%}",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground=color),
            mpl.patheffects.Normal(),
        ]
    )
    # texts.append(text)

    text = ax.text(
        1.0,
        plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"] / 2,
        f"{plotdata_type['rel_n_positions']:.1%}",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    text.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=3, foreground=color),
            mpl.patheffects.Normal(),
        ]
    )
    # texts.append(text)
    # texts_left.append(ax.text(-0.5, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%}", ha = "right"))
    # texts_right.append(ax.text(2.5, plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%}", ha = "left"))
    # texts.append(ax.text(0, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%} {type}", ha = "center", va = "center"))
    # ax.text(1, plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%} {type}", ha = "center", va = "center")

ax.set_xticks([0, 1])
ax.set_xticklabels(["regions", "positions"])
ax.set_ylim(1, 0)
ax.set_xlim(-0.4, 1.4)
ax.set_yticks([0, 0.5, 1])
sns.despine(ax=ax)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.xaxis.tick_top()
# adjustText.adjust_text(texts, autoalign=False, only_move = {"text":"y"}, ha = "center", lim = 3000)
# adjustText.adjust_text(texts_right, autoalign=False, only_move = {"text":"y"}, ha = "left")

# %%
plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby(["type", "cluster_ix"]).size(),
        "n_positions": slicetopologies.groupby(["type", "cluster_ix"])["length"].sum(),
    }
)

# %%
plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby(["cluster_ix", "type"]).size(),
        "n_positions": slicetopologies.groupby(["cluster_ix", "type"])["length"].sum(),
    }
)
plotdata["rel_n_regions"] = (
    plotdata["n_regions"] / plotdata.groupby("cluster_ix")["n_regions"].sum()
)
plotdata["cum_n_regions"] = (
    plotdata.groupby("cluster_ix")["rel_n_regions"].cumsum() - plotdata["rel_n_regions"]
)
plotdata["rel_n_positions"] = (
    plotdata["n_positions"] / plotdata.groupby("cluster_ix")["n_positions"].sum()
)
plotdata["cum_n_positions"] = (
    plotdata.groupby("cluster_ix")["rel_n_positions"].cumsum()
    - plotdata["rel_n_positions"]
)

plotdata_grouped = plotdata.groupby("cluster_ix")

fig, axes = plt.subplots(
    ncols=len(plotdata_grouped), figsize=(1.5 * len(plotdata_grouped), 3), sharey=True
)

for ax, (cluster_ix, plotdata_cluster) in zip(axes, plotdata_grouped):
    ax.bar(
        0,
        plotdata_cluster["rel_n_regions"],
        bottom=plotdata_cluster["cum_n_regions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.bar(
        1,
        plotdata_cluster["rel_n_positions"],
        bottom=plotdata_cluster["cum_n_positions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.set_title(cluster_info.iloc[cluster_ix]["label"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["regions", "positions"])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, 1.4)

    ax.set_yticks([0, 0.5, 1])
    sns.despine(ax=ax)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# %%
slicetopologies["position_group"] = np.array(["upstream", "promoter", "downstream"])[
    np.argmax(
        np.vstack(
            [
                slicetopologies["mid"] < -1000,
                slicetopologies["mid"] < 1000,
                slicetopologies["mid"] <= window[1],
            ]
        ),
        0,
    )
]

position_group_info = pd.DataFrame(
    [
        ["upstream", "Upstream -10kb→-1kb"],
        ["promoter", "Promoter -1kb→+1kb"],
        ["downstream", "Downstream +1kb→+10kb"],
    ],
    columns=["position_group", "label"],
).set_index("position_group")
position_group_info.index = pd.Categorical(
    position_group_info.index, categories=position_group_info.index
)

# %%
grouper = "position_group"
group_info = position_group_info

# %%
import textwrap

# %%
slicetopologies[grouper] = pd.Categorical(
    slicetopologies[grouper], categories=group_info.index
)
plotdata = pd.DataFrame(
    {
        "n_regions": slicetopologies.groupby([grouper, "type"]).size(),
        "n_positions": slicetopologies.groupby([grouper, "type"])["length"].sum(),
    }
)
plotdata["rel_n_regions"] = (
    plotdata["n_regions"] / plotdata.groupby(grouper)["n_regions"].sum()
)
plotdata["cum_n_regions"] = (
    plotdata.groupby(grouper)["rel_n_regions"].cumsum() - plotdata["rel_n_regions"]
)
plotdata["rel_n_positions"] = (
    plotdata["n_positions"] / plotdata.groupby(grouper)["n_positions"].sum()
)
plotdata["cum_n_positions"] = (
    plotdata.groupby(grouper)["rel_n_positions"].cumsum() - plotdata["rel_n_positions"]
)

plotdata_grouped = plotdata.groupby(grouper)

fig, axes = plt.subplots(
    ncols=len(plotdata_grouped), figsize=(1.5 * len(plotdata_grouped), 3), sharey=True
)

for ax, (group_index, plotdata_cluster) in zip(axes, plotdata_grouped):
    ax.bar(
        0,
        plotdata_cluster["rel_n_regions"],
        bottom=plotdata_cluster["cum_n_regions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.bar(
        1,
        plotdata_cluster["rel_n_positions"],
        bottom=plotdata_cluster["cum_n_positions"],
        color=chd.slicetypes.types_info.loc[
            plotdata_cluster.index.get_level_values("type"), "color"
        ],
        lw=0,
    )
    ax.set_title(
        "\n".join(textwrap.wrap(group_info.loc[group_index]["label"], width=10))
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["regions", "positions"])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, 1.4)

    ax.set_yticks([0, 0.5, 1])
    sns.despine(ax=ax)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# %% [markdown]
# ### Accessibility summary

# %%
import scipy.stats
import chromatinhd.slicetypes

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(2, 3), sharex=True)
sns.violinplot(
    data=slicetopologies,
    y="average_lfc",
    x="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax1,
    linewidth=0,
)
sns.violinplot(
    data=slicetopologies,
    y="max_lfc",
    x="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax2,
    linewidth=0,
)
sns.stripplot(
    data=slicetopologies,
    y="length",
    x="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax3,
    linewidth=0,
    s=1,
)

ax1.set_ylim(0, 2)
ax2.set_ylim(0, 3)
ax3.set_yscale("log")

chd.slicetypes.label_axis(ax2, ax2.xaxis)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 1.5), sharey=True)
sns.ecdfplot(
    data=slicetopologies,
    x="average_lfc",
    hue="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax1,
    legend=False,
)
ax1.set_xlim(0, 2)
sns.ecdfplot(
    data=slicetopologies,
    x="max_lfc",
    hue="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax2,
    legend=False,
)
ax2.set_xlim(0, 3)
sns.ecdfplot(
    data=slicetopologies,
    x="average",
    hue="type",
    palette=chd.slicetypes.types_info["color"].to_dict(),
    ax=ax3,
    legend=False,
)

# chd.slicetypes.label_axis(ax2, ax2.xaxis)

# %%
slice_oi = slicetopologies.query("type == 'ridge'").iloc[3]

expanded_slice_oi = slice_oi.copy()
expanded_slice_oi["start"] = np.clip(slice_oi["start"] - 1000, *(window - window[0]))
expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 1000, *(window - window[0]))

# %%
probs_interpolated_mean = probs_interpolated.mean(1)

# %%
# slicetopologies.query("gene_ix == 13")

# %%
transcriptome.var.iloc[expanded_slice_oi["gene_ix"]]

# %%
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(
    np.arange(expanded_slice_oi["start"], expanded_slice_oi["end"]),
    np.exp(
        probs_interpolated[
            expanded_slice_oi["gene_ix"],
            expanded_slice_oi["cluster_ix"],
            expanded_slice_oi["start"] : expanded_slice_oi["end"],
        ]
    ),
)
ax.plot(
    np.arange(expanded_slice_oi["start"], expanded_slice_oi["end"]),
    np.exp(
        probs_interpolated_mean[
            expanded_slice_oi["gene_ix"],
            expanded_slice_oi["start"] : expanded_slice_oi["end"],
        ]
    ),
)
ax.axvline(slice_oi["start"])
ax.axvline(slice_oi["end"])
ax.set_ylim(0)
# ax.axvline(expanded_slice_oi["start"]
# plt.plot(np.exp(probs_mean[slice_oi["gene_ix"], slice_oi["start"]:slice_oi["end"]]))

# %% [markdown]
# ## Overlap with binding sites

# %% [markdown]
# ### Gather ChIP-seq sites
# %%
sites_file = chd.get_output() / "bed/gm1282_tf_chipseq_filtered" / "sites.csv"

sites_file.parent.mkdir(exist_ok = True, parents = True)
if not sites_file.exists():
    !rsync -a --progress wsaelens@updeplasrv6.epfl.ch:{sites_file} {sites_file.parent} -v

# %%
sites = pd.read_csv(sites_file, index_col = 0)
sites["gene"] = pd.Categorical(sites["gene"], categories = transcriptome.var.index)
sites["mid"] = (sites["start"] + sites["end"]) / 2

sites_genes = dict(list(sites.groupby("gene")))

# %%
slicetopologies["gene"] = pd.Categorical(transcriptome.var.iloc[slicetopologies["gene_ix"]].index)
slicetopologies["cluster"] = pd.Categorical(cluster_info.iloc[slicetopologies["cluster_ix"]]["label"])

# %%
import tqdm.auto as tqdm
slicetopologies["end2"] = slicetopologies["end"] + window[0]
slicetopologies["start2"] = slicetopologies["start"] + window[0]
n_sites = []
for _, slice_oi in tqdm.tqdm(slicetopologies.iterrows(), total = len(slicetopologies)):
    end = slice_oi["end2"]
    start = slice_oi["start2"]
    n = len(sites_genes[slice_oi["gene"]].query("mid <= @end & mid >= @start"))
    n_sites.append(n)
slicetopologies["n_sites"] = n_sites

# %%
slicetopologies["perc_sites"] = slicetopologies["n_sites"] / slicetopologies["length"]

# %%
fig, ax = plt.subplots(figsize=(3, 3))

plotdata = (
    slicetopologies.groupby("type")["n_sites"].sum()
    / (slicetopologies.groupby("type")["length"].sum() / 100)
)
ax.bar(
    np.arange(len(plotdata)),
    plotdata,
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)

# %%
slicetopologies_oi = slicetopologies.query("cluster == 'B'")
slicetopologies_ref = slicetopologies.query("cluster !=  'B'")
# slicetopologies_ref = slicetopologies.query("cluster == 'NK'")

typescores = pd.DataFrame({
    "perc_sites_oi":    (slicetopologies_oi.groupby("type")["n_sites"].sum()
    / (slicetopologies_oi.groupby("type")["length"].sum() / 100)),
    "perc_sites_ref":    (slicetopologies_ref.groupby("type")["n_sites"].sum()
    / (slicetopologies_ref.groupby("type")["length"].sum() / 100)),
})
typescores["perc_sites_ratio"] = typescores["perc_sites_oi"] / typescores["perc_sites_ref"]

# %%
fig, ax = plt.subplots(figsize=(3, 3))

plotdata = typescores
ax.bar(
    np.arange(len(plotdata)),
    plotdata["perc_sites_ratio"],
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.set_xticklabels(plotdata.index)
ax.set_yscale("log")

# %%
fig, ax = plt.subplots(figsize=(3, 3))

plotdata = typescores
ax.bar(
    np.arange(len(plotdata)),
    plotdata["perc_sites_oi"],
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.set_xticklabels(plotdata.index)
ax.set_yscale("log")

# %% [markdown]
# ## Overlap with predictivity

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / "permutations_5fold5repeat"
    / "v20"
)

# %%
# %%
if not (prediction.path / "scoring" / "window_gene" / "chdscores_genes.pkl").exists():
    # chromatinhd scores
    chdscores_genes = {}
    genes_oi = transcriptome.var.index
    for gene in tqdm.tqdm(genes_oi):
        try:
            scores_folder = prediction.path / "scoring" / "window_gene" / gene
            window_scoring = chd.scoring.prediction.Scoring.load(scores_folder) 
        except FileNotFoundError:
            continue

        promoter = promoters.loc[gene]

        windowscores = window_scoring.genescores.sel(gene=gene).sel(phase = ["test", "validation"]).mean("phase").mean("model").to_pandas()

        chdscores_genes[gene] = windowscores
    chdscores_genes = pd.concat(chdscores_genes, names = ["gene"])
else:
    chdscores_genes = pd.read_pickle(prediction.path / "scoring" / "window_gene" / "chdscores_genes.pkl")

# %%
deltacors_interpolated = {}
effect_interpolated = {}
lost_interpolated = {}
for gene, chdscores_gene in chdscores_genes.drop(columns = ["gene"]).groupby("gene"):
    x_desired = np.arange(*window)
    x = chdscores_gene.index.get_level_values("window").values
    y = chdscores_gene["deltacor"].values
    y_interpolated = np.interp(x_desired, x, y)
    deltacors_interpolated[gene] = y_interpolated

    y = chdscores_gene["effect"].values
    y_interpolated = np.interp(x_desired, x, y)
    effect_interpolated[gene] = y_interpolated

    y = chdscores_gene["lost"].values
    y_interpolated = np.interp(x_desired, x, y)
    lost_interpolated[gene] = y_interpolated

# %%
import tqdm.auto as tqdm
slicetopologies["end2"] = slicetopologies["end"] + window[0]
slicetopologies["start2"] = slicetopologies["start"] + window[0]

deltacor = []
effect = []
lost = []
for _, slice_oi in tqdm.tqdm(slicetopologies.iterrows(), total = len(slicetopologies)):
    end = slice_oi["end"]
    start = slice_oi["start"]
    deltacor.append(deltacors_interpolated[slice_oi["gene"]][start:end].mean())
    effect.append(effect_interpolated[slice_oi["gene"]][start:end].mean())
    lost.append(lost_interpolated[slice_oi["gene"]][start:end].mean())
slicetopologies["deltacor"] = deltacor
slicetopologies["effect"] = effect
slicetopologies["lost"] = lost

# %%
slicetopologies["deltacor_norm"] = slicetopologies["deltacor"] / slicetopologies["lost"]
slicetopologies["effect_norm"] = slicetopologies["effect"] / slicetopologies["lost"]
# slicetopologies["deltacor_norm"] = slicetopologies["deltacor"] / np.exp(slicetopologies["average"])
# slicetopologies["effect_norm"] = slicetopologies["effect"] / np.exp(slicetopologies["average"])

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = (
    slicetopologies.groupby("type")["deltacor_norm"].mean()
)
ax.bar(
    np.arange(len(plotdata)),
    plotdata,
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.set_xticklabels(plotdata.index)
ax.invert_yaxis()
ax.set_title("Deltacor per fragment")
chd.slicetypes.label_axis(ax, ax.xaxis)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
for type, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["effect_norm"],
        color=chd.slicetypes.types_info.loc[type, "color"],
    )

# %% [markdown]
# ### Overlap with peak/window

# %%
def position_chosen_type(slices, window, n_clusters, n_genes):
    position_chosen = np.zeros(
        n_clusters * n_genes * (window[1] - window[0]),
        dtype=bool,
    )
    for start, end, gene_ix, cluster_ix in zip(
        slices["start"], slices["end"], slices["gene_ix"], slices["cluster_ix"]
    ):
        position_chosen[
            (start + (window[1] - window[0]) * gene_ix * cluster_ix) : (
                end + (window[1] - window[0]) * gene_ix * cluster_ix
            )
        ] = True
    return position_chosen


# %%
peak_position_chosen = peakresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), window[1] - window[0])
)
ns = []
percs = []
for _, (start, end, gene_ix, cluster_ix) in slicetopologies[
    ["start", "end", "gene_ix", "cluster_ix"]
].iterrows():
    pos = peak_position_chosen[gene_ix, cluster_ix, start:end]
    ns.append(pos.sum())
    percs.append(pos.mean())

# %%
slicetopologies["n_overlap"] = ns
slicetopologies["perc_overlap"] = percs
slicetopologies["high_overlap"] = np.array(percs) > 0.5

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = (
    slicetopologies.groupby("type")["n_overlap"].sum()
    / slicetopologies.groupby("type")["length"].sum()
)
ax.bar(
    np.arange(len(plotdata)),
    plotdata,
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.axhline(
    (slicetopologies["n_overlap"].sum() / slicetopologies["length"].sum()),
    dashes=(2, 2),
    color="#333",
)
ax.set_title(f"Overlap between {peakcaller} differential peaks\n and ChromatinHD")
ax.set_ylim(0, 1)
ax.set_xticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.xaxis)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["perc_overlap"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
sns.ecdfplot(slicetopologies["perc_overlap"], color="grey")
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# %% [markdown]
# ### Enrichment

# %%
motifscan_name = "cutoff_0001"
# motifscan_name = "onek1k_0.2"
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_cns"
# motifscan_name = "gtex"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)


# %%
def enrich_groups_cluster_vs_clusters(
    regions, clustering_id, grouping_id, inclusive=False
):
    motifscores = []
    for cluster_id in regions[clustering_id].cat.categories:
        oi = regions[clustering_id] == cluster_id
        for group_id in regions[grouping_id].cat.categories:
            print(group_id)
            oi_slices = oi & (regions[grouping_id] == group_id)
            background_slices = (
                (~oi) if inclusive else (~oi & (regions[grouping_id] == group_id))
            )
            motifscores_group = chd.models.diff.enrichment.enrich_windows(
                motifscan,
                regions[["start", "end"]].values,
                regions["gene_ix"].values,
                oi_slices=oi_slices,
                background_slices=background_slices,
                n_genes=fragments.n_genes,
                window=window,
                n_background=None,
            )
            motifscores_group[grouping_id] = group_id
            motifscores_group[clustering_id] = cluster_id

            motifscores.append(motifscores_group)
    motifscores = pd.concat(motifscores).reset_index()
    motifscores = motifscores.reset_index().set_index(
        [clustering_id, grouping_id, "motif"]
    )
    return motifscores


# %%
slicetopologies["cluster"] = pd.Categorical(
    cluster_info.index[slicetopologies["cluster_ix"]], categories=cluster_info.index
)

# %%
typeenrichments = enrich_groups_cluster_vs_clusters(
    slicetopologies, "cluster", "type", inclusive=False
)
type_group_enrichments = chd.models.diff.enrichment.enrich_cluster_vs_clusters(
    motifscan, window, slicetopologies, "type", fragments.n_genes
)

# %%
typeenrichments["perc_gene_mean"] = [x.mean() for x in typeenrichments["perc_gene"]]
# typeenrichments["perc_gene_mean"] = [x[transcriptome.var["chr"] == "chr6"].mean() for x in typeenrichments["perc_gene"]]

# %%
typeenrichments["significant"] = typeenrichments["qval"] < 0.05

# %%
fig, ax = plt.subplots(figsize=(2, 2))
plotdata = typeenrichments.groupby("type")[["perc", "logodds", "perc_gene_mean"]].mean()
ax.barh(
    np.arange(len(plotdata)),
    plotdata["perc"],
    color=chd.slicetypes.types_info.loc[plotdata.index]["color"],
)
ax.set_yticks(np.arange(len(plotdata)))
ax.set_yticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.yaxis)

# %%
# motifs_oi = typeenrichments.groupby("motif")["significant"].any()
# typeenrichments.loc[(slice(None), motifs_oi.index[motifs_oi]), :].groupby("type")[["perc", "logodds"]].mean().style.bar()

# %%
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("PO2F2")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("TFE2")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("PAX5")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA4")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("SPI1")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF4")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("CEBPB")][0]; cluster = "Monocytes"
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("TCF7")][0]; cluster = "CD4 T"
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("STAT3")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("NFA")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("bin")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("monoc")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("ZBT14")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("INSM1")][0]
# motif_id = motifscan.motifs.index[
#     motifscan.motifs.index.str.contains("Rheumatoid arthritis")
# ][0]

# %%
motifs = motifscan.motifs

# %%
# motifclustermapping = pd.DataFrame([
#     [motifs.loc[motifs.index.str.contains("SPI1")].index[0], ["Monocytes", "cDCs"]],
#     [motifs.loc[motifs.index.str.contains("CEBPB")].index[0], ["Monocytes", "cDCs"]],
#     [motifs.loc[motifs.index.str.contains("PEBB")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("IRF8")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("IRF4")].index[0], ["cDCs"]],
#     [motifs.loc[motifs.index.str.contains("PAX5")].index[0], ["Lymphoma"]],
#     [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["B", "pDCs"]], # TCF3
#     [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],
#     [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["B"]],
#     [motifs.loc[motifs.index.str.contains("NFKB2")].index[0], ["B"]],
#     [motifs.loc[motifs.index.str.contains("RUNX2")].index[0], ["NK"]],
#     [motifs.loc[motifs.index.str.contains("RUNX1")].index[0], ["T"]],
#     [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["T"]],
# ], columns = ["motif", "clusters"]).set_index("motif")
# motifclustermapping = motifclustermapping.explode("clusters").rename(columns = {"clusters":"cluster"}).reset_index()[["cluster", "motif"]]

motifclustermapping = pd.DataFrame(
    [
        [motifs.loc[motifs.index.str.contains("TCF7")].index[0], ["CD4 T", "CD8 T"]],
        [motifs.loc[motifs.index.str.contains("IRF8")].index[0], ["cDCs"]],
        [motifs.loc[motifs.index.str.contains("IRF4")].index[0], ["cDCs"]],
        [
            motifs.loc[motifs.index.str.contains("CEBPB")].index[0],
            ["Monocytes", "cDCs"],
        ],
        [motifs.loc[motifs.index.str.contains("GATA4")].index[0], ["CD4 T"]],
        [
            motifs.loc[motifs.index.str.contains("HNF6_HUMAN.H11MO.0.B")].index[0],
            ["CD4 T"],
        ],
        [motifs.loc[motifs.index.str.contains("RARA")].index[0], ["MAIT"]],
        [motifs.loc[motifs.index.str.contains("PEBB")].index[0], ["NK"]],
        [motifs.loc[motifs.index.str.contains("RUNX2")].index[0], ["NK"]],
        [motifs.loc[motifs.index.str.contains("RUNX1")].index[0], ["CD8 T"]],
        [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["CD8 T"]],
        [
            motifs.loc[motifs.index.str.contains("TBX21_HUMAN.H11MO.0.A")].index[0],
            ["NK"],
        ],
        [
            motifs.loc[motifs.index.str.contains("SPI1")].index[0],
            ["Monocytes", "B", "cDCs"],
        ],
        [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["B"]],
        [motifs.loc[motifs.index.str.contains("NFKB2")].index[0], ["B"]],
        [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["B", "pDCs"]],  # TCF3
        [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],
        [motifs.loc[motifs.index.str.contains("FOS")].index[0], ["cDCs"]],
        [motifs.loc[motifs.index.str.contains("RORA")].index[0], ["MAIT"]],
    ],
    columns=["motif", "clusters"],
).set_index("motif")
motifclustermapping = (
    motifclustermapping.explode("clusters")
    .rename(columns={"clusters": "cluster"})
    .reset_index()[["cluster", "motif"]]
)

# %%
typeenrichments.query("qval < 0.1").sort_values("odds", ascending=False)["perc_gene"]

# %%
pd.DataFrame(
    {
        "gene": transcriptome.var.index,
        "perc": typeenrichments.query("qval < 0.1").sort_values(
            "odds", ascending=False
        )["perc_gene"][0],
    }
).sort_values("perc")

# %%
gene_id = "ENSG00000090104"
transcriptome.var["ix"] = np.arange(len(transcriptome.var))
gene_ix = transcriptome.var.loc[gene_id, "ix"]

# %%
motifscan.motifs["ix"] = np.arange(len(motifscan.motifs))
motif_id = "Multiple sclerosis"
motif_ix = motifscan.motifs.loc[motif_id, "ix"]

# %%
position_range = (
    gene_ix * (window[1] - window[0]),
    ((gene_ix + 1) * (window[1] - window[0])),
)

# %%
motifscan.indices[
    motifscan.indptr[position_range[0]] : motifscan.indptr[position_range[1]]
]

# %%
motif_ix

# %%
motifscan.indptr

# %%

# %%
transcriptome.var.iloc[292]

# %%
transcriptome.var.iloc[768]

# %%
# typeenrichments.loc["CD8 T"].groupby("motif").mean().sort_values("logodds", ascending = False).query("significant > 0").head(20)

# %%
sc.pl.umap(transcriptome.adata, color="cluster")

# %%
cors = pd.DataFrame(
    np.corrcoef(typeenrichments["logodds"].unstack().T),
    index=motifscan.motifs.index,
    columns=motifscan.motifs.index,
)

# %%
cors["SNAI2_HUMAN.H11MO.0.A"].sort_values(ascending=False).head(10)

# %%
# sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["NFIL3", "CEBPB", "CEBPD", "NFIL3"])) # also includes ATF4
sc.pl.umap(transcriptome.adata, color=transcriptome.gene_id(["TBX21"]))

# %%
typeenrichments_oi = motifclustermapping.set_index(["cluster", "motif"]).join(
    typeenrichments
)
typeenrichments_oi["logodds"].unstack(level="type").style.bar(axis=1, vmin=0)

# %%
plotdata

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata_mean = typeenrichments_oi.groupby("type")[["logodds"]].mean()
plotdata_mean = plotdata_mean.sort_values("logodds")
plotdata_mean["y"] = np.arange(len(plotdata_mean))
plotdata = typeenrichments_oi[["logodds"]].reset_index()

# scatter of individual
ax.scatter(
    np.exp(plotdata_mean["logodds"]),
    plotdata_mean["y"],
    color=chd.slicetypes.types_info.loc[plotdata_mean.index]["color"],
    marker="o",
)
# lines between individual
for motif, plotdata_motif in plotdata.groupby(["motif", "cluster"]):
    plotdata_motif["y"] = plotdata_mean["y"][plotdata_motif.type].values
    plotdata_motif = plotdata_motif.sort_values("y")
    ax.plot(
        np.exp(plotdata_motif["logodds"]), plotdata_motif["y"], color="#3331", zorder=-1
    )
# scatter of mean
ax.scatter(
    x=np.exp(plotdata["logodds"]).values,
    y=plotdata_mean["y"][plotdata["type"]].values,
    color=chd.slicetypes.types_info.loc[plotdata["type"]]["color"].values,
    s=1,
)
ax.axvline(1, dashes=(2, 2), color="grey")
ax.set_yticks(np.arange(len(plotdata_mean)))
ax.set_yticklabels(plotdata_mean.index)
chd.slicetypes.label_axis(ax, ax.yaxis)
ax.set_xscale("log")
ax.set_xlabel("Motif odds-ratio")
ax.set_xticks([1, 3 / 2, 2, 3])
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
sns.despine(ax=ax)

# %%
typeenrichments["logodds"].unstack().loc["B"].loc["ridge"].sort_values()

# %%
fig, ax = plt.subplots()
ax.scatter(
    typeenrichments["logodds"].unstack().loc["Monocytes"].loc["ridge"],
    typeenrichments["logodds"].unstack().loc["Monocytes"].loc["volcano"],
)

# %%
sns.heatmap(np.corrcoef(typeenrichments["logodds"].unstack().loc["Monocytes"]))

# %%
plt.hist(
    np.diag(
        typeenrichment_cors.xs("volcano", level="type").xs("peak", level="type", axis=1)
    )
)
plt.hist(
    np.diag(
        typeenrichment_cors.xs("flank", level="type").xs("peak", level="type", axis=1)
    )
)

# %%
typeenrichment_cors = pd.DataFrame(
    np.corrcoef(typeenrichments["logodds"].unstack()),
    typeenrichments["logodds"].unstack().index,
    typeenrichments["logodds"].unstack().index,
)
sns.heatmap(typeenrichment_cors)

# %% [markdown]
# ### Characteristics

# %% [markdown]
# #### Conservation

# %%
import chromatinhd.conservation

folder_cons = chd.get_output() / "data" / "cons" / "hs" / "gerp"
conservation = chd.conservation.Conservation(folder_cons / "hg38.phastCons100way.bw")

# %%
promoters["gene_ix"] = np.arange(len(promoters))

# %%
slicelocations = pureregionresult.get_slicelocations(promoters)
# slicelocations = pureregionresult.get_randomslicelocations(promoters)

# %%
conservations = []
for slice in slicelocations.itertuples():
    cons = conservation.get_values(slice.chr, slice.start_genome, slice.end_genome)
    cons[np.isnan(cons)] = 0.0
    conservations.append(cons.mean())
conservations = np.array(conservations)

# %%
slicetopologies["conservation"] = conservations
slicetopologies.groupby("type")["conservation"].mean().plot()

# %%
n_random = 1

slicelocations_random = pureregionresult.get_randomslicelocations(
    promoters, n_random=n_random
)

conservations = []
for slice in slicelocations_random.itertuples():
    cons = conservation.get_values(slice.chr, slice.start_genome, slice.end_genome)
    cons[np.isnan(cons)] = 0.0
    conservations.append(cons.mean())
conservations = np.array(conservations)

# %%
slicetopologies_random = slicetopologies.iloc[
    np.repeat(np.arange(len(slicetopologies)), n_random)
].copy()
slicetopologies_random["conservation"] = conservations
slicetopologies_random.groupby("type")["conservation"].mean().plot()

# %%
(
    slicetopologies.groupby("type")["conservation"].mean()
    / slicetopologies_random.groupby("type")["conservation"].mean()
).plot()

# %%
slicetopologies["conservation_random"] = (
    slicetopologies_random["conservation"]
    .values.reshape((len(slicetopologies), n_random))
    .mean(1)
)

# %%
slicetopologies["conservation_norm"] = np.log(
    slicetopologies["conservation"] / slicetopologies["conservation_random"]
)

# %%
slicetopologies.groupby("type")["conservation_norm"].mean()

# %%
sns.boxplot(x="type", y="conservation_norm", data=slicetopologies)

# %% [markdown]
# #### QTLs

# %%
motifscan_name = "gwas_immune"
# motifscan_name = "gtex_immune"
# motifscan_name = "onek1k_0.2"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)

# %%
slicetopologies["cluster_type"] = pd.Categorical(
    slicetopologies["cluster_ix"].astype(str)
    + "_"
    + slicetopologies["type"].astype(str)
)

# %%
qtl_enrichments = chd.models.diff.enrichment.enrich_cluster_vs_all(
    motifscan,
    window,
    slicetopologies,
    "cluster_type",
    fragments.n_genes,
    fragments.var.index,
)
# qtl_enrichments = chd.models.diff.enrichment.enrich_cluster_vs_clusters(motifscan, window, slicetopologies, "cluster_type", fragments.n_genes)

# %%
qtl_enrichments["type"] = (
    qtl_enrichments.index.get_level_values("cluster_type").str.split("_").str[1]
)
qtl_enrichments["cluster"] = cluster_info.index[
    qtl_enrichments.index.get_level_values("cluster_type")
    .str.split("_")
    .str[0]
    .astype(int)
]

# %%
qtl_enrichments.groupby("type")["n_found"].sum()

# %%
qtl_enrichments

# %%
qtl_enrichments.loc[
    qtl_enrichments.index.get_level_values("cluster_type") == "0_canyon"
].query("n_found > 0")

# %%
motif_id = "Multiple sclerosis"
cluster = "B"
# motif_id = "Hodgkin's lymphoma"; cluster = "CD4 T"
motif_id = "Crohn's disease"
cluster = "cDCs"

# %%
qtl_enrichments["label"] = transcriptome.symbol(
    qtl_enrichments.index.get_level_values("gene")
).values

# %%
qtl_enrichments.query("`disease/trait` == @motif_id").query(
    "cluster == @cluster"
).query("type == 'canyon'").sort_values("n_found")

# %%
qtl_enrichments.query("cluster_type == '0_ridge'").query("n_found > 0")

# %%
# qtl_enrichments.query("qval < 0.1").sort_values("odds", ascending = False)["perc_gene"]

# %%
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
).plot(kind="bar")

# %%
(
    qtl_enrichments.groupby("type")["n"].sum()
    / (fragments.n_genes * (window[1] - window[0]))
)

# %%
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
) / (
    qtl_enrichments.groupby("type")["n"].sum()
    / (fragments.n_genes * (window[1] - window[0]))
)

# %%
fig, ax = plt.subplots()
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
    * qtl_enrichments.groupby("type")["n"].sum()
).plot()
ax.set_ylim(0)

# %%
fig, ax = plt.subplots()
(
    qtl_enrichments.groupby("type")["n_found"].sum()
    / slicetopologies.groupby("type")["length"].sum()
).plot()
ax.set_ylim(0)

# %% [markdown]
# ### Get interesting genes

# %%
slicetopologies["gene_symbol"] = transcriptome.var.iloc[slicetopologies["gene_ix"]][
    "symbol"
].values
slicetopologies["cluster"] = cluster_info.index[slicetopologies["cluster_ix"]]

# %%
slicetopologies.query("type == 'canyon'").query("cluster == 'B'").query("length > 100")[
    ["expression_lfc", "cluster", "start", "length", "gene_symbol"]
].head(20)

# %%
slicetopologies.query("type == 'canyon'").query("expression_lfc < log(2)").sort_values(
    "average_lfc"
).query("cluster == 'Monocytes'").query("length > 100")[
    ["expression_lfc", "cluster", "start", "length", "gene_symbol"]
].head(
    10
)

# %%
slicetopologies.query("type == 'canyon'").query("expression_lfc <= log(2)")[
    ["cluster_ix", "gene_ix"]
]

# %% [markdown]
# ## Expression

# %%
y_cells = np.array(transcriptome.X.to_scipy_csr().todense())
y_clusters = (
    pd.DataFrame(y_cells, index=pd.from_dummies(latent)).groupby(level=0).mean().values
)

# %%
transcriptome.var["ix"] = np.arange(len(transcriptome.var))

# %%
sc.tl.rank_genes_groups(transcriptome.adata, "cluster")

cluster_diffexp = []
for cluster in cluster_info.index:
    diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, cluster)
    diffexp["gene"] = diffexp["names"]
    diffexp = diffexp.set_index("gene")
    diffexp["lfc"] = diffexp["logfoldchanges"]
    diffexp["cluster"] = cluster
    diffexp["gene_ix"] = transcriptome.var.loc[diffexp.index, "ix"]
    diffexp["cluster_ix"] = cluster_info.loc[cluster, "dimension"]
    cluster_diffexp.append(diffexp)
cluster_diffexp = pd.concat(cluster_diffexp)
cluster_diffexp = cluster_diffexp.reset_index().set_index(["cluster", "gene"])

# %%
lfc_clusters = (
    cluster_diffexp.set_index(["cluster_ix", "gene_ix"])["lfc"].unstack().values
)
minmax_clusters = lfc_clusters.copy()
minmax_clusters = (minmax_clusters - minmax_clusters.min(0, keepdims=True)) / (
    minmax_clusters.max(0, keepdims=True) - minmax_clusters.min(0, keepdims=True)
)

# %% [markdown]
# ### Correlation with differential expression and slice type

# %%
fimp_full = np.zeros((fragments.n_genes, (window[1] - window[0])))

# %%
mixture_interpolated = probs_interpolated - probs_interpolated.mean(-1, keepdims=True)

# %%
tasks = []
for gene_ix in tqdm.tqdm(range(0, 5000)):
    # X = basepair_ranking[gene_ix]
    X = probs_interpolated[gene_ix]
    # X = mixture_interpolated[gene_ix]
    X[X == -np.inf] = 0
    y = y_clusters[:, gene_ix]
    tasks.append((X, y))


# %%
def calculate_importance(task):
    X, y = task
    rf = sklearn.ensemble.RandomForestRegressor(
        n_estimators=1000, max_depth=1, max_features=0.05
    )
    rf.fit(X, y)
    # plt.plot(rf.feature_importances_)
    fimp = scipy.ndimage.gaussian_filter(rf.feature_importances_, 20.0)
    return fimp


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    cor = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
    cor[np.isnan(cor)] = 0
    return cor


def calculate_importance(task):
    X, y = task
    return corr2_coeff(X.T, y[None, :])[:, 0]


# %% [markdown]
# $${\displaystyle r={\frac {\sum _{i}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})/(n-1)}{s(x)s(y)}}}$$

# %%
results = list(tqdm.tqdm(map(calculate_importance, tasks)))
results = np.vstack(list(results))

# pool = ProcessPoolExecutor(10)
# results = pool.map(calculate_importance, tasks)
# results = np.vstack(list(results))

# %%
fimp_full[range(results.shape[0])] = results

# %%
gene_ix = transcriptome.gene_ix("PTGDS")

# %%
fig, ax = plt.subplots()
plt.plot(fimp_full[gene_ix])

transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
for _, slice in slicetopologies.query("gene_ix == @gene_ix").iterrows():
    ax.plot(
        [slice["start"], slice["end"]],
        [0.99, 0.99],
        color=chd.slicetypes.types_info.loc[slice["type"], "color"],
        transform=transform,
        lw=3,
        clip_on=False,
        solid_capstyle="butt",
    )
sns.despine()

# %%
slice_fimps = []
slicetype_fimps = {
    slicetype: [] for slicetype in slicetopologies["type"].cat.categories
}
for ix, slice in slicetopologies.iterrows():
    fimp_slice = fimp_full[slice["gene_ix"], slice["start"] : slice["end"]]
    slicetype_fimps[slice.type].extend(fimp_slice)
    slice_fimps.append(fimp_slice.max())
slicetopologies["fimp"] = slice_fimps

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Positions")
for slicetype, fimp in slicetype_fimps.items():
    fimp = np.array(fimp)
    print(slicetype, (fimp > 0.2).mean(), (fimp < -0.2).mean())
    sns.ecdfplot(fimp, color=chd.slicetypes.types_info.loc[slicetype, "color"])
ax.set_xlabel("Correlation")

# %%
expressionscores = []
cor_cutoff = 0.2
for slicetype, fimp in slicetype_fimps.items():
    fimp = np.array(fimp)
    expressionscores.append(
        {
            "slicetype": slicetype,
            "up": (fimp > cor_cutoff).mean(),
            "down": (fimp < -cor_cutoff).mean(),
        }
    )
expressionscores = pd.DataFrame(expressionscores)
expressionscores["same"] = 1 - expressionscores["up"] - expressionscores["down"]
expressionscores.plot(kind="bar")

# %%
slicetypes_info = chd.slicetypes.types_info

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Regions")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    print(slicetype, ((np.abs(slicetopologies_type["fimp"]) > 0.25).mean()))
    sns.ecdfplot(
        slicetopologies_type["fimp"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
ax.set_xlabel("Mean absolute correlation between expression and accessibility")
ax.set_xlim(0, 1)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Regions outside promoter (end is within -10kb → 0)")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    print(slicetype, (np.abs(slicetopologies_type["fimp"] > 0.25).mean()))
    sns.ecdfplot(
        slicetopologies_type.query("(end + @window[0]) < 0")["fimp"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
ax.set_xlabel("Mean absolute correlation between expression and accessibility")
ax.set_xlim(0, 1)

# %%
slicetopologies.groupby("type")["fimp"].mean().plot()

# %%
yrank_clusters = scipy.stats.rankdata(y_clusters, axis=0, method="min")

# %%
slice_ranks = []
for ix, slice in slicetopologies.iterrows():
    expression_rank = yrank_clusters[slice["cluster_ix"], slice["gene_ix"]]
    slice_ranks.append(expression_rank)
slicetopologies["expression_rank"] = slice_ranks

# %%
fig, ax = plt.subplots()
ax.set_title("Expression rank")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["expression_rank"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )

# %% [markdown]
# ### Correlation with presence of slice type and expression effect

# %%
slicetopologies["expression_lfc"] = (lfc_clusters)[
    slicetopologies["cluster_ix"], slicetopologies["gene_ix"]
]
slicetopologies["expression_minmax"] = (minmax_clusters)[
    slicetopologies["cluster_ix"], slicetopologies["gene_ix"]
]

# %%
slicetype_scores = pd.DataFrame(
    {
        "mean_expression_lfc": np.exp(
            slicetopologies.groupby("type")["expression_lfc"].mean()
        ),
        "mean_expression_minmax": slicetopologies.groupby("type")["expression_minmax"].mean(),
        "median_expression_lfc": np.exp(
            slicetopologies.groupby("type")["expression_lfc"].median()
        ),
    }
)

# %%
sns.histplot(slicetopologies.query("type == 'hill'")["expression_minmax"], stat = "density", bins = 20)
sns.histplot(slicetopologies.query("type == 'peak'")["expression_minmax"], color="red", alpha=0.5, bins=20, stat="density")

# %%
fig, ax = plt.subplots(figsize=(3, 3))
# ax.set_title("Regions outside promoter (end is within -10kb → 0)")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        np.exp(slicetopologies_type["expression_lfc"]),
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )
ax.set_xlabel("Gene differential expression")
ax.set_xscale("log")
ax.set_xlim(1 / 10, 10)

# %%
expression_changes_info = pd.DataFrame(
    {
        "upper_limit": np.array([1 / 8, 1 / 4, 1 / 2, 2, 4, 8, np.inf]),
        "label": ["⅛", "¼", "½", r"1", "2", "4", "8"],
    }
)
expression_changes_info.index = expression_changes_info["label"]
# expression_changes_info["label"] = expression_changes_info.index
expression_changes_info["ix"] = np.linspace(0, 1, len(expression_changes_info) + 2)[
    1:-1
]

# %%
slicetopologies["expression_change"] = expression_changes_info.index[
    np.digitize(
        slicetopologies["expression_lfc"],
        np.log(expression_changes_info["upper_limit"]),
    )
]

# %%
slicetype_expression_changes = slicetopologies.groupby(
    ["type", "expression_change"]
).size()
slicetype_expression_changes = (
    slicetype_expression_changes / slicetype_expression_changes.groupby("type").sum()
)

# %%
expression_changes_info["color"] = [
    c for c in mpl.cm.RdBu_r(np.linspace(0, 1, len(expression_changes_info)))
]

# %%
plotdata = slicetype_expression_changes.reset_index(name="perc")
plotdata["expression_change"] = pd.Categorical(
    plotdata["expression_change"], categories=expression_changes_info.index
)

plotdata = plotdata.sort_values("expression_change")
plotdata["cumperc"] = plotdata.groupby("type")["perc"].cumsum()

# %%
slicetype_scores = slicetype_scores.sort_values(
    "median_expression_lfc", ascending=False
)
slicetype_scores["ix"] = np.arange(len(slicetype_scores))

# %%
plotdata["ix"] = slicetype_scores.loc[plotdata["type"], "ix"].values

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.barh(
    plotdata["ix"],
    plotdata["perc"],
    left=plotdata["cumperc"] - plotdata["perc"],
    color=expression_changes_info.loc[plotdata["expression_change"], "color"],
    lw=0,
    height=0.9
    # color = chd.slicetypes.types_info.loc[plotdata["type"], "color"]
)
ax.set_yticks(np.arange(len(slicetype_scores)))
# ax.set_title(f"Overlap between peaks ({peaks_name}) and ChromatinHD")
ax.set_xlim(0, 1)
ax.set_ylim(-0.4, len(slicetype_scores) - 0.6)
ax.set_yticklabels(slicetype_scores.index)
chd.slicetypes.label_axis(ax, ax.yaxis)
sns.despine(ax=ax)

plotdata_oi = plotdata.loc[plotdata["ix"] == plotdata["ix"].max()].set_index(
    "expression_change"
)
texts = []
for expression_change_oi, expression_change_info in expression_changes_info.iterrows():
    texts.append(
        ax.text(
            plotdata_oi.loc[expression_change_oi, "cumperc"]
            - plotdata_oi.loc[expression_change_oi, "perc"] / 2,
            1.0,
            expression_change_info["label"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
    )
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = slicetype_scores
ax.bar(
    np.arange(len(plotdata)),
    np.log(plotdata["mean_expression_lfc"]),
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.axhline(0, dashes=(2, 2), color="#333")
ax.set_ylabel("Average fold-change")
ax.set_ylim([np.log(1 / 8), np.log(8)])
ax.set_xticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.xaxis)

# %%


# %%
plotdata = slicetopologies.query("type == 'canyon'")
fig, ax = plt.subplots(figsize=(3, 3))
# ax.scatter((plotdata["expression_lfc"]), plotdata["balances_raw"], s = 1)
# sns.regplot(x = plotdata["expression_lfc"], y = plotdata["balances_raw"], marker = "None")

ax.scatter((plotdata["expression_lfc"]), np.log(plotdata["balance"]))
sns.regplot(x=plotdata["expression_lfc"], y=np.log(plotdata["balance"]))

# %% [markdown]
# ### Association with high expression and slice type

# %%
yrank = scipy.stats.rankdata(y_clusters.max(0), method="min")

# %%
slice_ranks = []
for ix, slice in slicetopologies.iterrows():
    expression_rank = yrank[slice["gene_ix"]]
    slice_ranks.append(expression_rank)
slicetopologies["expression_rank"] = slice_ranks

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Expression rank")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type["expression_rank"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title("Expression rank")
for slicetype, slicetopologies_type in slicetopologies.groupby("type"):
    sns.ecdfplot(
        slicetopologies_type.query("(end + @window[0]) > -1000")["expression_rank"],
        color=chd.slicetypes.types_info.loc[slicetype, "color"],
    )

# %%
slicetopologies.groupby("type")["expression_rank"].mean().plot()

# %%
slice_fimps = []
for ix, slice in slicetopologies.iterrows():
    slice_fimps.append(
        np.abs(fimp_full[slice["gene_ix"], slice["start"] : slice["end"]]).max()
    )
slicetopologies["fimp"] = slice_fimps

# %%
slicetopologies.groupby("type")["fimp"].mean().plot()

# %%
fig, ax = plt.subplots(figsize=(3, 3))
plotdata = (
    slicetopologies.groupby("type")["n_overlap"].sum()
    / slicetopologies.groupby("type")["length"].sum()
)
ax.bar(
    np.arange(len(plotdata)),
    plotdata,
    color=chd.slicetypes.types_info.loc[plotdata.index, "color"],
)
ax.set_xticks(np.arange(len(plotdata)))
ax.axhline(
    (slicetopologies["n_overlap"].sum() / slicetopologies["length"].sum()),
    dashes=(2, 2),
    color="#333",
)
ax.set_title(f"Overlap between peaks ({peaks_name}) and ChromatinHD")
ax.set_ylim(0, 1)
ax.set_xticklabels(plotdata.index)
chd.slicetypes.label_axis(ax, ax.xaxis)

# %% [markdown]
# ## Conservation

# %%
import chromatinhd.conservation

folder_cons = chd.get_output() / "data" / "cons" / "hs" / "gerp"
conservation = chd.conservation.Conservation(folder_cons / "hg38.phastCons100way.bw")

# %%
promoters["gene_ix"] = np.arange(len(promoters))

# %%
# sliceinfo = regionresult.get_slicelocations(promoters)
sliceinfo = peakresult.get_slicelocations(promoters)
# sliceinfo = pureregionresult.get_slicelocations(promoters)

# %%
conservations = []
for slice in sliceinfo.itertuples():
    conservations.extend(
        conservation.get_values(slice.chr, slice.start_genome, slice.end_genome)
    )
conservations = np.array(conservations)

# %%
conservations[np.isnan(conservations)] = 0.0

# %%
np.mean(conservations)
