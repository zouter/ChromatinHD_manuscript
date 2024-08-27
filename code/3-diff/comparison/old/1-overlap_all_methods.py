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
import IPython

if IPython.get_ipython():
    IPython.get_ipython().magic("load_ext autoreload")
    IPython.get_ipython().magic("autoreload 2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_name = "pbmc10k"
promoter_name = "10k10k"
window = np.array([-10000, 10000])

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / dataset_name

# %%
folder_plots = folder_root / "plots" / "overlap_all_methods"

# %% [markdown]
# ### Load data

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
latent_name = "leiden_0.1"

# %%
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = pd.Categorical(
    pd.from_dummies(latent).iloc[:, 0]
)

# %% [markdown]
# ## Commonalities between peak approaches

# %%
from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_combinations,
)
from chromatinhd_manuscript.designs import dataset_latent_method_combinations

# %%
dataset_latent_peakcaller_diffexp_combinations["method"] = (
    dataset_latent_peakcaller_diffexp_combinations["diffexp"]
    + "/"
    + dataset_latent_peakcaller_diffexp_combinations["peakcaller"]
)

# %%
design = (
    pd.concat(
        [
            dataset_latent_peakcaller_diffexp_combinations,
            dataset_latent_method_combinations,
        ]
    )
    .query("dataset == @dataset_name")
    .query("latent ==  @latent_name")
)


def get_score_folder(x):
    if not pd.isnull(x.peakcaller):
        return (
            chd.get_output()
            / "prediction_differential"
            / x.dataset
            / x.promoter
            / x.latent
            / x.method
        )
    return (
        chd.get_output()
        / "prediction_likelihood"
        / x.dataset
        / x.promoter
        / x.latent
        / x.method
        / "scoring"
        / "significant_up"
    )


design["score_folder"] = design.apply(get_score_folder, axis=1)
design = design.set_index("method")
design = design.query("promoter == '10k10k'")
# assert design.index.duplicated().sum() == 0

# %%
# design = design.query(
#     "(diffexp == 'scanpy_wilcoxon') | (method == 'v9_128-64-32') | (diffexp == 'scanpy')"
# )

# %%
import chromatinhd.differential

peakresults = {}
for design_ix, design_row in design.iterrows():
    scores_dir = design_row["score_folder"]
    try:
        peakresults[design_ix] = pickle.load((scores_dir / "slices.pkl").open("rb"))
        peakresults[design_ix].position_chosen_cached = peakresults[
            design_ix
        ].position_chosen
    except FileNotFoundError as e:
        print(e)

# %%
import itertools

# %%
peakpair_scores = pd.DataFrame(
    index=pd.MultiIndex.from_frame(
        pd.DataFrame(
            list(itertools.combinations(list(peakresults.keys()), 2)),
            columns=["peakcaller1", "peakcaller2"],
        )
    )
)


# %% [markdown]
# ### F1

# %%
def intersect_intervals(start1, start2, end1, end2):
    return np.maximum(
        0, np.minimum(end1, end2[:, None]) - np.maximum(start1, start2[:, None])
    )


def union_intervals(intersect, start1, start2, end1, end2):
    return (end1 - start1) + (end2[:, None] - start2[:, None]) - intersect


def jaccard_intervals(start1, start2, end1, end2):
    intersect = intersect_intervals(start1, start2, end1, end2)
    union = union_intervals(intersect, start1, start2, end1, end2)
    return intersect / union


start1 = np.array([300, 500, 1000])
end1 = np.array([500, 1000, 2000])

start2 = np.array([100, 1500])
end2 = np.array([200, 1800])

intersect = intersect_intervals(start1, start2, end1, end2)
union = union_intervals(intersect, start1, start2, end1, end2)
jaccard = jaccard_intervals(start1, start2, end1, end2)


# %%
def calculate_slicescore_F1(slicescores1, slicescores2):
    slicescores1["genexcluster"] = (
        slicescores1["gene_ix"].astype(int) * len(cluster_info)
        + slicescores1["cluster_ix"].astype(int)
    ).values
    slicescores1 = slicescores1.sort_values("genexcluster")

    slicescores2["genexcluster"] = (
        slicescores2["gene_ix"].astype(int) * len(cluster_info)
        + slicescores2["cluster_ix"].astype(int)
    ).values
    slicescores2 = slicescores2.sort_values("genexcluster")

    n_genexcluster = fragments.n_genes * len(cluster_info)
    genexcluster_indptr1 = chd.utils.indices_to_indptr(
        slicescores1["genexcluster"].values, n_genexcluster
    )

    start1 = slicescores1["start"].values
    end1 = slicescores1["end"].values
    start2 = slicescores2["start"].values
    end2 = slicescores2["end"].values

    genexcluster_indptr2 = chd.utils.indices_to_indptr(
        slicescores2["genexcluster"].values, n_genexcluster
    )

    return calculate_F1(
        genexcluster_indptr1,
        genexcluster_indptr2,
        n_genexcluster,
        start1,
        end1,
        start2,
        end2,
        window,
    )


# import numba
# @numba.jit
def calculate_F1(
    genexcluster_indptr1,
    genexcluster_indptr2,
    n_genexcluster,
    start1,
    end1,
    start2,
    end2,
    window,
):
    recoveries = []
    relevances = []
    for genexcluster in np.arange(n_genexcluster):
        i1 = genexcluster_indptr1[genexcluster]
        j1 = genexcluster_indptr1[genexcluster + 1]
        i2 = genexcluster_indptr2[genexcluster]
        j2 = genexcluster_indptr2[genexcluster + 1]

        if j1 - i1 == 0:
            if j2 - i2 == 0:
                continue
            else:
                relevances.append([0] * (j2 - i2))
        elif j2 - i2 == 0:
            recoveries.append([0] * (j1 - i1))
        else:
            jac = jaccard_intervals(
                start1[i1:j1], start2[i2:j2], end1[i1:j1], end2[i2:j2]
            )

            relevances.append(jac.max(1))
            recoveries.append(jac.max(0))
    recoveries = np.hstack(recoveries)
    relevances = np.hstack(relevances)
    assert len(recoveries) == len(start1)
    assert len(relevances) == len(start2)
    return 2 / (1 / recoveries.mean() + 1 / relevances.mean())


# %%
for peakcaller1, peakcaller2 in peakpair_scores.index:
    print(peakcaller1, peakcaller2)
    peakresult1 = peakresults[peakcaller1]
    peakresult2 = peakresults[peakcaller2]
    slicescores1 = peakresult1.get_slicescores()
    slicescores2 = peakresult2.get_slicescores()
    peakpair_scores.loc[
        (peakcaller1, peakcaller2), "region_f1"
    ] = calculate_slicescore_F1(slicescores1, slicescores2)

# %% [markdown]
# ### Positional jaccard

# %%
for peakcaller1, peakcaller2 in peakpair_scores.index:
    print(peakcaller1, peakcaller2)
    peakresult1 = peakresults[peakcaller1]
    peakresult2 = peakresults[peakcaller2]
    # peakpair_scores.loc[(peakcaller1, peakcaller2), "position_jaccard"] = 0
    peakpair_scores.loc[(peakcaller1, peakcaller2), "position_jaccard"] = (
        peakresult1.position_chosen_cached & peakresult2.position_chosen_cached
    ).sum() / (
        peakresult1.position_chosen_cached | peakresult2.position_chosen_cached
    ).sum()

# %%
import scipy.spatial.distance

# %%
peakpair_scores_flipped = peakpair_scores.copy()
peakpair_scores_flipped.index = peakpair_scores_flipped.index.rename(
    {"peakcaller1": "peakcaller2", "peakcaller2": "peakcaller1"}
).reorder_levels(["peakcaller1", "peakcaller2"])
peakpair_scores_all = pd.concat([peakpair_scores, peakpair_scores_flipped])

# %%
methods_info = pd.DataFrame(design[["peakcaller", "diffexp"]], index=design.index)
methods_info = methods_info.loc[
    methods_info.index.isin(peakpair_scores_all.index.get_level_values(0))
]
methods_info["label"] = np.where(
    ~pd.isnull(methods_info["peakcaller"]),
    (
        chdm.peakcallers.reindex(methods_info["peakcaller"])["label"].reset_index(
            drop=True
        )
        + " ("
        + chdm.diffexps.reindex(methods_info["diffexp"])["label_short"].reset_index(
            drop=True
        )
        + ")"
    ).values,
    "ChromatinHD differential",
)


# %% [markdown]
# ### Plot OI

# %%
method_oi = "v9_128-64-32"
methods_info_oi = methods_info.loc[methods_info.index != method_oi].copy()
methods_info_oi = methods_info_oi.loc[
    peakpair_scores_all.loc[method_oi]
    .sort_values("position_jaccard", ascending=True)
    .index
]
methods_info_oi["ix"] = np.arange(len(methods_info_oi))

# %%
norm_jaccard = mpl.colors.Normalize(0, 1)
norm_f1 = mpl.colors.Normalize(0, 1)

cmap_jaccard = mpl.cm.Blues
cmap_f1 = mpl.cm.Reds

grid = polyptich.grid.Grid()
fig = polyptich.grid.Figure(grid)

resolution = 0.2
dim = (1, resolution * len(design))

panel_main = grid[0, 0] = polyptich.grid.Ax(dim)
ax = panel_main.ax
peakcaller1 = method_oi
for peakcaller2 in list(methods_info_oi.index):
    if (peakcaller1, peakcaller2) in peakpair_scores_all.index:
        pairscore = peakpair_scores_all.loc[peakcaller1, peakcaller2]
        x = 0
        y = methods_info_oi.loc[peakcaller2, "ix"]
        rect = mpl.patches.Rectangle(
            (x, y),
            1,
            1,
            fc=cmap_jaccard(norm_jaccard(pairscore["position_jaccard"])),
            lw=0,
        )
        ax.add_patch(rect)

        x = 1
        rect = mpl.patches.Rectangle(
            (x, y), 1, 1, fc=cmap_f1(norm_f1(pairscore["region_f1"])), lw=0
        )
        ax.add_patch(rect)

ax.set_ylim(0, len(methods_info_oi))
ax.set_xlim(0, 2)
ax.set_yticks(methods_info_oi["ix"] + 0.5)
ax.set_yticklabels(methods_info_oi["label"])

ax.set_xticks(np.arange(2) + 0.5)
ax.set_xticklabels(["Jaccard\npositions", "F1\nregions"])

grid_cbars = grid[0, 1] = polyptich.grid.Grid(padding_height=0.7)
panel_jaccard = grid_cbars[0, 0] = polyptich.grid.Ax((1, 0.1))
ax = panel_jaccard.ax
cax = plt.colorbar(
    mpl.cm.ScalarMappable(norm_jaccard, cmap_jaccard),
    cax=ax,
    orientation="horizontal",
    label="Jaccard positions",
    format=mpl.ticker.PercentFormatter(1),
)
panel_jaccard = grid_cbars[1, 0] = polyptich.grid.Ax((1, 0.1))
ax = panel_jaccard.ax
cax = plt.colorbar(
    mpl.cm.ScalarMappable(norm_f1, cmap_f1),
    cax=ax,
    orientation="horizontal",
    label="F1 regions",
    format=mpl.ticker.PercentFormatter(1),
)

fig.plot()
# fig.savefig("figures/peakcaller_comparison.pdf")

# %% [markdown]
# ### Plot all

# %%
f1 = peakpair_scores_all["region_f1"].unstack()
np.fill_diagonal(f1.values, 1)
jac = peakpair_scores_all["position_jaccard"].unstack()
np.fill_diagonal(jac.values, 1)

# define distance
distance = 1 - f1.values
# distance = 2 - np.stack([np.corrcoef(f1), np.corrcoef(jac)]).mean(0)
distance = (distance + distance.T) / 2
np.fill_diagonal(distance, 0)

# cluster
dist = scipy.spatial.distance.squareform(distance)
clustering = scipy.cluster.hierarchy.linkage(dist, method="single")

methods_info["ix"] = pd.Series(
    np.arange(len(f1)), f1.index[scipy.cluster.hierarchy.leaves_list(clustering)]
).reindex(methods_info.index)

# %%
norm_jaccard = mpl.colors.Normalize(0, 1)
norm_f1 = mpl.colors.Normalize(0, 1)

cmap_jaccard = mpl.cm.Blues
cmap_f1 = mpl.cm.Reds

grid = polyptich.grid.Grid()
fig = polyptich.grid.Figure(grid)

resolution = 0.1
dim = (resolution * len(design), resolution * len(design))

panel_main = grid[0, 0] = polyptich.grid.Ax(dim)
ax = panel_main.ax
for peakcaller1, peakcaller2 in itertools.combinations(list(methods_info.index), 2):
    if (peakcaller1, peakcaller2) in peakpair_scores_all.index:
        pairscore = peakpair_scores_all.loc[peakcaller1, peakcaller2]
        x1 = methods_info.loc[peakcaller1, "ix"]
        x2 = methods_info.loc[peakcaller2, "ix"]

        x, y = (x1, x2) if (x1 > x2) else (x2, x1)
        rect = mpl.patches.Rectangle(
            (x, y),
            1,
            1,
            fc=cmap_jaccard(norm_jaccard(pairscore["position_jaccard"])),
            lw=0,
        )
        ax.add_patch(rect)

        y, x = (x1, x2) if (x1 > x2) else (x2, x1)
        rect = mpl.patches.Rectangle(
            (x, y), 1, 1, fc=cmap_f1(norm_f1(pairscore["region_f1"])), lw=0
        )
        ax.add_patch(rect)
ax.set_xlim(0, len(methods_info))
ax.set_ylim(0, len(methods_info))
ax.set_xticks(methods_info["ix"] + 0.5)
ax.set_yticks(methods_info["ix"] + 0.5)
ax.set_xticklabels(methods_info["label"], rotation=90, ha="left", fontsize=8)
ax.set_yticklabels(methods_info["label"], fontsize=8)
# rect = mpl.patches.Rectangle((0, 0)
ax.xaxis.tick_top()

ax.tick_params(axis="both", which="both", length=1, pad=2)
ax.invert_yaxis()

# make tick containing "ChromatinHD" bold
ax.get_xticklabels()[methods_info.index.get_loc(method_oi)].set_weight("bold")
ax.get_yticklabels()[methods_info.index.get_loc(method_oi)].set_weight("bold")

# rectangle around the column of method_oi
rect = mpl.patches.Rectangle(
    (methods_info.loc[method_oi, "ix"], 0),
    1,
    len(methods_info),
    fc="none",
    ec="black",
    lw=1,
    ls="dashed",
)
ax.add_patch(rect)

# rectangle around the row of method_oi
rect2 = mpl.patches.Rectangle(
    (0, methods_info.loc[method_oi, "ix"]),
    len(methods_info),
    1,
    fc="none",
    ec="black",
    lw=1,
    ls="dashed",
)
ax.add_patch(rect2)


grid_cbars = grid[0, 1] = polyptich.grid.Grid(padding_height=0.7)
panel_jaccard = grid_cbars[0, 0] = polyptich.grid.Ax((1, 0.1))
ax = panel_jaccard.ax
cax = plt.colorbar(
    mpl.cm.ScalarMappable(norm_jaccard, cmap_jaccard),
    cax=ax,
    orientation="horizontal",
    label="Jaccard positions",
    format=mpl.ticker.PercentFormatter(1),
)
panel_jaccard = grid_cbars[1, 0] = polyptich.grid.Ax((1, 0.1))
ax = panel_jaccard.ax
cax = plt.colorbar(
    mpl.cm.ScalarMappable(norm_f1, cmap_f1),
    cax=ax,
    orientation="horizontal",
    label="F1 regions",
    format=mpl.ticker.PercentFormatter(1),
)

fig.plot()

manuscript.save_figure(fig, "4", "overlap_all_methods")

# %%
