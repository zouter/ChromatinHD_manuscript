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

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")


# %%
from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_method_combinations as design,
)


# %%
promoter_name = "10k10k"


# %%
def get_score_folder(x):
    return (
        chd.get_output()
        / "prediction_likelihood"
        / x.dataset
        / promoter_name
        / x.latent
        / str(x.method)
        / "scoring"
        / x.peakcaller
        / x.diffexp
    )


design["score_folder"] = design.apply(get_score_folder, axis=1)

# %%
# design = design.query("dataset != 'alzheimer'")
# design = design.query("dataset == 'GSE198467_H3K27ac'").copy()
# design = design.query("enricher == 'cluster_vs_clusters'")
# design = design.query("enricher == 'cluster_vs_background'")

# %% [markdown]
# ## Aggregate

# %%
window = {"10k10k": np.array([-10000, 10000])}[promoter_name]

# %%
def load_peakresult(dataset, promoter, latent, diffexp, peakcaller, **kwargs):
    scores_dir = (
        chd.get_output()
        / "prediction_differential"
        / dataset
        / promoter
        / latent
        / diffexp
        / peakcaller
    )
    peakresult = pickle.load((scores_dir / "slices.pkl").open("rb"))
    return peakresult


class Prediction(chd.flow.Flow):
    pass


def load_regionresult(dataset, promoter, latent, method, peakcaller, diffexp, **kwargs):
    prediction = Prediction(
        chd.get_output()
        / "prediction_likelihood"
        / dataset
        / promoter
        / latent
        / method
    )
    scores_dir = prediction.path / "scoring" / peakcaller / diffexp
    regionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))
    return regionresult


# %%
for ix, design_row in design.iterrows():
    pass

# %%
regionresult = load_regionresult(**design_row.to_dict())
peakresult = load_peakresult(**design_row.to_dict())

# %%
from region_vs_chhd import RegionVsPeak

# %%
regionvspeak = RegionVsPeak(regionresult, peakresult, window)
regionvspeak.calculate_overlap()
regionvspeak.plot_overlap()

# %%
rows = pd.DataFrame(
    index=pd.MultiIndex.from_frame(
        design[["dataset", "promoter", "latent"]].drop_duplicates()
    )
)
columns = pd.DataFrame(
    index=pd.MultiIndex.from_frame(design[["peakcaller", "diffexp"]].drop_duplicates())
)
columns["label"] = np.where(
    ~pd.isnull(columns.index.get_level_values("peakcaller")),
    (
        chdm.peakcallers.reindex(columns.index.get_level_values("peakcaller"))[
            "label"
        ].reset_index(drop=True)
        + " ("
        + chdm.diffexps.reindex(columns.index.get_level_values("diffexp"))[
            "label_short"
        ].reset_index(drop=True)
        + ")"
    ).values,
    "ChromatinHD differential",
)

rows["ix"] = np.arange(len(rows))
columns["ix"] = np.arange(len(columns))

# %%
design["y"] = design.apply(
    lambda x: rows.loc[(x.dataset, x.promoter, x.latent), "ix"], axis=1
)
design["x"] = design.apply(
    lambda x: columns.loc[(x.peakcaller, x.diffexp), "ix"], axis=1
)
design["x_label"] = design.apply(
    lambda x: columns.loc[(x.peakcaller, x.diffexp), "label"], axis=1
)

# %%
for ix, design_row in design.iterrows():
    if ("regionvspeak" not in design_row.keys()) or (
        pd.isnull(design_row.loc["regionvspeak"])
    ):
        print(ix)
        try:
            regionresult = load_regionresult(**design_row.to_dict())
            peakresult = load_peakresult(**design_row.to_dict())
        except FileNotFoundError:
            print(design_row)
        regionvspeak = RegionVsPeak(regionresult, peakresult, window)
        regionvspeak.calculate_overlap()
        design.loc[ix, "regionvspeak"] = regionvspeak

# %%
design = design.query("dataset != 'alzheimer'")

# %%
fig = polyptich.grid.Figure(
    polyptich.grid.Grid(len(rows), len(columns), padding_height=0, padding_width=0)
)

for ix, design_row in design.iterrows():
    regionvspeak = design_row["regionvspeak"]

    if not pd.isnull(regionvspeak):
        fig.main[design_row.y, design_row.x] = panel = polyptich.grid.Panel((0.5, 0.5))
        ax = panel.ax
        # if design_row.y == 0:
        #     ax.set_title(design_row.dataset)

        regionvspeak.plot_overlap(ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if design_row.y == 0:
            ax.set_xlabel(
                design_row["x_label"],
                rotation=-30,
                ha="right",
                va="bottom",
            )
            ax.xaxis.set_label_position("top")
        if design_row.x == 0:
            ax.set_ylabel(design_row.dataset, rotation=0, ha="right", va="bottom")
fig.plot()

manuscript.save_figure(fig, "4", "peak_vs_chhd_positional_all")

# %%
