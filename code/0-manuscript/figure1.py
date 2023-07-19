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
fig = chd.grid.Figure(chd.grid.Grid())

import chromatinhd.differential.plot

import scipy

x = np.linspace(0, 1, 100)[1:]
prob_oi = scipy.stats.norm.pdf((x - 0.5) * 5) * 3
prob_diff = scipy.stats.norm.pdf((x - 0.5) * 4) * 3.5

plotdata_genome = pd.DataFrame(
    {"prob_diff": prob_diff, "cluster": np.repeat(0, len(x)), "position": x}
).set_index(["cluster", "position"])
plotdata_genome_mean = pd.DataFrame({"prob": prob_oi, "position": x}).set_index(
    "position"
)
plotdata_genome["prob"] = (
    plotdata_genome_mean.loc[
        plotdata_genome.index.get_level_values("position"), "prob"
    ].values
    + plotdata_genome["prob_diff"]
)
cluster_info = pd.DataFrame({"cluster": ["A"], "dimension": [0]}).set_index("cluster")
window = np.array([0.05, 1])
width = 0.5
height = 0.5
panel = fig.main.add_under(
    chd.models.diff.plot.Differential(
        plotdata_genome=plotdata_genome,
        plotdata_genome_mean=plotdata_genome_mean,
        cluster_info=cluster_info,
        window=window,
        width=width,
        panel_height=height,
        title=False,
    )
)
panel.elements[0].ax.spines["top"].set_visible(False)
panel.elements[0].ax.spines["right"].set_visible(False)

fig.plot()

manuscript.save_figure(fig, "1", "differential_simple", dpi=300)
# %%
fig = chd.grid.Figure(chd.grid.Grid())

import chromatinhd.differential.plot

import scipy

window = np.array([0.05, 2])
x = np.linspace(0, 2, 100)[1:]
prob_oi = (
    scipy.stats.norm.pdf((x - 1) * 2.5) + scipy.stats.norm.pdf((x - 1.5) * 2.5)
) * 4.0
prob_diff = scipy.stats.norm.pdf((x - 0.5) * 4) * 4.5

plotdata_genome = pd.DataFrame(
    {"prob_diff": prob_diff, "cluster": np.repeat(0, len(x)), "position": x}
).set_index(["cluster", "position"])
plotdata_genome_mean = pd.DataFrame({"prob": prob_oi, "position": x}).set_index(
    "position"
)
plotdata_genome["prob"] = (
    plotdata_genome_mean.loc[
        plotdata_genome.index.get_level_values("position"), "prob"
    ].values
    + plotdata_genome["prob_diff"]
)
cluster_info = pd.DataFrame({"cluster": ["A"], "dimension": [0]}).set_index("cluster")

width = 0.5
height = 0.5
panel = fig.main.add_under(
    chd.models.diff.plot.Differential(
        plotdata_genome=plotdata_genome,
        plotdata_genome_mean=plotdata_genome_mean,
        cluster_info=cluster_info,
        window=window,
        width=width,
        panel_height=height,
        title=False,
    )
)
panel.elements[0].ax.spines["top"].set_visible(False)
panel.elements[0].ax.spines["right"].set_visible(False)

fig.plot()

manuscript.save_figure(fig, "1", "differential_assymetric", dpi=300)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

import chromatinhd.differential.plot

import scipy

window = np.array([0.05, 2])
x = np.linspace(0, 2, 100)[1:]
prob_oi = (scipy.stats.norm.pdf((x - 1) * 2.5)) * 5.0
prob_diff = (
    scipy.stats.norm.pdf((x - 0.0) * 2) * 3.5
    + scipy.stats.norm.pdf((x - 1.8) * 10) * 5.0
)

plotdata_genome = pd.DataFrame(
    {"prob_diff": prob_diff, "cluster": np.repeat(0, len(x)), "position": x}
).set_index(["cluster", "position"])
plotdata_genome_mean = pd.DataFrame({"prob": prob_oi, "position": x}).set_index(
    "position"
)
plotdata_genome["prob"] = (
    plotdata_genome_mean.loc[
        plotdata_genome.index.get_level_values("position"), "prob"
    ].values
    + plotdata_genome["prob_diff"]
)
cluster_info = pd.DataFrame({"cluster": ["A"], "dimension": [0]}).set_index("cluster")

width = 1.0
height = 0.5
panel = fig.main.add_under(
    chd.models.diff.plot.Differential(
        plotdata_genome=plotdata_genome,
        plotdata_genome_mean=plotdata_genome_mean,
        cluster_info=cluster_info,
        window=window,
        width=width,
        panel_height=height,
        title=False,
    )
)
panel.elements[0].ax.spines["top"].set_visible(False)
panel.elements[0].ax.spines["right"].set_visible(False)

fig.plot()

manuscript.save_figure(fig, "1", "differential_multires", dpi=300)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

import chromatinhd.differential.plot

import scipy

window = np.array([0.05, 2])
x = np.linspace(0, 2, 100)[1:]
prob_oi = (scipy.stats.norm.pdf((x - 1) * 2.5)) * 5.0
prob_diff = (
    scipy.stats.norm.pdf((x - 0.0) * 2) * 3.5
    + scipy.stats.norm.pdf((x - 1.8) * 10) * 3.0
    - scipy.stats.norm.pdf((x - 0.6) * 10) * 5.0
)

plotdata_genome = pd.DataFrame(
    {"prob_diff": prob_diff, "cluster": np.repeat(0, len(x)), "position": x}
).set_index(["cluster", "position"])
plotdata_genome_mean = pd.DataFrame({"prob": prob_oi, "position": x}).set_index(
    "position"
)
plotdata_genome["prob"] = (
    plotdata_genome_mean.loc[
        plotdata_genome.index.get_level_values("position"), "prob"
    ].values
    + plotdata_genome["prob_diff"]
)
cluster_info = pd.DataFrame({"cluster": ["A"], "dimension": [0]}).set_index("cluster")

width = 1.0
height = 0.5
panel = fig.main.add_under(
    chd.models.diff.plot.Differential(
        plotdata_genome=plotdata_genome,
        plotdata_genome_mean=plotdata_genome_mean,
        cluster_info=cluster_info,
        window=window,
        width=width,
        panel_height=height,
        title=False,
    )
)
panel.elements[0].ax.spines["top"].set_visible(False)
panel.elements[0].ax.spines["right"].set_visible(False)

fig.plot()

manuscript.save_figure(fig, "1", "differential_differentially", dpi=300)

# %% [markdown]
# # Figure 3
# %%
fig = chd.grid.Figure(chd.grid.Grid())

import chromatinhd.differential.plot

import scipy

cluster_info = pd.DataFrame(
    {"cluster": ["A", "B", "C"], "dimension": [0, 1, 2]}
).set_index("cluster")

window = np.array([0.05, 2])
x = np.linspace(0, 2, 100)[1:]
prob_oi = (scipy.stats.norm.pdf((x - 1) * 2.5)) * 5.0
prob_diff1 = (
    scipy.stats.norm.pdf((x - 0.0) * 2) * 3.5
    + scipy.stats.norm.pdf((x - 1.8) * 10) * 3.0
    - scipy.stats.norm.pdf((x - 0.6) * 10) * 5.0
)
prob_diff2 = (
    scipy.stats.norm.pdf((x - 0.0) * 2) * 3.5
    + scipy.stats.norm.pdf((x - 1.0) * 10) * 3.0
    - scipy.stats.norm.pdf((x - 0.6) * 10) * 6.0
)
prob_diff3 = (
    scipy.stats.norm.pdf((x - 0.0) * 2) * 1.5
    + scipy.stats.norm.pdf((x - 2.0) * 10) * 3.0
    - scipy.stats.norm.pdf((x - 0.6) * 10) * 0.0
)

plotdata_genome = pd.concat(
    [
        pd.DataFrame(
            {"prob_diff": prob_diff1, "cluster": np.repeat(0, len(x)), "position": x}
        ).set_index(["cluster", "position"]),
        pd.DataFrame(
            {"prob_diff": prob_diff2, "cluster": np.repeat(1, len(x)), "position": x}
        ).set_index(["cluster", "position"]),
        pd.DataFrame(
            {"prob_diff": prob_diff3, "cluster": np.repeat(2, len(x)), "position": x}
        ).set_index(["cluster", "position"]),
    ]
)
plotdata_genome_mean = pd.DataFrame({"prob": prob_oi, "position": x}).set_index(
    "position"
)
plotdata_genome["prob"] = (
    plotdata_genome_mean.loc[
        plotdata_genome.index.get_level_values("position"), "prob"
    ].values
    + plotdata_genome["prob_diff"]
)

width = 1.0
height = 0.4
panel = fig.main.add_under(
    chd.models.diff.plot.Differential(
        plotdata_genome=plotdata_genome,
        plotdata_genome_mean=plotdata_genome_mean,
        cluster_info=cluster_info,
        window=window,
        width=width,
        panel_height=height,
        title=False,
    )
)
for element in panel.elements:
    element.ax.spines["top"].set_visible(False)
    element.ax.spines["right"].set_visible(False)

fig.plot()

manuscript.save_figure(fig, "3", "differential_interpretation", dpi=300)

# %%
