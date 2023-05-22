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
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name


prediction_name = "v20_initdefault"; splitter = "permutations_5fold5repeat"; promoter_name = "100k100k"; window = np.array([-100000, 100000])
# prediction_name = "v20_initdefault"; splitter = "random_5fold"; promoter_name = "10k10k"; window = np.array([-10000, 10000])
# prediction_name = "v20"; splitter = "permutations_5fold5repeat"; promoter_name = "10k10k"; window = np.array([-10000, 10000])

transcriptome = chd.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

fragments = chd.data.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# create design to run
from design import get_design, get_folds_inference

class Prediction(chd.flow.Flow):
    pass


# %%
prediction = Prediction(chd.get_output() / "prediction_positional" / dataset_name / promoter_name / splitter / prediction_name)
scorer_folder = (prediction.path / "scoring" / "size")

# %%
size_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

# %% [markdown]
# ### Global view

# %%
scores = size_scoring.genescores.mean("gene").mean("model")


# %%
def zscore(x, dim = 0, mean = None):
    if mean is None:
        mean = x.values.mean(dim, keepdims = True)
    return (x - mean)/x.values.std(dim, keepdims = True)
def minmax(x, dim = 0):
    return (x - x.values.min(dim, keepdims = True))/(x.max(dim, keepdims = True) - x.min(dim, keepdims = True))


# %%
fig, ax_perc = plt.subplots(figsize = (4, 4))

ax_perc.axhline(0, dashes = (2, 2), color = "grey")

ax_mse = ax_perc.twinx()
ax_mse.plot(scores.coords["window"], scores["deltacor"].sel(phase = "validation"), color = chd.plotting.colors[0])
ax_mse.set_ylabel(r"$\Delta$ cor", rotation = 0, ha = "left", va = "center", color = chd.plotting.colors[0])
ax_mse.invert_yaxis()
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["validation"]) 
ax_perc.set_xlabel("Fragment length")

ax_perc.plot(scores.coords["window"], 1-scores["retained"].sel(phase = "validation"), color = chd.plotting.colors[1])
ax_perc.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center", color = chd.plotting.colors[1])
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax_perc.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_mse.yaxis.set_major_locator(plt.MaxNLocator(3))

sns.despine(ax = ax_mse)
sns.despine(ax = ax_perc, right = False)

# %%
deltacor_mean = (scores["deltacor"] * (1-scores["retained"])).sum("window") / (1-scores["retained"]).sum("window")
retained_mean = (scores["retained"] * (1-scores["retained"])).sum("window") / (1-scores["retained"]).sum("window")

# %%
scores["normeffect"] = scores["effect"] / scores["lost"]
scores["normdeltacor"] = scores["deltacor"] / scores["lost"]

# %%
scores["normdeltacor"].sel(phase = "test").plot()
scores["normdeltacor"].sel(phase = "validation").plot()

# %%
fig, ax = plt.subplots(figsize = (2, 2))

plotdata = pd.DataFrame({
    "normdeltacor": (scores["normdeltacor"].sel(phase = "validation") + scores["normdeltacor"].sel(phase = "test"))/2,
    "normeffect": (scores["normeffect"].sel(phase = "validation") + scores["normeffect"].sel(phase = "test"))/2,
    "window_mid":scores["normdeltacor"].sel(phase = "validation").coords["window"].to_pandas(),
})


ax.plot(plotdata["window_mid"], plotdata["normdeltacor"] * 1000, color = "black")

ax.set_ylim(0, plotdata["normdeltacor"].min() * 1.05 * 1000)

import scipy
relmax = plotdata["window_mid"].iloc[scipy.signal.argrelmax(plotdata["normdeltacor"].values, axis=0)[0]].tolist()
relmax = relmax[:-1]
relmax = [plotdata["window_mid"].iloc[0], *relmax, plotdata["window_mid"].iloc[-1]]
ax.set_xticks(relmax)

twinx = ax.twiny()
twinx.set_xlim(ax.get_xlim())
relmin = plotdata["window_mid"].iloc[scipy.signal.argrelmin(plotdata["normdeltacor"].values, axis=0)[0]].tolist()
relmin = [*relmin[:-2], relmin[-1]]
twinx.set_xticks(relmin)

for relmax_ in relmax:
    ax.plot([relmax_, relmax_], [ax.get_ylim()[0], plotdata.loc[relmax_, "normdeltacor"] * 1000], color = "#333333", linestyle = "--", linewidth = 0.5)

for relmin_ in relmin:
    ax.plot([relmin_, relmin_], [ax.get_ylim()[1], plotdata.loc[relmin_, "normdeltacor"] * 1000], color = "#333333", linestyle = "--", linewidth = 0.5)
ax.set_xlabel("Fragment length ± 10bp")
ax.set_ylabel("$\\Delta$ cor per\n1000 fragments", rotation = 0, ha = "right", va = "center")

twiny = ax.twinx()
twiny.plot(plotdata["window_mid"], plotdata["normeffect"] * 1000, color = "red")
twiny.set_ylim(0, plotdata["normeffect"].min() * 1.05 * 1000)
twiny.tick_params(axis='y', colors='red')
twiny.set_ylabel("Effect per \n1000 fragments", rotation = 0, ha = "left", va = "center", color = "red")

manuscript.save_figure(fig, "5", "lengths_vs_normdeltacor")

# %%
scores.sel(phase = "validation").to_pandas().style.bar(subset = ["deltacor", "normeffect", "normdeltacor", "lost"])

# %%
plotdata = pd.DataFrame({
    "deltacor": scores["deltacor"].sel(phase = "validation").values,
    "retained": scores["retained"].sel(phase = "validation").values,
    "censored": 1-scores["retained"].sel(phase = "validation").values,
})
plt.plot(
    np.log(plotdata["censored"]),
    np.log(np.abs(scores["deltacor"].sel(phase = "validation"))),
    # c = scores["window"],
)

# %%
fig, ax = plt.subplots(figsize = (4, 4))
scores["reldeltacor"] = ((zscore(scores["deltacor"], 1, mean = deltacor_mean) - zscore(scores["retained"], 1, mean = retained_mean)))
scores["reldeltacor"].to_pandas().loc["test"].plot()
scores["reldeltacor"].to_pandas().loc["validation"].plot()
scores["reldeltacor"].to_pandas().loc["train"].plot()
ax.set_ylabel("Relative\n$\\Delta$ cor", rotation = 0, ha = "right", va = "center")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_inverted(True)
ax.axhline(0, dashes = (2, 2), color = "#333333")

# %%
scores.to_pickle(scores_dir / "scores.pkl")

# %% [markdown]
# -------------------------
# %% [markdown]
# Do (mono/di/...)"nucleosome" fragments still have an overall positive or negative effect on gene expression?

# %%
fig, ax_perc = plt.subplots(figsize = (4, 4))

ax_perc.axhline(0, dashes = (2, 2), color = "grey")

ax_mse = ax_perc.twinx()
ax_mse.plot(effect.index, scores.loc["validation", "effect"], color = chd.plotting.colors[0])
ax_mse.set_ylabel(r"effect", rotation = 0, ha = "left", va = "center", color = chd.plotting.colors[0])
ax_mse.invert_yaxis()
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["validation"]) 
ax_perc.set_xlabel("Fragment length")

ax_perc.plot(retained.index, 1-retained["validation"], color = chd.plotting.colors[1])
ax_perc.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center", color = chd.plotting.colors[1])
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax_perc.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_mse.yaxis.set_major_locator(plt.MaxNLocator(3))

sns.despine(ax = ax_mse)
sns.despine(ax = ax_perc, right = False)

# %% [markdown]
# ## Across datasets

# %%
dataset_names = [
    "pbmc10k",
    "e18brain",
    "lymphoma"
]

# %%
datasets = pd.DataFrame({"dataset":dataset_names}).set_index("dataset")
datasets["color"] = sns.color_palette("Set1", datasets.shape[0])

# %%
relative_deltacor = {}

fig, (ax, ax1) = plt.subplots(1, 2, figsize = (8, 4), gridspec_kw={"wspace":0.6})

for dataset_name in dataset_names:
    prediction = Prediction(chd.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)
    scores_dir = (prediction.path / "scoring" / "lengths")
    
    scores = pd.read_pickle(scores_dir / "scores.pkl")
    
    ax.plot(scores.loc["validation"].index, -scores.loc["validation"]["reldeltacor"], label = dataset_name, color = datasets.loc[dataset_name, "color"])
    ax1.plot(scores.loc["validation"].index, 1-scores.loc["validation"]["retained"], label = dataset_name, color = datasets.loc[dataset_name, "color"])
ax.axhline(0, dashes = (2, 2), color = "#333333")
ax.set_xlabel("Fragment length")
ax.set_ylabel("Relative\n$\\Delta$ cor", rotation = 0, ha = "right", va = "center")
ax.legend(title = "dataset")
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax1.set_xlabel("Fragment length")
ax1.set_ylabel("% Fragments", rotation = 0, ha = "right", va = "center")
sns.despine()

# %%
for 
relative_deltacor

# %%

# %%

# %%
