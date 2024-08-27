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
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

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

dataset_name = "pbmc10k"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

splitter = "5x5"
regions_name = "100k100k"
prediction_name = "v33"
layer = "magic"

fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)


# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "pred"
    / dataset_name
    / regions_name
    / splitter
    / layer
    / prediction_name
)

# %%
size = chd.models.pred.interpret.Size(prediction.path / "scoring" / "size")

# %%
genes_oi = size.scores["scored"].sel_xr().all("fold").to_pandas()
genes_oi = genes_oi.index[genes_oi]

# %%
prediction_reference = chd.flow.Flow(
    chd.get_output() / "pred" / dataset_name / regions_name / "5x1" / layer / "v33"
)
performance = chd.models.pred.interpret.Performance(prediction_reference / "scoring" / "performance")

# %%
len(genes_oi)

# %% [markdown]
# ### Global view

# %%
# x = size.scores["reldeltacor"].sel_xr(genes_oi).sel(phase = "test").mean("fold").to_pandas()
# y = size.scores["lost"].sel_xr(genes_oi).sel(phase = "test").mean("fold").to_pandas()
# lost = size.scores["lost"].sel_xr(genes_oi)
# censored = (lost/lost.sum("window")).sel(phase = "test").mean("fold").to_pandas()

# %%
lengthscores = pd.DataFrame({
    "reldeltacor":size.scores["reldeltacor"].sel_xr(genes_oi).sel(phase = "test").mean("gene").mean("fold").to_pandas(),
    "deltacor":size.scores["deltacor"].sel_xr(genes_oi).sel(phase = "test").mean("gene").mean("fold").to_pandas(),
    "lost":size.scores["lost"].sel_xr(genes_oi).sel(phase = "test").mean("gene").mean("fold").to_pandas(),
    "censored":size.scores["censored"].sel_xr(genes_oi).sel(phase = "test").mean("gene").mean("fold").to_pandas(),
    # "censored":censored.mean(0),
    "effect":size.scores["effect"].sel_xr(genes_oi).sel(phase = "test").mean("gene").mean("fold").to_pandas()
})
lengthscores["window_mid"] = lengthscores.index
scores = lengthscores


# %%
def zscore(x, dim=0, mean=None):
    if mean is None:
        mean = x.values.mean(dim, keepdims=True)
    return (x - mean) / x.values.std(dim, keepdims=True)


def minmax(x, dim=0):
    return (x - x.values.min(dim, keepdims=True)) / (
        x.max(dim, keepdims=True) - x.min(dim, keepdims=True)
    )


# %%
fig, ax_perc = plt.subplots(figsize=(4, 4))

ax_perc.axhline(0, dashes=(2, 2), color="grey")

ax_mse = ax_perc.twinx()
ax_mse.plot(
    scores.index,
    scores["deltacor"],
    # color=chd.plot.colors[0],
)
ax_mse.set_ylabel(
    r"$\Delta$ cor", rotation=0, ha="left", va="center"
)
ax_mse.invert_yaxis()
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["validation"])
ax_perc.set_xlabel("Fragment length")

ax_perc.plot(
    scores.index,
    scores["lost"],
    # color=chd.plot.colors[1],
)
ax_perc.set_ylabel(
    "% Fragments", rotation=0, ha="right", va="center"
)
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax_perc.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_mse.yaxis.set_major_locator(plt.MaxNLocator(3))

sns.despine(ax=ax_mse)
sns.despine(ax=ax_perc, right=False)

# %%
lengthscores["normeffect"] = lengthscores["effect"] / lengthscores["lost"]
lengthscores["normdeltacor"] = np.abs(lengthscores["deltacor"] / (lengthscores["lost"]))
lengthscores["normdeltacor2"] = np.abs(lengthscores["reldeltacor"] / (lengthscores["lost"]))
# lengthscores["normdeltacor"] = lengthscores["deltacor"] / lengthscores["censored"]

# %%
lengthscores.style.bar()

# %%
# Find max and min values
import scipy

# relmax = (
#     lengthscores["window_mid"]
#     .iloc[scipy.signal.argrelmax(lengthscores["normdeltacor"].values, axis=0)[0]]
#     .tolist()
# )
# relmax = relmax[:-1]
# relmax = [
#     lengthscores["window_mid"].iloc[0],
#     *relmax,
#     lengthscores["window_mid"].iloc[-1],
# ]
# relmin = (
#     lengthscores["window_mid"]
#     .iloc[scipy.signal.argrelmin(lengthscores["normdeltacor"].values, axis=0)[0]]
#     .tolist()
# )
# relmin = [*relmin[:-2], relmin[-1]]

relmax = [10, 170, 410, 590]
relmin = [110, 270, 470 ,690]

# %%
# scores.sel(phase="validation").to_pandas().style.bar(
#     subset=["deltacor", "normeffect", "normdeltacor", "lost"]
# )

# %%
plotdata = lengthscores
norm = mpl.colors.Normalize(vmin=0, vmax=plotdata.index.max())
cmap = mpl.cm.get_cmap("viridis_r")

fig, ax = plt.subplots(figsize=(3.2, 2.7))
ax.plot(
    -plotdata["lost"],
    -plotdata["deltacor"],
    color="#AAAAAA",
    zorder=0,
)
ax.scatter(
    -plotdata["lost"],
    -plotdata["deltacor"],
    c=cmap(norm(plotdata.index)),
    marker="o",
    s=8,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("$\\Delta$ cor")
ax.set_xlabel("# fragments per 1000 cells")
# ax.set_aspect(1)

# annotate position

# retained_min = plotdata.sort_values("retained").index[0]
# positions = [*relmin, *relmax, retained_min]
# relmin = [10, 110, 270, 470, 690]
# relmax = [50, 170, 390, 590]
positions = [*relmin, *relmax, 770]
for pos in positions:
    # some position-specific changes to make it look nice
    # change this if some annotations overlap
    stroke = False
    if pos == relmax[0]:
        xytext = (-5, 10)
        ha = "left"
        stroke = True
    elif pos == relmin[-1]:
        xytext = (0, 10)
        ha = "left"
    elif pos in relmin:
        xytext = (-10, 0)
        ha = "right"
    elif pos == positions[-1]:
        xytext = (10, 0)
        ha = "left"
        stroke = True
    else:
        xytext = (10, 0)
        ha = "left"

    color = cmap(norm(pos))
    text = ax.annotate(
        f"{int(pos)}",
        xy=(-plotdata.loc[pos, "lost"], -plotdata.loc[pos, "deltacor"]),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="center",
        color=color,
        fontsize=10,
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            shrinkA=0,
            shrinkB=0,
        ),
        bbox=dict(
            boxstyle="round",
            fc="white",
            alpha=0.8,
            pad=0.0,
        ),
        zorder=5,
        fontweight = "bold" if stroke else "normal",
    )
    if stroke:
        pass
        # text.set_path_effects(
        #     [
        #         mpl.patheffects.Stroke(linewidth=1, foreground="k"),
        #         mpl.patheffects.Normal(),
        #     ]
        # )

cax = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    label="Fragment length ± 10bp",
    pad=0.1,
)
manuscript.save_figure(fig, "6", "lengths_vs_deltacor")

# %%
# extrema_interleaved = np.empty((len(relmax) + len(relmin),), dtype=np.int64)
# extrema_interleaved[::2] = relmax
# extrema_interleaved[1::2] = relmin
extrema_interleaved = [10, 110, 170, 270, 390, 470, 590, 690, 770]
cuts = [0, *(extrema_interleaved[:-1] + np.diff(extrema_interleaved) / 2), 99999]

sizes = pd.DataFrame(
    {
        "start": cuts[:-1],
        "end": cuts[1:],
        "length": np.diff(cuts),
        "mid": [*(cuts[:-2] + np.diff(cuts)[:-1] / 2), cuts[-2] + 30],
        "label": [
            "TF footprint",
            "submono",
            "mono",
            "supermono",
            "di",
            "superdi",
            "tri",
            "supertri",
            "multi",
        ],
        "label_short": [
            "TF\nfootprint",
            "mono-",
            "mono",
            "mono+",
            "di",
            "di+",
            "tri",
            "tri+",
            "multi",
        ],
    }
)

# %%
width = 3
padding_height = 0.1

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=padding_height))

# abundance
panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 0.5)))
ax.plot(lengthscores["window_mid"], -lengthscores["lost"], zorder=0, color="#333")
ax.scatter(
    lengthscores["window_mid"],
    -lengthscores["lost"],
    c=cmap(norm(plotdata.index)),
    s=8,
    zorder=10,
)
ax.set_ylim(0, -lengthscores["lost"].min() * 1.05)
ax.tick_params(axis="y")
ax.set_ylabel("# fragments per\n1000 cells", rotation=0, ha="right", va="center")
ax.set_xlabel("Fragment length ± 10bp")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.xaxis.set_ticks_position("top")

# predictivity (delta cor)
panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 1.5)))

ax.plot(lengthscores["window_mid"], lengthscores["normdeltacor"] * 1000, color="black")
ax.scatter(
    lengthscores["window_mid"],
    lengthscores["normdeltacor"] * 1000,
    c=cmap(norm(plotdata.index)),
    s=8,
    zorder=10,
)

ax.set_ylim(0, lengthscores["normdeltacor"].max() * 1.25 * 1000)

ax.set_xticks(relmax)
ax.set_xticks([])

for relmax_ in relmax:
    ax.plot(
        [relmax_, relmax_],
        [ax.get_ylim()[0], lengthscores.loc[relmax_, "normdeltacor"] * 1000],
        color="#333333",
        linestyle="--",
        linewidth=0.5,
    )
    ax.annotate(
        f"{int(relmax_)}",
        xy=(relmax_, ax.get_ylim()[0]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        # arrowprops = dict(
        #     arrowstyle = "-",
        #     color = "#333333",
        #     shrinkA = 0,
        #     shrinkB = 0,
        # ),
        bbox=dict(
            boxstyle="round",
            fc="white",
            alpha=0.8,
            pad=0.0,
        ),
    )

for relmin_ in relmin:
    ax.plot(
        [relmin_, relmin_],
        [ax.get_ylim()[1], lengthscores.loc[relmin_, "normdeltacor"] * 1000],
        color="#333333",
        linestyle="--",
        linewidth=0.5,
    )
    ax.annotate(
        f"{int(relmin_)}",
        xy=(relmin_, ax.get_ylim()[1]),
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
        # arrowprops = dict(
        #     arrowstyle = "-",
        #     color = "#333333",
        #     shrinkA = 0,
        #     shrinkB = 0,
        # ),
        bbox=dict(
            boxstyle="round",
            fc="white",
            alpha=0.8,
            pad=0.0,
        ),
    )
ax.set_ylabel("$\\Delta$ cor per\n1000 fragments", rotation=0, ha="right", va="center")

# effect
panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 0.5)))
ax.plot(
    lengthscores["window_mid"],
    lengthscores["normeffect"] * 1000,
    zorder=0,
    color="#333",
)
ax.scatter(
    lengthscores["window_mid"],
    lengthscores["normeffect"] * 1000,
    c=cmap(norm(plotdata.index)),
    s=8,
    zorder=10,
)
ax.set_ylim(0, lengthscores["normeffect"].max() * 1.05 * 1000)
ax.tick_params(axis="y")
ax.set_ylabel("Effect per \n1000 fragments", rotation=0, ha="right", va="center")
ax.set_xticks([])
xlim = ax.get_xlim()

# types
panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 0.5)))
texts = []
adjustments = np.array([-30, -15, -5, 0, 0, 0, 0, 0, 5])
for i, size in sizes.iterrows():
    color = cmap(norm(size["mid"]))
    text = ax.annotate(
        size["label_short"],
        xy=(size["mid"], 0.5),
        xytext=(adjustments[i], -15),
        xycoords="data",
        textcoords="offset points",
        ha="center",
        va="top",
        color="black",
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    texts.append(text)
    # text.set_path_effects([
    #     mpl.patheffects.Stroke(linewidth=1, foreground="black"),
    #     mpl.patheffects.Normal()
    # ])
    rectangle = mpl.patches.Rectangle(
        (size["start"], 0.5),
        # (size["start"], 0.5 - (i % 2) * 0.2),
        size["length"],
        0.5,
        color=color,
        zorder=0,
        ec="white",
    )
    ax.add_patch(rectangle)
import adjustText

ax.set_xlim(*xlim)
ax.set_ylim(-0.5, 0.8)
ax.set_xticks([])
ax.axis("off")

fig.plot()
# adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), expand_text=(0, 10.1), only_move={'text': 'y+'}, avoid_self = False)

manuscript.save_figure(fig, "6", "lengths_vs_normdeltacor")

# %%
