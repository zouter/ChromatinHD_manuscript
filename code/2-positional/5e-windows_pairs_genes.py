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

import tqdm.auto as tqdm
import xarray as xr

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

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# prediction_name = "v20_initdefault"

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
# ## Pairwindow

scores = []
for gene in tqdm.tqdm(transcriptome.var.index):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interpolation_file = scores_folder / "interpolated.pkl"

    if interpolation_file.exists():
        scores.append(
            pd.read_pickle(interpolation_file)["interaction"]
            .assign(gene=gene)
            .reset_index()
        )
scores = pd.concat(scores)


# %%
scores.query("qval < 0.05").groupby("gene").first().sort_values(
    "deltacor_interaction", ascending=False
).head(10)

# %%
scores["distance"] = np.abs(scores["window_start1"] - scores["window_start2"])
sns.histplot(scores["distance"], bins=100, stat="density")
sns.histplot(
    scores.loc[scores["deltacor_interaction"] < 0]["distance"],
    bins=100,
    # stat="density",
)
sns.histplot(
    scores.loc[scores["deltacor_interaction"] > 0]["distance"],
    bins=100,
    # stat="density",
)

# %%
scores["distance"] = np.abs(scores["window_start1"] - scores["window_start2"])
bins = np.linspace(0, (window[1] - window[0]), 50)
synergism_distance = pd.DataFrame(
    {
        "total": np.bincount(
            np.digitize(
                scores.query("(qval < 0.05)")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic": np.bincount(
            np.digitize(
                scores.query("(qval < 0.05) & (deltacor_interaction < 0)")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic": np.bincount(
            np.digitize(
                scores.query("(qval < 0.05) & (deltacor_interaction > 0)")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)

synergism_distance["synergistic_relative"] = (
    synergism_distance["synergistic"] / synergism_distance["synergistic"].sum()
)
synergism_distance["antagonistic_relative"] = (
    synergism_distance["antagonistic"] / synergism_distance["antagonistic"].sum()
)

synergism_distance["synergism_relative_ratio"] = (
    synergism_distance["synergistic_relative"]
    / synergism_distance["antagonistic_relative"]
)
synergism_distance["synergism_ratio"] = (
    synergism_distance["synergistic"] / synergism_distance["antagonistic"]
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_distance["bin"][:-1],
    synergism_distance["synergism_ratio"][:-1],
    color="black",
)
ax.set_yscale("log")
ax.set_ylabel("Ratio synergistic\nto antagonistic co-predictivity")
ax.set_xlabel("Distance between windows")
ax.xaxis.set_major_formatter(chd.plotting.distance_ticker)

manuscript.save_figure(fig, "5", "synergism_distance")

# %%
cutoff = 0.00001
bins = np.linspace(window[0], window[1], 50)
synergism_position = pd.DataFrame(
    {
        "total": np.bincount(
            np.digitize(
                scores.query("(qval < 0.05)")[
                    ["window_mid1", "window_mid2"]
                ].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic": np.bincount(
            np.digitize(
                scores.query("(qval < 0.05) & (deltacor_interaction < -@cutoff)")[
                    ["window_mid1", "window_mid2"]
                ].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic": np.bincount(
            np.digitize(
                scores.query("(qval < 0.05) & (deltacor_interaction > @cutoff)")[
                    ["window_mid1", "window_mid2"]
                ].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic_distant": np.bincount(
            np.digitize(
                scores.query(
                    "(qval < 0.05) & (deltacor_interaction < -@cutoff) & (distance > 1000)"
                )[["window_mid1", "window_mid2"]].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic_distant": np.bincount(
            np.digitize(
                scores.query(
                    "(qval < 0.05) & (deltacor_interaction > @cutoff) & (distance > 1000)"
                )[["window_mid1", "window_mid2"]].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)

synergism_position["synergistic_relative"] = (
    synergism_position["synergistic"] / synergism_position["synergistic"].sum()
)
synergism_position["antagonistic_relative"] = (
    synergism_position["antagonistic"] / synergism_position["antagonistic"].sum()
)

synergism_position["synergism_relative_ratio"] = (
    synergism_position["synergistic_relative"]
    / synergism_position["antagonistic_relative"]
)
synergism_position["synergism_ratio"] = (
    synergism_position["synergistic"] / synergism_position["antagonistic"]
)
synergism_position["synergism_distant_ratio"] = (
    synergism_position["synergistic_distant"]
    / synergism_position["antagonistic_distant"]
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_position["bin"][:-1],
    synergism_position["total"][:-1],
    color="black",
)
ax.plot(
    synergism_position["bin"][:-1],
    synergism_position["synergism_ratio"][:-1],
    color="black",
)
ax.plot(
    synergism_position["bin"][:-1],
    synergism_position["synergism_distant_ratio"][:-1],
    color="orange",
)

ax.set_yscale("log")
ax.set_ylabel("Ratio synergistic\nto antagonistic co-predictivity")
ax.xaxis.set_major_formatter(chd.plotting.gene_ticker)
ax.set_xlabel("Distance to TSS")
ax_position = ax

manuscript.save_figure(fig, "5", "synergism_position")

# %%
def interacts_with_tss(scores):
    return pd.concat(
        [
            scores.query(
                "(qval < 0.05)& (distance > 1000) & (window_mid1 > -500) & (window_mid1 < 500)"
            ).assign(window_mid=lambda x: x.window_mid2)[
                ["window_mid", "deltacor_interaction"]
            ],
            scores.query(
                "(qval < 0.05) & (distance > 1000) & (window_mid2 > -500) & (window_mid2 < 500)"
            ).assign(window_mid=lambda x: x.window_mid1)[
                ["window_mid", "deltacor_interaction"]
            ],
        ]
    )


bins = np.linspace(window[0], window[1], 50)
synergism_tss = pd.DataFrame(
    {
        "synergistic": np.bincount(
            np.digitize(
                interacts_with_tss(scores)
                .query("(deltacor_interaction < 0)")["window_mid"]
                .values,
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic": np.bincount(
            np.digitize(
                interacts_with_tss(scores)
                .query("(deltacor_interaction > 0)")["window_mid"]
                .values,
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)

synergism_tss["synergistic_relative"] = (
    synergism_tss["synergistic"] / synergism_tss["synergistic"].sum()
)
synergism_tss["antagonistic_relative"] = (
    synergism_tss["antagonistic"] / synergism_tss["antagonistic"].sum()
)

synergism_tss["synergism_relative_ratio"] = (
    synergism_tss["synergistic_relative"] / synergism_tss["antagonistic_relative"]
)
synergism_tss["synergism_ratio"] = (
    synergism_tss["synergistic"] / synergism_tss["antagonistic"]
)


# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_position["bin"][:-1],
    synergism_position["synergism_ratio"][:-1],
    color="#ccc",
)
ax.plot(
    synergism_tss["bin"][:-1],
    synergism_tss["synergism_ratio"][:-1],
    color="black",
)

ax.set_yscale("log")
ax.set_ylabel("Ratio synergistic\nto antagonistic co-predictivity")
ax.xaxis.set_major_formatter(chd.plotting.gene_ticker)
ax.set_xlabel("Distance to TSS")
ax.set_ylim(ax_position.get_ylim())

manuscript.save_figure(fig, "5", "synergism_tss")

# %% [markdown]
# ----------------------------
# %%
scores["deltacor_prod"] = np.prod(np.abs(scores[["deltacor1", "deltacor2"]]), 1)
scores["deltacor_prod"]

# %%
import scipy.stats

lm = scipy.stats.linregress(scores["deltacor_prod"], scores["deltacor_interaction"])

scores["deltacor_interaction_corrected"] = (
    scores["deltacor_interaction"] - lm.intercept - lm.slope * scores["deltacor_prod"]
)
scores["deltacor_interaction_corrected_ratio"] = (
    scores["deltacor_interaction"] - lm.intercept - lm.slope * scores["deltacor_prod"]
) / scores["deltacor_prod"]

# %%
scores["synergistic"] = scores["deltacor_interaction_corrected"] < 0

# %%
fig, ax = plt.subplots()
symbols_oi = ["CD74", "CCL4", "EBF1"]
symbols_oi = ["KLF12"]
genes_oi = transcriptome.gene_id(symbols_oi)
# genes_oi = scores["gene"].unique()[:100]
scores_oi = scores.query("gene in @genes_oi").copy()

cmap, norm = mpl.cm.get_cmap("RdBu_r"), mpl.colors.Normalize()
plt.scatter(
    scores_oi["deltacor_prod"],
    np.abs(scores_oi["deltacor_interaction"]),
    c=cmap(norm(scores_oi["deltacor_interaction_corrected"])),
    # c=scores_oi["gene"].astype("category").cat.codes,
)
plt.colorbar()
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_ylim(1e-7, 0.002)
# ax.set_xlim(1e-7, 0.00018)

# %%
sns.histplot(
    scores.query("synergistic & (distance > 1000)")[
        ["window_mid1", "window_mid2"]
    ].values.flatten(),
    bins=10,
    stat="density",
    binrange=window,
)
sns.histplot(
    scores.query("~synergistic & (distance > 1000)")[
        ["window_mid1", "window_mid2"]
    ].values.flatten(),
    bins=10,
    stat="density",
    binrange=window,
)

# %%
sns.histplot(
    scores.sort_values("deltacor_interaction_corrected")
    .groupby("gene")
    .first()[["window_mid1", "window_mid2"]]
    .values.flatten(),
    bins=10,
)
sns.histplot(
    scores.sort_values("deltacor_interaction_corrected")
    .groupby("gene")
    .last()[["window_mid1", "window_mid2"]]
    .values.flatten(),
    bins=10,
)

# %%
sns.histplot(
    scores.query("synergistic")[["window_mid1", "window_mid2"]].values.flatten(),
    bins=10,
    stat="density",
    binrange=window,
)
sns.histplot(
    scores.query("~synergistic")[["window_mid1", "window_mid2"]].values.flatten(),
    bins=10,
    stat="density",
    binrange=window,
)

# %%
plt.scatter(
    scores_oi["deltacor_prod"],
    scores_oi["deltacor_interaction"],
    c=scores_oi["deltacor_interaction_corrected"],
)
plt.colorbar()

# %%
sns.histplot(
    scores_oi.query("(deltacor_interaction_corrected < 0) & (deltacor_prod > 0.001)")[
        ["window_mid1", "window_mid2"]
    ].values.flatten(),
    bins=20,
    stat="density",
)

# %%
# deltacor_prod_cutoff = 0.0001
# cutoff = 0.00001
cutoff = 1e-6

bins = np.linspace(0, (window[1] - window[0]), 50)
baseline = scores.query("(deltacor_prod > @cutoff)")["synergistic"].mean()
synergism_distance = pd.DataFrame(
    {
        "all": np.bincount(
            np.digitize(
                scores.query("(deltacor_prod > @cutoff)")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic": np.bincount(
            np.digitize(
                scores.query("synergistic & (deltacor_prod > @cutoff)")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic": np.bincount(
            np.digitize(
                scores.query("~synergistic & (deltacor_prod > @cutoff)")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)

synergism_distance["synergistic_relative"] = (
    synergism_distance["synergistic"] / synergism_distance["synergistic"].sum()
)
synergism_distance["antagonistic_relative"] = (
    synergism_distance["antagonistic"] / synergism_distance["antagonistic"].sum()
)

synergism_distance["synergism_relative_ratio"] = (
    synergism_distance["synergistic_relative"]
    / synergism_distance["antagonistic_relative"]
)
synergism_distance["synergism_distant_ratio"] = (
    synergism_distance["synergistic"] / (synergism_distance["all"]) / baseline
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_distance["bin"][:-1],
    synergism_distance["synergism_distant_ratio"][:-1],
    color="black",
)
ax.set_yscale("log")
ax.set_ylabel("Ratio synergistic\nto antagonistic co-predictivity")
ax.set_xlabel("Distance between windows")
ax.xaxis.set_major_formatter(chd.plotting.distance_ticker)

manuscript.save_figure(fig, "5", "synergism_distance")


# %%
bins = np.linspace(window[0], window[1], 25)

baseline = scores.query("(deltacor_prod > @cutoff) & (distance > 1000)")[
    "synergistic"
].mean()
synergism_position = pd.DataFrame(
    {
        "synergistic": np.bincount(
            np.digitize(
                scores.query("synergistic & (deltacor_prod > @cutoff)")[
                    ["window_mid1", "window_mid2"]
                ].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic": np.bincount(
            np.digitize(
                scores.query("~synergistic & (deltacor_prod > @cutoff)")[
                    ["window_mid1", "window_mid2"]
                ].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic_distant": np.bincount(
            np.digitize(
                scores.query(
                    "synergistic & (deltacor_prod > @cutoff) & (distance > 1000)"
                )[["window_mid1", "window_mid2"]].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "antagonistic_distant": np.bincount(
            np.digitize(
                scores.query(
                    "~synergistic & (deltacor_prod > @cutoff) & (distance > 1000)"
                )[["window_mid1", "window_mid2"]].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)

synergism_position["synergistic_relative"] = (
    synergism_position["synergistic"] / synergism_position["synergistic"].sum()
)
synergism_position["antagonistic_relative"] = (
    synergism_position["antagonistic"] / synergism_position["antagonistic"].sum()
)

synergism_position["synergism_relative_ratio"] = (
    synergism_position["synergistic_relative"]
    / synergism_position["antagonistic_relative"]
)
synergism_position["synergism_ratio"] = (
    synergism_position["synergistic"] / synergism_position["antagonistic"]
)
synergism_position["synergism_distant_ratio"] = (
    synergism_position["synergistic_distant"]
    / (
        synergism_position["antagonistic_distant"]
        + synergism_position["synergistic_distant"]
    )
    / baseline
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
# ax.plot(
#     synergism_position["bin"][:-1],
#     synergism_position["total"][:-1],
#     color="black",
# )
ax.plot(
    synergism_position["bin"][:-1],
    synergism_position["synergism_distant_ratio"][:-1],
    color="black",
)
# ax.plot(
#     synergism_position["bin"][:-1],
#     synergism_position["synergistic"][:-1] / synergism_position["total"][:-1],
#     color="orange",
# )

ax.set_yscale("log")
ax.set_ylabel("Ratio synergistic\nto antagonistic co-predictivity")
ax.xaxis.set_major_formatter(chd.plotting.gene_ticker)
ax.set_xlabel("Distance to TSS")
ax_position = ax
# ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

manuscript.save_figure(fig, "5", "synergism_position")

# %%
sns.histplot(
    np.concatenate(
        [
            scores.query("deltacor1 < -0.001")["window_mid1"].values,
            scores.query("deltacor2 < -0.001")["window_mid2"],
        ]
    ),
    bins=10,
)
# %%
