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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

from IPython import get_ipython
import chromatinhd as chd

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

sns.set_style("ticks")

# %%
device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, promoter_window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"
outcome_source = "counts"

# splitter = "permutations_5fold5repeat"
# promoter_name, promoter_window = "10k10k", np.array([-10000, 10000])
# outcome_source = "magic"
# prediction_name = "v20"
# prediction_name = "v21"

splitter = "permutations_5fold5repeat"
promoter_name, promoter_window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"
outcome_source = "magic"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = promoter_window[1] - promoter_window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %%
genes_oi = transcriptome.var.index
# genes_oi = transcriptome.gene_id(["CD74"])


# %% [markdown]
# ## Window + size

# %%
windowsize_scores = []
window_scores = []
for gene in tqdm.tqdm(genes_oi):
    try:
        scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
        windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

        scores_folder = prediction.path / "scoring" / "window_gene" / gene
        window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

    except FileNotFoundError:
        continue

    score = (
        windowsize_scoring.genescores.mean("model")
        .sel(phase=["validation", "test"], gene=gene)
        .mean("phase")
        .to_pandas()
    ).reset_index()
    score.loc[:, ["window", "size"]] = windowsize_scoring.design[
        ["window", "size"]
    ].values
    score["deltacor_gene"] = (
        window_scoring.genescores["deltacor"]
        .sel(phase=["validation", "test"])
        .mean("model")
        .mean("phase")
        .sel(gene=gene)
        .to_pandas()[score["window"]]
    ).values
    # score = score.loc[score["deltacor_gene"] < -0.01]
    windowsize_scores.append(score)

    score = (
        window_scoring.genescores.mean("model")
        .sel(phase=["validation", "test"])
        .mean("gene")
        .mean("phase")
        .to_pandas()
    )
    score["gene"] = gene
    window_scores.append(score)
windowsize_scores = pd.concat(windowsize_scores)
window_scores = pd.concat(window_scores)

# %%
windowsize_scores["gene"] = pd.Categorical(windowsize_scores["gene"], categories = transcriptome.var.index)

# %%
windowsize_scores.to_pickle(prediction.path / "scoring" / "windowsize_gene" / "scores.pkl")
window_scores.to_pickle(prediction.path / "scoring" / "windowsize_gene" / "window_scores.pkl")

# %%
x = windowsize_scores.set_index(["gene", "window", "size"])["deltacor"].unstack()
# scores["reldeltacor"] = scores["deltacor"] / (scores["lost"] + 1)
# x = scores.set_index(["gene", "window", "size"])["reldeltacor"].unstack()

# %%
cors = []
for gene, x_ in x.groupby("gene"):
    cor = np.corrcoef(x_.T)
    cors.append(cor)
cors = np.stack(cors)
cor = pd.DataFrame(np.nanmean(cors, 0), index=x.columns, columns=x.columns)

# %%
plt.scatter(x.iloc[:, 0], x.iloc[:, 5])

# %%
design_windows = windowsize_scoring.design.groupby("window").first()
design_size = windowsize_scoring.design.groupby("size").first()
design_size["label"] = [
    "footprint",
    "submono",
    "mono",
    "supermono",
    "di",
    "superdi",
    "tri",
    "supertri",
    "multi",
]

# %%
fig, ax = plt.subplots(figsize=(2, 2))

ax.matshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cor)))
ax.set_yticks(range(len(cor)))
ax.set_xticklabels(design_size.loc[cor.columns, "label"], rotation=90)
ax.set_yticklabels(design_size.loc[cor.columns, "label"])

# %%
scores = pd.read_pickle(prediction.path / "scoring" / "windowsize_gene" / "scores.pkl")

# %% [markdown]
# ## Position fragment dependency

# %%
lost_by_window = scores.set_index(["gene", "window", "size"])["lost"].unstack()
deltacor_by_window = scores.set_index(["gene", "window", "size"])["deltacor"].unstack()


# %%
windowsize_scores = pd.DataFrame(
    {
        "lost_30": lost_by_window[30],
        "lost_100": lost_by_window[100],
        "deltacor_30": deltacor_by_window[30],
        "deltacor_100": deltacor_by_window[100],
    }
)
windowsize_scores["lost_ratio"] = (windowsize_scores["lost_100"] + 0.1) / (
    windowsize_scores["lost_30"] + 0.1
)
windowsize_scores["log_lost_ratio"] = np.log(windowsize_scores["lost_ratio"])
windowsize_scores["deltacor_ratio"] = (windowsize_scores["deltacor_30"] - 0.0001) / (
    windowsize_scores["deltacor_100"] - 0.0001
)

# %%
bins = np.linspace(*promoter_window, 200)
windowsize_scores["bin"] = pd.Categorical(
    bins[
        np.digitize(
            windowsize_scores.index.get_level_values("window"),
            bins,
        )
    ],
    categories=bins,
)

# %%
plotdata = windowsize_scores.groupby("bin").mean()

fig, ax = plt.subplots(figsize=(10, 2))
ax.scatter(
    windowsize_scores.index.get_level_values("window"),
    windowsize_scores["log_lost_ratio"],
    s = 1,
    alpha = 0.1,
)
ax.plot(
    plotdata.index,
    plotdata["log_lost_ratio"],
    color = "orange"
)
fig

# %%
plotdata = windowsize_scores.groupby("bin").median()

fig, ax = plt.subplots(figsize=(2, 2))
ax.scatter(
    windowsize_scores.index.get_level_values("window"),
    windowsize_scores["log_lost_ratio"],
    s = 1,
    alpha = 0.1,
)
ax.plot(
    plotdata.index,
    plotdata["log_lost_ratio"],
    color = "orange"
)
ax.set_xlim(-10000, 10000)
fig

# %% [markdown]
# ## Different motifs in footprint vs submono

# %%
lost_by_window = scores.set_index(["gene", "window", "size"])["lost"].unstack()
deltacor_by_window = scores.set_index(["gene", "window", "size"])["deltacor"].unstack()


# %%
windowsize_scores = pd.DataFrame(
    {
        "lost_30": lost_by_window[30],
        "lost_100": lost_by_window[100],
        "deltacor_30": deltacor_by_window[30],
        "deltacor_100": deltacor_by_window[100],
    }
)
windowsize_scores["lost_ratio"] = (windowsize_scores["lost_30"]) / (
    windowsize_scores["lost_100"]
)
windowsize_scores["deltacor_ratio"] = (windowsize_scores["deltacor_30"] - 0.0001) / (
    windowsize_scores["deltacor_100"] - 0.0001
)

# %%
np.corrcoef(
    np.log(
        windowsize_scores.query("lost_30>0").query("lost_100>0")["lost_ratio"].abs()
    ),
    np.log(
        windowsize_scores.query("lost_30>0").query("lost_100>0")["deltacor_ratio"].abs()
    ),
)

# %%
windowsize_scores["chr"] = promoters.loc[
    windowsize_scores.index.get_level_values("gene")
]["chr"].values
windowsize_scores["strand"] = promoters.loc[
    windowsize_scores.index.get_level_values("gene")
]["strand"].values
windowsize_scores["start"] = window_scoring.design.loc[
    windowsize_scores.index.get_level_values("window")
]["window_start"].values
windowsize_scores["end"] = window_scoring.design.loc[
    windowsize_scores.index.get_level_values("window")
]["window_end"].values
windowsize_scores["tss"] = promoters.loc[
    windowsize_scores.index.get_level_values("gene")
]["tss"].values
windowsize_scores["gstart"] = (
    windowsize_scores["tss"]
    + (windowsize_scores["start"] * (windowsize_scores["strand"] == 1))
    - (windowsize_scores["end"] * (windowsize_scores["strand"] == -1))
).values
windowsize_scores["gend"] = (
    (windowsize_scores["tss"])
    + (windowsize_scores["end"] * (windowsize_scores["strand"] == 1))
    - (windowsize_scores["start"] * (windowsize_scores["strand"] == -1))
).values

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / "cutoff_0001"
)
motifscan = chd.data.Motifscan(motifscan_folder)

# %%
motifscan_scores = []
for gene, x in tqdm.tqdm(
    windowsize_scores.groupby("gene"), total=len(windowsize_scores.index.levels[0])
):
    gene_ix = transcriptome.var.index.get_loc(gene)

# %%
!wget https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_non-redundant_pfms_transfac.txt
# %%
import gimmemotifs

motifs = gimmemotifs.motif.read_motifs(
    "JASPAR2022_CORE_non-redundant_pfms_transfac.txt", fmt = "transfac"
)

# %%


# %% [markdown]
# ## Focus on summits ("Foci"), compare fragment bins to predictivity

# %%

genes_oi = transcriptome.gene_id(["CD74"])
genes_oi = transcriptome.gene_id(["CCL4"])
genes_oi = transcriptome.gene_id(["TCF4"])
genes_oi = transcriptome.var.index

# %%
deltacor_stacked = []
lost_stacked = []
distance_cutoff = 5000
pad = distance_cutoff // 100 - 5


def pad_leftright(x, pad, ix, left_ix, right_ix):
    x = np.pad(
        x,
        ((pad - (ix - left_ix), (pad + 1) - (right_ix - ix)), (0, 0)),
        mode="constant",
        constant_values=np.nan,
    )
    return x


scores = []
foci = []
for gene in tqdm.tqdm(genes_oi):
    try:
        scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
        windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

        scores_folder = prediction.path / "scoring" / "window_gene" / gene
        window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

    except FileNotFoundError:
        continue

    score = (
        windowsize_scoring.genescores.mean("model")
        .sel(phase=["validation", "test"], gene=gene)
        .mean("phase")
        .to_pandas()
    ).reset_index()
    score.loc[:, ["window", "size"]] = windowsize_scoring.design[
        ["window", "size"]
    ].values

    deltacor_by_window = (
        score.set_index(["window", "size"])["deltacor"]
        .unstack()
        .reindex(index=window_scoring.design.index)
    )
    lost_by_window = (
        score.set_index(["window", "size"])["lost"]
        .unstack()
        .reindex(index=window_scoring.design.index)
    )

    # determine foci
    windowscores = (
        window_scoring.genescores["deltacor"]
        .sel(phase=["validation", "test"])
        .mean("model")
        .mean("phase")
        .sel(gene=gene)
        .to_pandas()
    )
    # windowscores = deltacor_by_window.sum(1)

    foci_gene = []
    for window, score in windowscores.sort_values().items():
        if score > -1e-4:
            break
        add = True
        for focus in foci_gene:
            if abs(window - focus["window"]) < distance_cutoff:
                add = False
                break
        if add:
            focus = {"window": window, "deltacor": score}
            foci_gene.append(focus)
    foci_gene = pd.DataFrame(foci_gene)
    foci.append(foci_gene.assign(gene=gene))

    # stack
    for focus in foci_gene.itertuples():
        ix = window_scoring.design.index.tolist().index(focus.window)
        left_ix = max(0, ix - pad)
        right_ix = min(deltacor_by_window.shape[0], ix + pad + 1)
        val = pad_leftright(
            deltacor_by_window.iloc[left_ix:right_ix], pad, ix, left_ix, right_ix
        )
        deltacor_stacked.append(val)
        val = pad_leftright(
            lost_by_window.iloc[left_ix:right_ix], pad, ix, left_ix, right_ix
        )
        lost_stacked.append(val)
foci = pd.concat(foci)


# %%
deltacor_stacked = np.stack(deltacor_stacked)
lost_stacked = np.stack(lost_stacked)

# %%
mean = np.nanmean(deltacor_stacked, 0)
# mean = mean / np.std(mean, 0)
sns.heatmap(mean)

# %%
mean = np.nanmean(lost_stacked, 0)
# mean = mean / np.std(mean, 0)
sns.heatmap(mean)

# %%
mean = np.nanmean(deltacor_stacked, 0) / np.nanmean(lost_stacked, 0)
# mean = mean / np.std(mean, 0)
sns.heatmap(mean)

# %%
mean = np.nanmean(deltacor_stacked, 0)
mean = mean / np.std(mean, 0)
sns.heatmap(mean)

# %%
plt.plot(mean.mean(1), marker="o")

# %%
mean = np.nanmean(lost_stacked, 0)
mean = mean / np.std(mean, 0)
sns.heatmap(mean)

# %% [markdown]
# ## Are those containing footprints less predictive than those with submono?

# %%
design_size["ix"] = np.arange(len(design_size))

# %%
import itertools

sizepair_scores = pd.DataFrame(
    itertools.combinations(design_size.index, 2), columns=["size1", "size2"]
).set_index(["size1", "size2"])

# %%
for size1, size2 in tqdm.tqdm(sizepair_scores.index):
    size1_ix = design_size.loc[size1, "ix"]
    size2_ix = design_size.loc[size2, "ix"]

    foci["n_1"] = np.nansum(lost_stacked, 1)[:, size1_ix]
    foci["n_2"] = np.nansum(lost_stacked, 1)[:, size2_ix]
    foci["2_over_1"] = foci["n_2"] / foci["n_1"]
    foci["log_2_over_1"] = np.log(foci["n_2"] / foci["n_1"])

    foci_oi = foci.loc[
        ~pd.isnull(foci["log_2_over_1"]) & ~np.isinf(foci["log_2_over_1"])
    ]

    genescores = []
    contingencies = []
    for gene, foci_gene in foci.groupby("gene"):
        foci_gene = foci_gene.dropna()
        foci_gene = foci_gene.loc[~np.isinf(foci_gene["log_2_over_1"])]
        cor = np.corrcoef(foci_gene["log_2_over_1"], foci_gene["deltacor"])[0, 1]
        a = foci_gene["log_2_over_1"] > foci_gene["log_2_over_1"].mean()
        b = foci_gene["deltacor"] > foci_gene["deltacor"].mean()
        contingency = pd.crosstab(a, b).reindex(
            index=[True, False], columns=[True, False], fill_value=0
        )
        contingencies.append(contingency)
        genescores.append(
            {
                "gene": gene,
                "cor": cor,
                "n": len(foci_gene),
                "contingency": contingency,
            }
        )
    genescores = pd.DataFrame(genescores)

    cor = np.corrcoef(foci_oi["log_2_over_1"], foci_oi["deltacor"])
    sizepair_scores.loc[(size1, size2), "cor"] = cor[0, 1]

    contingencies = np.stack(contingencies)
    contingency = pd.DataFrame(contingencies.mean(0), index=["submono", "footprint"])
    odds = (
        contingency.iloc[0, 0]
        * contingency.iloc[1, 1]
        / (contingency.iloc[0, 1] * contingency.iloc[1, 0])
    )

    sizepair_scores.loc[(size1, size2), "odds"] = odds

# %%
sizepair_scores["logodds"] = np.log(sizepair_scores["odds"])

# %%
def unstack_symmetric(series, symmetrizer="add"):
    df = series.unstack().fillna(0)
    cols = np.unique([*df.columns, *df.index])
    df = df.reindex(index=cols, columns=cols, fill_value=0)
    if symmetrizer == "add":
        df = df + df.T
    elif symmetrizer == "subtract":
        df = df - df.T
    else:
        raise ValueError
    df.values[np.diag_indices_from(df)] = np.diag(df) / 2
    return df


norm = mpl.colors.LogNorm(vmin=0.25, vmax=4)
cmap = mpl.cm.get_cmap("RdBu_r")

fig, ax = plt.subplots(figsize=(2, 2))
ax.matshow(
    np.exp(unstack_symmetric(sizepair_scores["logodds"], symmetrizer="subtract")),
    cmap=cmap,
    norm=norm,
)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
# sns.heatmap(unstack_symmetric(sizepair_scores["cor"]))
