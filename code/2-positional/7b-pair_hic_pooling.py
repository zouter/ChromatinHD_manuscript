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
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

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

# %%
import itertools


def clean_hic(hic, bins_hic):
    hic = (
        pd.DataFrame(
            index=pd.MultiIndex.from_frame(
                pd.DataFrame(
                    itertools.product(bins_hic.index, bins_hic.index),
                    columns=["window1", "window2"],
                )
            )
        )
        .join(hic, how="left")
        .fillna({"balanced": 0.0})
    )
    hic["distance"] = np.abs(
        hic.index.get_level_values("window1").astype(float)
        - hic.index.get_level_values("window2").astype(float)
    )
    hic.loc[hic["distance"] <= 1000, "balanced"] = 0.0
    return hic

# %% [markdown]

# ## HiC distance correspondence


# %%
def compare_contingency(a, b):
    contingency = pd.crosstab(
        pd.Categorical(a > np.median(a), [False, True]),
        pd.Categorical(b > np.median(b), [False, True]),
        dropna=False,
    )
    result = {}
    result["contingency"] = contingency.values
    return result


# %%
distwindows = pd.DataFrame(
    {
        "distance": [
            1000,
            *np.arange(10000, 150000, 10000),
            np.inf,
        ],
    }
)
distslices = pd.DataFrame(
    {
        "distance1": distwindows["distance"][:-1].values,
        "distance2": distwindows["distance"][1:].values,
    }
)

# %%
outputs_dir = chd.get_git_root() / "tmp" / "pooling"
outputs_dir.mkdir(exist_ok=True)


# %%
genes_oi = transcriptome.gene_id(
    ["BCL2", "CD74", "CD79A", "CD19", "LYN", "TNFRSF13C", "PAX5", "IRF8", "IRF4"]
)
genes_oi = list(set([*genes_oi, *transcriptome.var.index]))

# %%
for gene in tqdm.tqdm(genes_oi):
    if (outputs_dir / (gene + ".pkl")).exists():
        pooling_scores = pd.read_pickle(outputs_dir / (gene + ".pkl"))
    else:
        # load scores
        scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
        interaction_file = scores_folder / "interaction.pkl"

        if interaction_file.exists():
            scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
        else:
            # print("No scores found")
            continue

        if len(scores_oi) == 0:
            # print("No significant windows")
            continue

        # load hic
        promoter = promoters.loc[gene]
        hic, bins_hic = chdm.hic.extract_hic(promoter)
        hic = clean_hic(hic, bins_hic)

        # match
        scores_oi["hicwindow1"] = chdm.hic.match_windows(
            scores_oi["window1"].values, bins_hic
        )
        scores_oi["hicwindow2"] = chdm.hic.match_windows(
            scores_oi["window2"].values, bins_hic
        )

        scores_oi["cor"] = np.clip(scores_oi["cor"], 0, np.inf)

        import scipy.stats

        pooling_scores = []
        for k in range(0, 40, 1):
            if k == 0:
                hic_oi = hic
            else:
                hic_oi = chdm.hic.maxipool_hic(hic, bins_hic, k=k)
                # hic_oi = chdm.hic.meanpool_hic(hic, bins_hic, k=k)

            matching = chdm.hic.create_matching(bins_hic, scores_oi, hic_oi)

            distance_scores = []
            ranks = []
            for distance1, distance2 in distslices[["distance1", "distance2"]].values:
                matching_oi = matching.query("distance > @distance1").query(
                    "distance <= @distance2"
                )
                distance_scores.append(
                    {
                        **compare_contingency(
                            *(matching_oi[["cor", "balanced"]].values.T)
                        ),
                        "distance1": distance1,
                        "distance2": distance2,
                    }
                )

                ranks.append(
                    scipy.stats.rankdata(
                        matching_oi[["cor", "balanced"]].values, axis=0, method="min"
                    )
                    / matching_oi.shape[0]
                )
            score = {}

            # odds
            distance_scores = pd.DataFrame(distance_scores)

            contingency = np.stack(distance_scores["contingency"].values).sum(0)
            odds = (contingency[0, 0] * contingency[1, 1]) / (
                contingency[0, 1] * contingency[1, 0]
            )

            score.update({"k": k, "odds": odds})

            # slice rank correlation
            cor = np.corrcoef(np.concatenate(ranks, 0).T)[0, 1]
            score.update({"cor": cor})

            pooling_scores.append(score)
        pooling_scores = pd.DataFrame(pooling_scores)
        pooling_scores.to_pickle(outputs_dir / (gene + ".pkl"))

# %%
gene_scores = []
for gene in tqdm.tqdm(genes_oi):
    if (outputs_dir / (gene + ".pkl")).exists():
        pooling_scores = pd.read_pickle(outputs_dir / (gene + ".pkl"))
        gene_scores.append(pooling_scores.assign(gene=gene))
gene_scores = pd.concat(gene_scores)

# %%
x = gene_scores.set_index(["gene", "k"])["cor"].unstack()
x = x / x.max(1).values[:, None]
x = x.iloc[np.argsort(np.argmax(x.values, 1))]
fig, ax = plt.subplots()

ax.matshow(x, aspect="auto", cmap="magma", norm=mpl.colors.Normalize(0, 1))

# %%
gene_scores["logodds"] = np.log(gene_scores["odds"])

# %%
gene_scores.dropna().groupby("k")["cor"].mean().plot()
gene_scores.loc[np.isinf(gene_scores["logodds"]), "logodds"] = 2.0
# %%
np.exp(gene_scores.groupby("k")["logodds"].mean()).plot()

# %%
gene_scores.query("k == 0")["logodds"].hist()

# %%
(
    np.exp(gene_scores.query("k == 0")["logodds"].quantile(0.1)),
    np.exp(gene_scores.query("k == 0")["logodds"].quantile(0.5)),
    np.exp(gene_scores.query("k == 0")["logodds"].quantile(0.9)),
)
# %%
(
    gene_scores.query("k == 5")["cor"].quantile(0.1),
    gene_scores.query("k == 5")["cor"].quantile(0.5),
    gene_scores.query("k == 5")["cor"].quantile(0.9),
)

# %%
import scanpy as sc

sc.tl.rank_genes_groups(transcriptome.adata, groupby="celltype")
diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="naive B")
    .rename(columns={"names": "gene"})
    .assign(symbol=lambda x: transcriptome.var.loc[x["gene"], "symbol"].values)
    .set_index("gene")
)

# %%
genes_diffexp = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.1").index
# %%
gene_scores_diffexp = gene_scores.query("gene in @genes_diffexp")
# %%
# %%
gene_scores.query("k == 0")["logodds"].hist()
gene_scores_diffexp.query("k == 0")["logodds"].hist()
# %%
np.exp(gene_scores.groupby("k")["logodds"].mean()).plot(label="All differential genes")
np.exp(gene_scores_diffexp.groupby("k")["logodds"].mean()).plot(label="B-cell genes")
plt.legend()

# %%
(gene_scores.groupby("k")["cor"].mean()).plot(label="All differential genes")
(gene_scores_diffexp.groupby("k")["cor"].mean()).plot(label="B-cell genes")
plt.legend()

# %%
gene_scores

# %%

print(
    f"odds = {np.exp(gene_scores.query('k == 0')['logodds'].mean()):.2f}, q90 = {np.exp(gene_scores.query('k == 0')['logodds'].quantile(0.9)):.2f}, q10 = {np.exp(gene_scores.query('k == 0')['logodds'].quantile(0.1)):.2f}"
)
# %%
print(
    f"odds = {np.exp(gene_scores_diffexp.query('k == 0')['logodds'].mean()):.2f}, q90 = {np.exp(gene_scores_diffexp.query('k == 0')['logodds'].quantile(0.9)):.2f}, q10 = {np.exp(gene_scores_diffexp.query('k == 0')['logodds'].quantile(0.1)):.2f}"
)

# %%
print(
    f"odds = {np.exp(gene_scores.query('k == 1')['logodds'].mean()):.2f}, q90 = {np.exp(gene_scores.query('k == 1')['logodds'].quantile(0.9)):.2f}, q10 = {np.exp(gene_scores.query('k == 1')['logodds'].quantile(0.1)):.2f}"
)

# %%
print(
    f"odds = {np.exp(gene_scores.query('k == 5')['logodds'].mean()):.2f}, q90 = {np.exp(gene_scores.query('k == 5')['logodds'].quantile(0.9)):.2f}, q10 = {np.exp(gene_scores.query('k == 5')['logodds'].quantile(0.1)):.2f}"
)

# %% [markdown]
# # Check spot


# %%
genes_oi = list(
    set(
        [
            *transcriptome.gene_id(
                [
                    "BCL2",
                    "CD74",
                    "CD79A",
                    "CD19",
                    "LYN",
                    "TNFRSF13C",
                    "PAX5",
                    "IRF8",
                    "IRF4",
                ]
            ),
            *transcriptome.var.index[:5000],
        ]
    )
)

# %%
cool_name = "rao_2014_1kb"

# %%
import pickle

scores_folder = prediction.path / "scoring" / "pairwindow_gene"
matchings_file = scores_folder / f"matching_{cool_name}.pkl"

if matchings_file.exists():
    with open(matchings_file, "rb") as f:
        matchings = pickle.load(f)
elif "matchings" not in globals():
    matchings = {}

# %%
hic_file = folder_data_preproc / "hic" / promoter_name / f"{cool_name}.pkl"
gene_hics = pd.read_pickle(hic_file)

# %%
outputs_dir = chd.get_output()
for gene in tqdm.tqdm(genes_oi):
    if gene in matchings:
        continue
    else:
        # load scores
        scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
        interaction_file = scores_folder / "interaction.pkl"

        if interaction_file.exists():
            scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
        else:
            continue

        if len(scores_oi) == 0:
            continue

        # load hic
        promoter = promoters.loc[gene]
        # hic, bins_hic = chdm.hic.extract_hic(promoter)
        if gene not in gene_hics:
            continue
        hic, bins_hic = gene_hics[gene]
        hic = clean_hic(hic, bins_hic)

        # match
        scores_oi["hicwindow1"] = chdm.hic.match_windows(
            scores_oi["window1"].values, bins_hic
        )
        scores_oi["hicwindow2"] = chdm.hic.match_windows(
            scores_oi["window2"].values, bins_hic
        )

        scores_oi["cor"] = np.clip(scores_oi["cor"], 0, np.inf)

        matching = chdm.hic.create_matching(bins_hic, scores_oi, hic)

        matchings[gene] = (matching, bins_hic)

# %%
import pickle

scores_folder = prediction.path / "scoring" / "pairwindow_gene"
matchings_file = scores_folder / f"matching_{cool_name}.pkl"
pickle.dump(matchings, open(matchings_file, "wb"))

# %%
! ls -lh {matchings_file}

# %%
import pickle

matchings = pickle.load(open(matchings_file, "rb"))

# %%
import random

def random_interval(start: int, end: int, window_start: int, window_end: int):
    # Calculate the distance
    distance = abs(end - start)

    # Make sure the window is large enough to contain an interval of the given distance
    if window_end - window_start < distance:
        raise ValueError("The window is too small for the given distance")

    # Generate a random start point for the new interval within the window
    new_start = random.randrange(window_start, window_end - distance + 1)

    # Calculate the end point of the new interval
    new_end = new_start + distance

    # Return the new interval
    return (new_start, new_end)


# %%
pad = 50
spots = []
randomspots = []
for gene in tqdm.tqdm(genes_oi):
    if gene in matchings:
        (matching, bins_hic) = matchings[gene]
        y = np.pad(
            matching["balanced"].unstack(),
            pad_width=((pad, pad + 1), (pad, pad + 1)),
            mode="constant",
            constant_values=np.nan,
        )

        y[np.tril_indices_from(y)] = np.nan

        bins_hic["ix"] = np.arange(len(bins_hic))
        matching["ix1"] = bins_hic.loc[
            matching.index.get_level_values("window1"), "ix"
        ].values.astype(int)
        matching["ix2"] = bins_hic.loc[
            matching.index.get_level_values("window2"), "ix"
        ].values.astype(int)

        matching_oi = matching.query("distance < 10000")
        # matching_oi = matching.query("distance > 10000")
        # matching_oi = matching.query("(distance > 10000) & (distance < 20000)")
        x = matching_oi.query("cor > 0")

        for row in x.itertuples():
            ix1 = row.ix1
            ix2 = row.ix2

            if ix1 > ix2:
                continue
            spot = y[
                slice(ix1 - pad + pad, ix1 + pad + pad + 1),
                slice(ix2 - pad + pad, ix2 + pad + pad + 1),
            ]
            assert spot.shape == (2 * pad + 1, 2 * pad + 1)
            spots.append(spot)

            randomix1, randomix2 = random_interval(ix1, ix2, 0, len(bins_hic))
            randomspot = y[
                slice(randomix1 - pad + pad, randomix1 + pad + pad + 1),
                slice(randomix2 - pad + pad, randomix2 + pad + pad + 1),
            ]
            assert randomspot.shape == (2 * pad + 1, 2 * pad + 1)
            randomspots.append(randomspot)

spots = np.stack(spots)
randomspots = np.stack(randomspots)

# %%
randomspots[np.isinf(randomspots)] = spots[np.isinf(randomspots)]
spots[np.isinf(spots)] = randomspots[np.isinf(spots)]

# %%
spots_mean = np.nanmean(spots, 0)
randomspots_mean = np.nanmean(randomspots, 0)

# %%
def format_distance(x, _, shift=50, scale=1000):
    return f"{int((x - shift) * scale / 1000):.0f}kb"

def center_bullseye(ax, pad=50, focus=None, center_square = True):
    if focus is None:
        focus = pad
    ax.set_xlim(pad - focus - 0.5, pad + focus + 0.5)
    ax.set_ylim(pad - focus - 0.5, pad + focus + 0.5)
    if center_square:
        rect = mpl.patches.Rectangle(
            (pad - 0.5, pad - 0.5), 1, 1, linewidth=1, edgecolor="#333333", facecolor="none"
        )
        ax.add_patch(rect)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_distance))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_distance))
    ax.invert_yaxis()
    return ax

# %%
fig = chd.grid.Figure(chd.grid.Grid())

vmax = np.nanmax(spots_mean - randomspots_mean)
norm = mpl.colors.Normalize(-vmax, vmax)
cmap = mpl.cm.get_cmap("RdBu_r")

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
ax.matshow(spots_mean - randomspots_mean, cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, center_square = False)

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
ax.matshow(spots_mean - randomspots_mean, cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, focus=20)

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
ax.matshow(spots_mean - randomspots_mean, cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, focus=10)

fig.plot()

# %%
spots_norm = (spots - np.nanmean(spots, (1, 2), keepdims=True)) / np.nanstd(
    spots, (1, 2), keepdims=True
)
randomspots_norm = (
    randomspots - np.nanmean(randomspots, (1, 2), keepdims=True)
) / np.nanstd(randomspots, (1, 2), keepdims=True)

# %%
spots_norm_mean = np.nanmean(spots_norm, 0)
randomspots_norm_mean = np.nanmean(randomspots_norm, 0)
spots_norm_std = np.nanmean(spots_norm, 0)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

vmax = np.nanmax(spots_norm_mean - randomspots_norm_mean)
norm = mpl.colors.Normalize(-vmax, vmax)
cmap = mpl.cm.get_cmap("RdBu_r")

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
ax.matshow(spots_norm_mean - randomspots_norm_mean, cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, center_square = False)

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
ax.matshow(spots_norm_mean - randomspots_norm_mean, cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, focus=20)

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2)))
ax.matshow(spots_norm_mean - randomspots_norm_mean, cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, focus=10)

fig.plot()

# %%
