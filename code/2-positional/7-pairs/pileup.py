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
cool_name = "rao_2014_1kb"
step = 1000

# cool_name = "gu_2021_500bp"
# step = 500

# %%
hic_file = folder_data_preproc / "hic" / promoter_name / f"{cool_name}.pkl"
gene_hics = pd.read_pickle(hic_file)

# %%
hic, bins_hic = gene_hics["ENSG00000159958"]
hic["distance"] = np.abs(
    hic.index.get_level_values("window1") - hic.index.get_level_values("window2")
)

sns.heatmap(np.log1p(hic.query("distance > 500")["balanced"].unstack()))

# %% [markdown]
# # Check spot

# %%
genes_oi = sorted(list(
    set(
        [
            *transcriptome.gene_id(
                [
                    # "BCL2",
                    # "CD74",
                    # "CD79A",
                    # "CD19",
                    # "LYN",
                    "TNFRSF13C",
                    # "PAX5",
                    # "IRF8",
                    # "IRF4",
                ]
            ),
            *transcriptome.var.index[:5000],
            # *transcriptome.var.index[:500],
        ]
    )
))
# %%
sns.heatmap(np.log1p(gene_hics[genes_oi[0]][0]["balanced"].unstack()))


# %%
import pickle

scores_folder = prediction.path / "scoring" / "pairwindow_gene"
matchings_file = scores_folder / f"matching_{cool_name}.pkl"

if matchings_file.exists():
    with open(matchings_file, "rb") as f:
        matchings = pickle.load(f)
else:
    matchings = {}

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
        hic, bins_hic = chdm.hic.clean_hic(hic, bins_hic)

        # match
        scores_oi["hicwindow1"] = chdm.hic.match_windows(
            scores_oi["window1"].values, bins_hic
        )
        scores_oi["hicwindow2"] = chdm.hic.match_windows(
            scores_oi["window2"].values, bins_hic
        )

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

scores_folder = prediction.path / "scoring" / "pairwindow_gene"
matchings_file = scores_folder / f"matching_{cool_name}.pkl"
matchings = pickle.load(open(matchings_file, "rb"))


# %% [markdown]
# ### Stack regions around E-E

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
genes_oi = list(
    transcriptome.gene_id(["CD74"])
)
genes_oi = transcriptome.var.index
# genes_oi = diffexp.index[:100]

# %%
def create_matcher(left, right, cor = "positive", ee = True):
    def match(matching):
        matching_oi = matching.query("ix1 < ix2")

        matching_oi = matching_oi.loc[
            (matching_oi["distance"] > left) &
            (matching_oi["distance"] < right)
        ]

        if ee:
            matching_oi = matching_oi.query("((window1 > 1000) | (window1 < -1000)) & ((window2 > 1000) | (window2 < -1000))")
        if cor == "positive":
            matching_oi = matching_oi.query("cor > 0.")
        elif cor == "negative":
            matching_oi = matching_oi.query("cor < 0.")
        elif cor == "zero":
            matching_oi = matching_oi.query("cor == 0")

        return matching_oi
    return match

# matcher_name = "EE2kb-5kb"
# matcher = create_matcher(2000, 5001, cor = "positive", ee = True)

# matcher_name = "EE5kb-10kb"
# matcher = create_matcher(5000, 10000, cor = "positive", ee = True)

# matcher_name = "EE10kb-15kb"
# matcher = create_matcher(10000, 15000, cor = "positive", ee = True)

# matcher_name = "EE10kb-15kb"
# matcher = create_matcher(15000, 20000, cor = "positive", ee = True)

matcher_name = "EE20kb-25kb"
matcher = create_matcher(20000, 25000, cor = "positive", ee = True)

# matcher_name = "EE45kb-50kb"
# matcher = create_matcher(45000, 50000, cor = "positive", ee = True)

# matcher_name = "EE50kb-55kb"
# matcher = create_matcher(50000, 55000, cor = "positive", ee = True)

# matcher_name = "EEdiffexp"
# matcher = create_matcher(10000, 100000, cor = "positive", ee = True)

# %%
pad = 50
spots = []
randomspots = []
spot_scores = []
for gene in tqdm.tqdm(genes_oi):
    if gene in matchings:
        (matching, bins_hic) = matchings[gene]
        if len(bins_hic) < 201:
            continue
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

        x = matcher(matching)

        spot_scores.append(x.assign(gene = gene))

        for row in x.itertuples():
            ix1 = row.ix1
            ix2 = row.ix2

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
spot_scores = pd.concat(spot_scores)
len(spot_scores)

# %%
randomspots[np.isinf(randomspots)] = spots[np.isinf(randomspots)]
spots[np.isinf(spots)] = randomspots[np.isinf(spots)]
spot_scores["spot_ix"] = np.arange(len(spot_scores))

# %% [markdown]
# ### Combine

# %%
import functools
def format_distance(x, _, shift=50, scale=1000, zero = None):
    if zero is not None:
        if x == shift:
            return zero
    if (x - shift) == 0:
        return "0"
    return f"{int((x - shift) * scale / 1000):+.0f}kb"

def center_bullseye(ax, pad=50, focus=None, center_square = True, center_rect = True, center_line = False, ):
    if focus is None:
        focus = pad
    ax.set_xlim(pad - focus - 0.5, pad + focus + 0.5)
    ax.set_ylim(pad - focus - 0.5, pad + focus + 0.5)
    if center_square:
        rect = mpl.patches.Rectangle(
            (pad - 0.5, pad - 0.5), 1, 1, linewidth=1, edgecolor="#333333", facecolor="none"
        )
        ax.add_patch(rect)
    if center_rect:
        rect = mpl.patches.Rectangle(
            (pad - focus - 0.5, pad - 0.5), pad * 2 + 1, 1, linewidth=1, edgecolor="#333", facecolor="none", linestyle = "--"
        )
        ax.add_patch(rect)
        rect = mpl.patches.Rectangle(
            (pad - 0.5, pad - focus - 0.5), 1, pad * 2 + 1, linewidth=1, edgecolor="#333", facecolor="none", linestyle = "--"
        )
        ax.add_patch(rect)
    if center_line:
        ax.axvline(pad, linewidth=1, color="#333", linestyle = "--")
        ax.axhline(pad, linewidth=1, color="#333", linestyle = "--")
    format_distance_ = functools.partial(format_distance, shift = pad, scale = step)
    # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(symmetric = True))
    ax.set_xticks([pad - focus-0.5, pad, pad + focus])
    ax.set_yticks([pad - focus-0.5, pad, pad + focus])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(functools.partial(format_distance_)))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(functools.partial(format_distance_)))
    ax.invert_yaxis()
    return ax

# %%
spots_mean = np.nanmean(spots, 0)
randomspots_mean = np.nanmean(randomspots, 0)
# %%
spots_lr = np.log(spots_mean / randomspots_mean)
spots_lr[np.isinf(spots_lr)] = 0

# %%
sns.heatmap(spots_lr)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

vmax = np.nanquantile(spots_lr, 1.)
norm = mpl.colors.Normalize(-vmax, vmax)
norm = mpl.colors.LogNorm(np.exp(-vmax), np.exp(vmax))
cmap = mpl.cm.get_cmap("RdBu_r")

panel, ax = fig.main.add_right(chd.grid.Panel((1.4, 1.4)))
ax.matshow(np.exp(spots_lr), cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, focus=5)

if matcher_name == "EE20kb-25kb":
    panel, ax = fig.main.add_right(chd.grid.Panel((1.4, 1.4)))
    ax.matshow(np.exp(spots_lr), cmap=cmap, norm=norm)
    center_bullseye(ax, pad = pad, focus=20)

panel, ax = fig.main.add_right(chd.grid.Panel((1.4, 1.4)))
ax.matshow(np.exp(spots_lr), cmap=cmap, norm=norm)
center_bullseye(ax, pad = pad, center_square = False, center_rect = False, center_line = True, focus = 50)

if matcher_name == "EE20kb-25kb":
    panel, ax = fig.main.add_right(chd.grid.Panel((0.1, 1.4)), padding = 0.1)
    cax = plt.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax = ax, format = "%.2f")
    cax.minorformatter = mpl.ticker.FormatStrFormatter('%.2f')
    cax.set_label("contact frequency ratio \n(co-predictive vs random)", rotation = 90, ha = "center", va = "top")

fig.plot()

manuscript.save_figure(fig, "6", "hic_ee_pileup_" + matcher_name, dpi=300)

# %%
spot_ixs = spot_scores.groupby("gene").first().sort_values("cor", ascending = False)["spot_ix"].iloc[0:20]
# spot_ixs = spot_scores.loc[spot_scores["gene"].isin(transcriptome.gene_id(["TNFRSF13C", "BANK1", "AFF3", "HLA-DRA", "BCL2", "RALGPS2", "MS4A1", "PAX5", "CD79A", "FCRL1", "EBF1", "CD79B", "CD37", "BLK", "SETBP1", "HLA-DMB", "CIITA", "MICAL3"]))].groupby("gene").first().sort_values("cor", ascending = False)["spot_ix"].iloc[0:12]
# spot_ixs = spot_scores.loc[spot_scores["gene"].isin(diffexp.head(100).index)].groupby("gene").first().sort_values("cor", ascending = False)["spot_ix"].iloc[0:12]

# spot_lr = spots[spot_ix]

fig = chd.grid.Figure(chd.grid.Wrap(10, padding_width = 0.15, padding_height = 0.))

for spot_ix in spot_ixs:
    spot_lr = np.log(spots[spot_ix] / randomspots_mean)

    # vmax = np.nanmax(spot_lr)
    vmax = np.log(4)
    norm = mpl.colors.Normalize(-vmax, vmax)
    norm = mpl.colors.LogNorm(np.exp(-vmax), np.exp(vmax))
    cmap = mpl.cm.get_cmap("RdBu_r")

    panel, ax = fig.main.add(chd.grid.Panel((0.8, 0.8)))
    ax.matshow(np.exp(spot_lr), cmap=cmap, norm=norm)
    center_bullseye(ax, pad = pad, focus=5)
    ax.set_xticks([])
    ax.set_yticks([])
    title = (
        transcriptome.symbol(spot_scores.iloc[spot_ix]["gene"]) +
        "\n" +
        str(format_distance(spot_scores.iloc[spot_ix].name[0] / 1000, None)) + " and " +str(format_distance(spot_scores.iloc[spot_ix].name[1] / 1000, None))
    )
    ax.set_title(title, fontsize = 8)

fig.plot()

manuscript.save_figure(fig, "6", "hic_ee_pileup_spots_diffexp", dpi=300)

# %% [markdown]
# ### Diffexp

# %%
transcriptome.adata.obs["celltype"].value_counts()

# %%
import scanpy as sc
transcriptome.adata.obs["oi"] = pd.Categorical(np.array(["noi", "oi"])[transcriptome.adata.obs["celltype"].isin(["naive B", "memory B"]).values.astype(int)])
sc.tl.rank_genes_groups(transcriptome.adata, groupby="oi")
diffexp = (
    sc.get.rank_genes_groups_df(
        transcriptome.adata,
        # group="CD14+ Monocytes",
        # group="naive B",
        # group="memory B",
        group="oi",
        )
    .rename(columns={"names": "gene"})
    .assign(symbol=lambda x: transcriptome.var.loc[x["gene"], "symbol"].values)
    .set_index("gene")
)


# %%
genes_up = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.0").index
genes_down = diffexp.query("pvals_adj < 0.05").query("logfoldchanges < -0.0").index
genes_neutral = diffexp.query("pvals_adj > 0.05").index
genes_down_neutral = np.concatenate([genes_down, genes_neutral])

# %%
print(len(genes_up))
print(len(genes_down))
print(len(genes_neutral))

# %%
spots_ix_up = spot_scores.loc[spot_scores["gene"].isin(genes_up)]["spot_ix"].values
spots_lr_up = np.log(np.nanmean(spots[spots_ix_up], 0) / np.nanmean(randomspots[spots_ix_up], 0))

spots_ix_down = spot_scores.loc[spot_scores["gene"].isin(genes_down)]["spot_ix"].values
spots_lr_down = np.log(np.nanmean(spots[spots_ix_down], 0) / np.nanmean(randomspots[spots_ix_down], 0))

# %%
fig = chd.grid.Figure(chd.grid.Grid())

panel, ax = fig.main.add_right(chd.grid.Panel((1.4, 1.4)))
mappable = ax.matshow(spots_lr_up - spots_lr_down, vmin = -0.05, vmax = 0.05, cmap = "PiYG")
center_bullseye(ax, pad = pad, focus=20)

if matcher_name == "EE20kb-25kb":
    panel, ax = fig.main.add_right(chd.grid.Panel((0.1, 1.4)), padding = 0.1)
    cax = plt.colorbar(mappable, cax = ax, format = "%.2f")
    cax.minorformatter = mpl.ticker.FormatStrFormatter('%.2f')
    cax.set_label("$\Delta$ log contact frequency\n(up vs down genes in B-cells)", rotation = 90, ha = "center", va = "top")
else:
    pass

fig.plot()

manuscript.save_figure(fig, "6", "hic_ee_pileup_difference_" + matcher_name, dpi=300)

# %% [markdown]
# ### Predictivity magnitude

# Look at the difference in contact frequency between high and lowly predictive pairs

# %%
cutoff = 0.1

# %%
spots_ix_up = spot_scores.loc[spot_scores["cor"] >= cutoff]["spot_ix"].values
spots_lr_up = np.log(np.nanmean(spots[spots_ix_up], 0) / np.nanmean(randomspots[spots_ix_up], 0))

spots_ix_down = spot_scores.loc[spot_scores["cor"] < cutoff]["spot_ix"].values
spots_lr_down = np.log(np.nanmean(spots[spots_ix_down], 0) / np.nanmean(randomspots[spots_ix_down], 0))

# %%
fig = chd.grid.Figure(chd.grid.Grid())

panel, ax = fig.main.add_right(chd.grid.Panel((1.4, 1.4)))
mappable = ax.matshow(spots_lr_up - spots_lr_down, vmin = -0.05, vmax = 0.05, cmap = "PiYG")
center_bullseye(ax, pad = pad, focus=20)

if matcher_name == "EE20kb-25kb":
    panel, ax = fig.main.add_right(chd.grid.Panel((0.1, 1.4)), padding = 0.1)
    cax = plt.colorbar(mappable, cax = ax, format = "%.2f")
    cax.minorformatter = mpl.ticker.FormatStrFormatter('%.2f')
    cax.set_label("$\Delta$ log contact frequency\n(up vs down genes in B-cells)", rotation = 90, ha = "center", va = "top")
else:
    pass

fig.plot()

manuscript.save_figure(fig, "6", "hic_ee_pileup_magnitude_" + matcher_name, dpi=300)

  # %% [markdown]
# -----------------------------------------------

# %% [markdown]
# ### Approach two: random matching_oi and distance matched intervals

# %%
# get ys and matchings oi
pad = 50
ys = {}
matchings_oi = {}
matchings_oi_random = []
for gene in tqdm.tqdm(genes_oi):
    if gene in matchings:
        (matching, bins_hic) = matchings[gene]
        if len(bins_hic) < 201:
            continue

        assert len(matching) == len(bins_hic)**2
        y = np.pad(
            matching["balanced"].unstack(),
            pad_width=((pad, pad + 1), (pad, pad + 1)),
            mode="constant",
            constant_values=np.nan,
        )

        y[np.tril_indices_from(y)] = np.nan
        y[np.diag_indices_from(y)] = np.nan
        ys[gene] = y

        bins_hic["ix"] = np.arange(len(bins_hic))
        matching["ix1"] = bins_hic.loc[
            matching.index.get_level_values("window1"), "ix"
        ].values.astype(int)
        matching["ix2"] = bins_hic.loc[
            matching.index.get_level_values("window2"), "ix"
        ].values.astype(int)
        matching_oi = matching

        # matching_oi = matching_oi.query("(distance > 1000) & (distance < 10000)")
        # matching_oi = matching_oi.query("(distance > 10000) & (distance < 13000)")
        # matching_oi = matching_oi.query("(distance > 10000)")
        # matching_oi = matching_oi.query("(distance > 50000) & (distance < 56000)")
        matching_oi = matching_oi.query("(distance > 40000) & (distance < 46000)")
        # matching_oi = matching_oi.query("(distance > 20000) & (distance < 24000)")
        # matching_oi = matching_oi.query("(distance > 50000) & (distance < 100000)")

        # E-P
        matching_oi = matching_oi.query("(window1 < 1000) & (window1 > -1000)")
        # matching_oi = matching_oi.query("(window2 < 1000) & (window2 > -1000)")
        # E-E
        # matching_oi = matching_oi.query("((window1 > 1000) | (window1 < -1000)) & ((window2 > 1000) | (window2 < -1000))")
        
        # matching_oi = matching_oi.query("cor < 0")
        # matching_oi = matching_oi.query("cor > 0")
        # matching_oi = matching_oi.query("cor == 0").sample(100)

        # one direction
        matching_oi = matching_oi.query("ix1 < ix2")

        # only positive
        matchings_oi[gene] = matching_oi.loc[matching_oi["cor"] > 0]
        # matchings_oi[gene] = matching_oi.sample(1)
        if len(matching_oi) > 0:
            matchings_oi_random.append(matching_oi.groupby("distance").sample(10, replace = True).assign(gene = gene))
matchings_oi_random = pd.concat(matchings_oi_random)
random_distance_matched = dict(tuple(matchings_oi_random.groupby("distance")))

# %%
# center around each pair and get spot and randomspot
spots = []
randomspots = []
for gene in tqdm.tqdm(genes_oi):
    if gene in matchings_oi:
        y = ys[gene]
        x = matchings_oi[gene]

        for row in x.itertuples():
            ix1 = row.ix1
            ix2 = row.ix2

            spot = y[
                slice(ix1 - pad + pad, ix1 + pad + pad + 1),
                slice(ix2 - pad + pad, ix2 + pad + pad + 1),
            ]
            assert spot.shape == (2 * pad + 1, 2 * pad + 1)
            spots.append(spot)

            randomi = np.random.choice(len(random_distance_matched[row.distance]))
            randomgene, randomix1, randomix2 = random_distance_matched[row.distance].iloc[randomi][["gene", "ix1", "ix2"]]

            randomy = ys[randomgene]
            randomspot = randomy[
                slice(randomix1 - pad + pad, randomix1 + pad + pad + 1),
                slice(randomix2 - pad + pad, randomix2 + pad + pad + 1),
            ]
            assert randomspot.shape == (2 * pad + 1, 2 * pad + 1)
            randomspots.append(randomspot)

spots = np.stack(spots)
randomspots = np.stack(randomspots)


# %% [markdown]
# ### Approach 3: random cor=0 intervals
