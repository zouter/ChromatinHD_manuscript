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

import scanpy as sc

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
# ## Select gene

# %%
# symbol = "LYN"
# symbol = "TNFRSF13C"
# symbol = "PAX5"
# symbol = "IRF8"
# symbol = "BCL2"
symbol = "BACE2"
symbol = "CD79A"
symbol = "MAP4K2"
# symbol = "CD74"
# symbol = "CXCR5"
# symbol = "SPIB"
# symbol = "BCL2"
symbol = "CCL4"
gene = transcriptome.var.query("symbol == @symbol").index[0]
print(symbol)

# %%
sc.pl.umap(transcriptome.adata, color=["celltype", gene])

# %%
promoter = promoters.loc[gene]
promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"
promoter_str

# %%
import cooler

cool_name = "rao_2014_1kb"
step = 1000

# cool_name = "gu_2021_500bp"
# step = 500

# cool_name = "matrix_1kb"
# step = 1000

if cool_name == "rao_2014_1kb":
    c = cooler.Cooler(
        str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000"
    )
elif cool_name == "gu_2021_500bp":

hic, bins_hic = chdm.hic.extract_hic(promoter, c=c)
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


hic = clean_hic(hic, bins_hic)

# %%
hic = chdm.hic.maxipool_hic(hic, bins_hic, k=5)

# %%
fig = chd.grid.Figure(chd.grid.Wrap(padding_width=0))

for k in range(0, 12, 2):
    panel, ax = fig.main.add(chd.grid.Panel((1.5, 1.5)))
    if k == 0:
        hic2 = hic
    else:
        hic2 = chdm.hic.meanpool_hic(hic, bins_hic, k=k)
    ax.imshow(
        hic2["balanced"].unstack(),
    )
    ax.axis("off")
    ax.set_title(f"{k}kb")
for k in range(0, 12, 2):
    panel, ax = fig.main.add(chd.grid.Panel((1.5, 1.5)))
    if k == 0:
        hic2 = hic
    else:
        hic2 = chdm.hic.maxipool_hic(hic, bins_hic, k=k)
    ax.imshow(
        hic2["balanced"].unstack(),
    )
    ax.axis("off")
    ax.set_title(f"{k}kb")
fig.plot()
fig.suptitle("Example of pooling Hi-C")

# %%
scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
interaction_file = scores_folder / "interaction.pkl"

if interaction_file.exists():
    scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
else:
    raise ValueError("No scores found")
assert len(scores_oi) > 0

# %%
scores_oi["hicwindow1"] = chdm.hic.match_windows(scores_oi["window1"].values, bins_hic)
scores_oi["hicwindow2"] = chdm.hic.match_windows(scores_oi["window2"].values, bins_hic)

# %%
# only consider positive correlations
scores_oi["cor"] = np.clip(scores_oi["cor"], 0, np.inf)

# match windows to Hi-C
matching = chdm.hic.create_matching(bins_hic, scores_oi, hic)

# %%
fig, ax = plt.subplots()
distance_cutoff = 1000
ax.scatter(
    matching.query("distance > @distance_cutoff")["balanced"],
    matching.query("distance > @distance_cutoff")["cor"].abs(),
    alpha=0.1,
)
ax.set_xlabel("Hi-C")
ax.set_ylabel("ChromatinHD")

# %%
def compare_contingency(a, b, return_odds=True):
    contingency = pd.crosstab(
        pd.Categorical(a > np.median(a), [False, True]),
        pd.Categorical(b > np.median(b), [False, True]),
        dropna=False,
    )
    result = {}
    if return_odds:
        result["odds"] = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / (
            contingency.iloc[1, 0] * contingency.iloc[0, 1]
        )
    result["contingency"] = contingency.values
    return result


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

distance_scores = []
for distance1, distance2 in tqdm.tqdm(distslices[["distance1", "distance2"]].values):
    distance_scores.append(
        {
            **compare_contingency(
                *(
                    matching.query("distance > @distance1")
                    .query("distance <= @distance2")[["balanced", "cor"]]
                    .values.T
                )
            ),
            "distance1": distance1,
            "distance2": distance2,
        }
    )
distance_scores = pd.DataFrame(distance_scores)
distance_scores["logodds"] = np.log(distance_scores["odds"])
distance_scores

# %%
np.exp(distance_scores["logodds"].dropna().mean())

# %%
contingency = np.stack(distance_scores["contingency"].values).sum(0)
(contingency[0, 0] * contingency[1, 1]) / (contingency[0, 1] * contingency[1, 0])

# %%
matching["dist"] = matching.index.get_level_values(
    "window2"
) - matching.index.get_level_values("window1")
matching["windowmid"] = (
    matching.index.get_level_values("window1")
    + matching.index.get_level_values("window2")
) / 2

# %%
# plot slices
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.1))
for distance1, distance2 in tqdm.tqdm(distslices[["distance1", "distance2"]].values):
    matching_oi = matching.query("dist > @distance1").query("dist <= @distance2")
    resolution = 0.025
    width = len(matching_oi["windowmid"].unique()) * resolution
    height = len(matching_oi["dist"].unique()) * resolution
    panel = fig.main.add_under(chd.grid.Panel((width, height)), padding=0)
    chd.plotting.matshow45(
        panel.ax,
        matching_oi["cor"],
        radius=1000 / 2,
        cmap=mpl.cm.get_cmap("RdBu_r"),
        norm=mpl.colors.CenteredNorm(0, halfrange=matching_oi["cor"].abs().max()),
    )

    panel.ax.set_xticks([])
    for spine in panel.ax.spines.values():
        spine.set_visible(False)
    panel.ax.set_yticks([panel.ax.get_ylim()[0]])
    panel.ax.set_yticklabels([f"{distance1/1000:.0f}kb-{distance2/1000:.0f}kb"])

    panel = fig.main.add_under(chd.grid.Panel((width, height)))
    chd.plotting.matshow45(
        panel.ax,
        matching_oi["balanced"],
        radius=1000 / 2,
        cmap=mpl.cm.get_cmap("rocket"),
    )
    panel.ax.set_xticks([])
    panel.ax.set_yticks([])
    for spine in panel.ax.spines.values():
        spine.set_visible(False)
fig.plot()

# %%
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
                    *(matching_oi[["cor", "balanced"]].values.T),
                    return_odds=False,
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
    print(odds)
pooling_scores = pd.DataFrame(pooling_scores)

# %%
fig, ax = plt.subplots()
pooling_scores.set_index("k")["odds"].plot(ax=ax)
ax.set_ylabel("odds")
ax2 = ax.twinx()
ax2.tick_params(labelcolor="red")
ax2.set_ylabel("cor", color="red")
pooling_scores.set_index("k")["cor"].plot(ax=ax2, color="red")

# %%
pooling_scores["cor"].idxmax()


# %%
distance_scores["logodds"] = np.log(distance_scores["odds"])
plt.plot(distance_scores["distance1"], distance_scores["logodds"])

# %%
main = chd.grid.Grid(padding_width=0)
fig = chd.grid.Figure(main)

panel_dim = (6, 6)

panel_hic = main[0, 0] = chd.grid.Panel(panel_dim)
ax = panel_hic.ax

bins_hic["ix"] = np.arange(len(bins_hic))

k = 1
min_distance = k * 1000
plotdata = matching.query("distance > @min_distance").copy().reset_index()
plotdata["ix1"] = bins_hic.loc[plotdata["window1"]]["ix"].values
plotdata["ix2"] = bins_hic.loc[plotdata["window2"]]["ix"].values
plotdata_chd = plotdata.loc[plotdata["ix1"] <= plotdata["ix2"]]
plotdata_hic = plotdata.loc[plotdata["ix1"] >= plotdata["ix2"]]

cmap_chd = mpl.cm.get_cmap("RdBu_r")
norm_chd = mpl.colors.TwoSlopeNorm(
    0, vmin=-plotdata["cor"].abs().max(), vmax=plotdata["cor"].abs().max()
)

cmap_hic = mpl.cm.get_cmap("viridis")
norm_hic = mpl.colors.Normalize(vmin=0, vmax=np.log1p(plotdata["balanced"].max()))

ax.matshow(
    np.log1p(plotdata_hic.groupby(["window1", "window2"])["balanced"].max().unstack()),
    cmap=cmap_hic,
    norm=norm_hic,
)
ax.set_title("Hi-C (GM12878)")

# panel_chd = main[0, 1] = chd.grid.Panel(panel_dim)
# ax = panel_chd.ax
ax.matshow(
    plotdata_chd.groupby(["window1", "window2"])["cor"].mean().unstack(),
    cmap=cmap_chd,
    norm=norm_chd,
)
ax.invert_yaxis()
ax.set_title("ChromatinHD co-predictivity (PBMCs)")
ax.set_yticks([])

fig.plot()

# %%
fig, ax = plt.subplots(figsize=panel_dim)
bins_hic["ix"] = np.arange(len(bins_hic))

ax.set_xlim(0, len(bins_hic))
ax.set_ylim(0, len(bins_hic))

import itertools

k = 1
min_distance = k * 1000

plotdata = matching.query("distance > @min_distance").copy().reset_index()
plotdata["ix1"] = bins_hic.loc[plotdata["window1"]]["ix"].values
plotdata["ix2"] = bins_hic.loc[plotdata["window2"]]["ix"].values
plotdata = plotdata.loc[plotdata["ix1"] <= plotdata["ix2"]]

plotdata["dot"] = plotdata["cor"] > np.quantile(plotdata["cor"], 0.9)

for (window1, window2), plotdata_row in plotdata.set_index(
    ["window1", "window2"]
).iterrows():
    ix1, ix2 = bins_hic.loc[window1]["ix"], bins_hic.loc[window2]["ix"]

    rect = mpl.patches.Rectangle(
        (ix1, ix2),
        1,
        1,
        linewidth=0,
        facecolor=cmap_hic(norm_hic(plotdata_row["balanced"])),
    )
    ax.add_patch(rect)

    rect = mpl.patches.Rectangle(
        (ix2 - k, ix1 + k),
        1,
        1,
        linewidth=0,
        facecolor=cmap_chd(norm_chd(plotdata_row["cor"])),
    )
    ax.add_patch(rect)

    # if plotdata_row["dot"]:
    #     ax.scatter(
    #         ix1 + 0.5,
    #         ix2 + 0.5,
    #         color="white",
    #         s=2,
    #     )
    #     ax.scatter(
    #         ix2 + 0.5 - k,
    #         ix1 + 0.5 + k,
    #         color="white",
    #         s=2,
    #     )
ax.invert_yaxis()

# %%
import scanpy as sc

sc.pl.umap(transcriptome.adata, color=["celltype", gene])

# %%
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype",
)

# %% [markdown]
# ## Pool

# %%


def maxpool_hic(hic, bins_hic, distance_cutoff=1000, k=1):
    x = pool_prepare_hic(hic, bins_hic, distance_cutoff=distance_cutoff)

    footprint = np.ones((k * 2 + 1, k * 2 + 1))
    footprint[k, k] = 0

    x2 = pd.DataFrame(
        scipy.ndimage.maximum_filter(x, footprint=footprint),
        index=bins_hic.index.copy(),
        columns=bins_hic.index.copy(),
    )
    x2.index.name = "window1"
    x2.columns.name = "window2"
    hic2 = x2.stack().to_frame().rename(columns={0: "balanced"})
    return hic2


def maxipool_hic(hic, bins_hic, distance_cutoff=1000, k=1):
    x = pool_prepare_hic(hic, bins_hic, distance_cutoff=distance_cutoff)

    footprint = np.ones((k * 2 + 1, k * 2 + 1))

    x2 = pd.DataFrame(
        scipy.ndimage.maximum_filter(x, footprint=footprint),
        index=bins_hic.index.copy(),
        columns=bins_hic.index.copy(),
    )
    x2.index.name = "window1"
    x2.columns.name = "window2"
    hic2 = x2.stack().to_frame().rename(columns={0: "balanced"})
    return hic2


# %%
hic_max = maxpool_hic(hic, bins_hic)
hic_max["distance"] = np.abs(
    hic_max.index.get_level_values("window1").astype(float)
    - hic_max.index.get_level_values("window2").astype(float)
)
hic_maxi = maxipool_hic(hic, bins_hic)
hic_maxi["distance"] = np.abs(
    hic_maxi.index.get_level_values("window1").astype(float)
    - hic_maxi.index.get_level_values("window2").astype(float)
)
hic_maxi_5 = maxipool_hic(hic, bins_hic, k=2)
hic_maxi_5["distance"] = np.abs(
    hic_maxi_5.index.get_level_values("window1").astype(float)
    - hic_maxi_5.index.get_level_values("window2").astype(float)
)

fig = chd.grid.Figure(chd.grid.Grid())
panel = fig.main.add_right(chd.grid.Panel((2, 2)))
panel.ax.matshow(hic.query("distance > 1000")["balanced"].unstack())

panel = fig.main.add_right(chd.grid.Panel((2, 2)))
panel.ax.matshow(hic_max.query("distance > 1000")["balanced"].unstack())

panel = fig.main.add_right(chd.grid.Panel((2, 2)))
panel.ax.matshow(hic_maxi.query("distance > 1000")["balanced"].unstack())

panel = fig.main.add_right(chd.grid.Panel((2, 2)))
panel.ax.matshow(hic_maxi_5.query("distance > 1000")["balanced"].unstack())

fig.plot()

# %% [markdown]
# ## For all genes

# %%
genes_all = transcriptome.var.index

scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)
genes_oi = (
    nothing_scoring.genescores.mean("model")
    .sel(phase=["test", "validation"])
    .mean("phase")
    .sel(i=0)
    .to_pandas()
    .query("cor > 0.1")
    .sort_values("cor", ascending=False)
    .index
)

# %%
# load or create gene hics
import pickle
import pathlib

if not pathlib.Path("gene_hics.pkl").exists():
    gene_hics = {}
    c = cooler.Cooler(
        str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000"
    )
    for gene in tqdm.tqdm(genes_all):
        promoter = promoters.loc[gene]
        promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"

        import cooler

        hic, bins_hic = chdm.hic.extract_hic(promoter, c=c)

        gene_hics[gene] = (hic, bins_hic)
    pickle.dump(gene_hics, open("gene_hics.pkl", "wb"))
else:
    gene_hics = pickle.load(open("gene_hics.pkl", "rb"))


# %%
genescores = []

import scipy.stats

for gene in tqdm.tqdm(genes_oi):
    score = {}

    score.update(
        {
            "gene": gene,
            "symbol": transcriptome.symbol(gene),
        }
    )

    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if gene in scores_by_gene:
        scores_oi = scores_by_gene[gene]
    elif interaction_file.exists():
        scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
        scores_by_gene[gene] = scores_oi
    else:
        continue

    promoter = promoters.loc[gene]
    promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"

    hic, bins_hic = gene_hics[gene]
    hic, bins_hic = chdm.hic.fix_hic(hic, bins_hic)

    scores_oi["hicwindow1"] = chdm.hic.match_windows(
        scores_oi["window1"].values, bins_hic
    )
    scores_oi["hicwindow2"] = chdm.hic.match_windows(
        scores_oi["window2"].values, bins_hic
    )

    matching = chdm.hic.create_matching(
        bins_hic,
        scores_oi,
        # scores_oi.query("qval < 0.2"),
        hic,
    )

    matching_oi = matching.query("distance > 1000")
    lm = scipy.stats.linregress(
        matching_oi["balanced"],
        matching_oi["cor"],
    )

    score.update(
        {
            "rvalue": lm.rvalue,
            "slope": lm_residuals.slope,
            "pvalue": lm_residuals.pvalue,
        }
    )

    # maxpool
    hic2 = maxpool_hic(hic, bins_hic)
    matching2 = chdm.hic.create_matching(
        bins_hic,
        scores_oi,
        # scores_oi.query("qval < 0.2"),
        hic2,
    )

    matching2_oi = matching2.query("distance > 1000")
    lm2 = scipy.stats.linregress(
        matching2_oi["balanced"],
        matching2_oi["cor"],
    )

    score.update(
        {
            "rvalue_maxpool": lm2.rvalue,
        }
    )

    # maxpool
    hic2 = maxipool_hic(hic, bins_hic)
    matching2 = chdm.hic.create_matching(
        bins_hic,
        scores_oi,
        # scores_oi.query("qval < 0.2"),
        hic2,
    )

    matching2_oi = matching2.query("distance > 1000")
    lm2 = scipy.stats.linregress(
        matching2_oi["balanced"],
        matching2_oi["cor"],
    )

    score.update(
        {
            "rvalue_maxipool": lm2.rvalue,
        }
    )

    # maxpool
    hic2 = maxipool_hic(hic, bins_hic, k=2)
    matching2 = chdm.hic.create_matching(
        bins_hic,
        scores_oi,
        # scores_oi.query("qval < 0.2"),
        hic2,
    )

    matching2_oi = matching2.query("distance > 1000")
    lm2 = scipy.stats.linregress(
        matching2_oi["balanced"],
        matching2_oi["cor"],
    )

    score.update(
        {
            "rvalue_maxipool_5": lm2.rvalue,
        }
    )

    genescores.append(score)
genescores = pd.DataFrame(genescores).set_index("gene")

# %%
fig, ax = plt.subplots(figsize=(2, 2))
genes_diffexp_b = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 1.0").index
genes_diffexp_b = genes_diffexp_b[genes_diffexp_b.isin(genescores.index)]
genes_diffexp_a = [gene for gene in genescores.index if gene not in genes_diffexp_b]
sns.ecdfplot(data=genescores, x="rvalue", label="actual")
sns.ecdfplot(data=genescores, x="rvalue_maxpool", label="maxpool")
sns.ecdfplot(data=genescores, x="rvalue_maxipool", label="maxipool")
sns.ecdfplot(data=genescores, x="rvalue_maxipool_5", label="maxipool")

plt.legend()

# %%
(
    genescores["rvalue"].mean(),
    genescores["rvalue_maxpool"].mean(),
    genescores["rvalue_maxipool"].mean(),
    genescores["rvalue_maxipool_5"].mean(),
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
genes_diffexp_b = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 1.0").index
genes_diffexp_b = genes_diffexp_b[genes_diffexp_b.isin(genescores.index)]
genes_diffexp_a = [gene for gene in genescores.index if gene not in genes_diffexp_b]
sns.ecdfplot(data=genescores, x="rvalue")
sns.ecdfplot(
    data=genescores.loc[genes_diffexp_a],
    x="rvalue",
)
sns.ecdfplot(
    data=genescores.loc[genes_diffexp_b],
    x="rvalue",
)

# %%
genescores.sort_values("rvalue_maxpool", ascending=False).head(10)

# %%
genescores["rvalue"].mean()

# %%
matching = chdm.hic.create_matching(
    bins_hic,
    scores_oi,
    # scores_oi.query("qval < 0.2"),
    hic,
)

import scipy.stats

matching_oi = matching.query("distance >= 2000")
lm = scipy.stats.linregress(
    matching_oi["balanced"],
    matching_oi["cor"],
)
lm.rvalue
# %%
matching = chdm.hic.create_matching(
    bins_hic,
    scores_oi,
    # scores_oi.query("qval < 0.2"),
    hic2,
)

import scipy.stats

matching_oi = matching.query("distance >= 2000")
lm2 = scipy.stats.linregress(
    matching_oi["balanced"],
    matching_oi["cor"],
)
lm2.rvalue


# %%
genescores = []

import scipy.stats

for gene in tqdm.tqdm(genes_oi):
    score = {}

    score.update(
        {
            "gene": gene,
            "symbol": transcriptome.symbol(gene),
        }
    )

    hic, bins_hic = gene_hics[gene]
    hic, bins_hic = chdm.hic.fix_hic(hic, bins_hic)

    hic["distance"] = np.abs(
        hic.index.get_level_values("window1").astype(float)
        - hic.index.get_level_values("window2").astype(float)
    )


# %%
