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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import polyptich as pp
pp.setup_ipython()

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

# cool_name = "harris_2023_500bp"
# step = 500


# %%
genes_oi = transcriptome.gene_id(
    ["BCL2", "CD74", "CD79A", "CD19", "LYN", "TNFRSF13C", "PAX5", "IRF8", "IRF4"]
)
# genes_oi = transcriptome.gene_id(
#     ["TNFRSF13C"]
# )
# genes_oi = list(set([*genes_oi, *transcriptome.var.index[:100]]))
genes_oi = sorted(list(set([*genes_oi, *transcriptome.var.index])))

# %%
outputs_dir = chd.get_git_root() / "tmp" / "pooling"
outputs_dir.mkdir(exist_ok=True)

# %% [markdown]
# ## HiC distance correspondence

# %%
hic_file = folder_data_preproc / "hic" / promoter_name / f"{cool_name}.pkl"
gene_hics = pd.read_pickle(hic_file)

# %%
hic, bins_hic = gene_hics["ENSG00000159958"]
hic["distance"] = np.abs(
    hic.index.get_level_values("window1") - hic.index.get_level_values("window2")
)

sns.heatmap(np.log1p(hic.query("distance > 500")["balanced"].unstack()))

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
for gene in tqdm.tqdm(genes_oi):
    if cool_name == "rao_2014_1kb":
        pooling_file = outputs_dir / (gene + ".pkl")
    else:
        pooling_file = outputs_dir / (gene + "_" + cool_name + ".pkl")
    if (pooling_file).exists():
        pooling_scores = pd.read_pickle(pooling_file)
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
        if gene not in gene_hics:
            continue
        hic, bins_hic = gene_hics[gene]
        # hic, bins_hic = chdm.hic.extract_hic(promoter)
        hic, bins_hic = chdm.hic.clean_hic(hic, bins_hic)

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
        pooling_scores.to_pickle(pooling_file)

# %% [markdown]
# ## Load

# %%
gene_scores = []
for gene in tqdm.tqdm(genes_oi):
    if cool_name == "rao_2014_1kb":
        pooling_file = outputs_dir / (gene + ".pkl")
    else:
        pooling_file = outputs_dir / (gene + "_" + cool_name + ".pkl")
    if (pooling_file).exists():
        pooling_scores = pd.read_pickle(pooling_file)
        gene_scores.append(pooling_scores.assign(gene=gene))
gene_scores = pd.concat(gene_scores)

# %%
x = gene_scores.set_index(["gene", "k"])["cor"].unstack()
x = x / x.max(1).values[:, None]
x = x.iloc[np.argsort(np.argmax(x.values, 1))]
fig, ax = plt.subplots(figsize=(5, 1.5))
norm = mpl.colors.Normalize(0, 1)
mappable = ax.matshow(x.T, aspect="auto", cmap="magma", norm=norm)
ax.set_xlabel("Genes" + f" ({len(gene_scores.gene.unique())})")
ax.set_xticks([])
ax.set_ylabel("Distance")
ax.yaxis.set_major_formatter(chd.plot.DistanceFormatter(base=1e-3))
cbar = fig.colorbar(mappable, ax=ax, pad=0.05)
cbar.set_label(
    "Odds ratio \nhigh Hi-C and \nhigh co-predictivity",
    rotation=0,
    ha="left",
    va="center",
)
ax.set_ylabel("HiC max-pool\ndistance", rotation=0, ha="right", va="center")

if cool_name == "rao_2014_1kb":
    manuscript.save_figure(fig, "5", "maxpool_hic_copredictivity_all_genes")

# %%
gene_scores["logodds"] = np.log(gene_scores["odds"])
gene_scores.loc[np.isinf(gene_scores["logodds"]), "logodds"] = np.nan

# %%
gene_scores.query("k == 0")["logodds"].hist()
# %%
import scanpy as sc

transcriptome.adata.obs["oi"] = pd.Categorical(
    np.array(["noi", "oi"])[
        transcriptome.adata.obs["celltype"]
        .isin(["naive B", "memory B", "Plasma"])
        .values.astype(int)
    ]
)
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
genes_diffexp = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.1").index
# %%
gene_scores_diffexp = gene_scores.query("gene in @genes_diffexp")
gene_scores_nondiffexp = gene_scores.query("gene not in @genes_diffexp")
# %%
# %%
gene_scores.query("k == 0")["logodds"].hist()
gene_scores_diffexp.query("k == 0")["logodds"].hist()
# %%
fig, ax = plt.subplots(figsize=(2, 2))

plotdata = gene_scores.groupby("k")["logodds"].mean()
ax.plot(plotdata.index, np.exp(plotdata))
ax.annotate(
    "All genes",
    (0.5, 0.1),
    ha="center",
    xycoords="axes fraction",
    color=sns.color_palette()[0],
)

plotdata = gene_scores_diffexp.groupby("k")["logodds"].mean()
ax.plot(plotdata.index, np.exp(plotdata))
ax.annotate(
    "B-cell\ngenes",
    (0.95, 0.98),
    ha="right",
    va="top",
    xycoords="axes fraction",
    color=sns.color_palette()[1],
)
ax.set_ylim(ax.get_ylim())
ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(chd.plot.DistanceFormatter(base=1e-3))
ax.set_xlabel("HiC max-pool distance")
ax.set_ylabel(
    "Odds ratio high Hi-C\nand high co-predictivity",
    # rotation=0,
    # ha="right",
    # va="center",
)

if cool_name == "rao_2014_1kb":
    manuscript.save_figure(fig, "5", "maxpool_hic_copredictivity")

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
odds = np.exp(gene_scores.query("k == 0")["logodds"].mean())
q90 = np.exp(gene_scores.query("k == 0")["logodds"].quantile(0.9))
q10 = np.exp(gene_scores.query("k == 0")["logodds"].quantile(0.1))
print(f"odds = {odds:.2f}, q90 = {q90:.2f}, q10 = {q10:.2f}")

odds_diffexp = np.exp(gene_scores_diffexp.query("k == 0")["logodds"].mean())
q90_diffexp = np.exp(gene_scores_diffexp.query("k == 0")["logodds"].quantile(0.9))
q10_diffexp = np.exp(gene_scores_diffexp.query("k == 0")["logodds"].quantile(0.1))
print(f"odds = {odds_diffexp:.2f}, q90 = {q90_diffexp:.2f}, q10 = {q10_diffexp:.2f}")

# %%
import IPython.display

IPython.display.Markdown(
    f"""When comparing the 1kb bins directly, we found a modest but consistent overlap between the two modalities across genes (odds = {odds:.2f}, q10-q90 across genes = {q10:.2f}-{q90:.2f}). Given the cell-type specificity of genome organization [@winick-ngCelltypeSpecializationEncoded2021], this overlap was stronger with genes highly expressed in B-cells (odds = {odds_diffexp:.2f}, q10-q90 across genes = {q10_diffexp:.2f}-{q90_diffexp:.2f})."""
)

# %%
k = 5
odds = np.exp(gene_scores.query("k == @k")["logodds"].mean())
q90 = np.exp(gene_scores.query("k == @k")["logodds"].quantile(0.9))
q10 = np.exp(gene_scores.query("k == @k")["logodds"].quantile(0.1))
print(f"odds = {odds:.2f}, q90 = {q90:.2f}, q10 = {q10:.2f}")

odds_diffexp = np.exp(gene_scores_diffexp.query("k == @k")["logodds"].mean())
q90_diffexp = np.exp(gene_scores_diffexp.query("k == @k")["logodds"].quantile(0.9))
q10_diffexp = np.exp(gene_scores_diffexp.query("k == @k")["logodds"].quantile(0.1))
print(f"odds = {odds_diffexp:.2f}, q90 = {q90_diffexp:.2f}, q10 = {q10_diffexp:.2f}")

# %%
import IPython.display

IPython.display.Markdown(
    f"""Furthermore, when we compared the local co-predictivity with increasingly max-pooled DNA proximity signals, the concordance between the two measurements increased up to 2-fold at about 5kb of max-pooling (FIG:maxpool_hic_copredictivity#, odds = {odds:.2f}, q10-q90 across genes = {q10_diffexp:.2f}-{q90_diffexp:.2f}), bringing it in line with that observed in co-accessibility analyses [@plinerCiceroPredictsCisRegulatory2018]."""
)

# %%
fig, ax = plt.subplots(figsize=(1.0, 1.2))
plotdata = np.exp(gene_scores.query("k == 0")["logodds"].dropna())
sns.ecdfplot(plotdata, ax=ax)
plotdata = np.exp(gene_scores_diffexp.query("k == 0")["logodds"].dropna())
sns.ecdfplot(plotdata, ax=ax)
plotdata = np.exp(gene_scores.query("k == 3")["logodds"].dropna())
sns.ecdfplot(plotdata, ax=ax)
plotdata = np.exp(gene_scores_diffexp.query("k == 3")["logodds"].dropna())
sns.ecdfplot(plotdata, ax=ax)
# ax.boxplot(plotdata.values, positions = [0], widths = [0.8])
# ax.boxplot(plotdata.values, positions=[1], widths = [0.8])
# ax.set_ylim(1, 4)
ax.set_xscale("log")
ax.set_xlim(0.5, 2)


# %%
genes_oi = transcriptome.var.index[:5000]

# %%
fig, ax = plt.subplots(figsize=(1.0, 1.2))
plotdata_all = np.exp(gene_scores_nondiffexp.query("k == 0")["logodds"].dropna())
ax.boxplot(
    plotdata_all.values,
    positions=[0],
    widths=[0.8],
    flierprops={"marker": ".", "markerfacecolor": "k"},
)
plotdata_diffexp = np.exp(gene_scores_diffexp.query("k == 0")["logodds"].dropna())
ax.boxplot(
    plotdata_diffexp.values,
    positions=[1],
    widths=[0.8],
    flierprops={"marker": ".", "markerfacecolor": "k"},
)
ax.set_ylim(1, 4)
ax.set_yscale("log")
# ax.set_xlim(0.5, 8)

# %%
import scipy.stats

scipy.stats.ranksums(plotdata_all, plotdata_diffexp)

# %%
