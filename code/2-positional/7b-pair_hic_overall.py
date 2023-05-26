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

# %% [markdown]
# ## Transcriptome diffexp

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
genes_oi = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.5").index

# %% [markdown]
# ## Pairwindow

genes_all = transcriptome.var.index
# genes_all = transcriptome.var.query("symbol in ['BCL2']").index

scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)
genes_all = (
    nothing_scoring.genescores.mean("model")
    .sel(phase=["test", "validation"])
    .mean("phase")
    .sel(i=0)
    .to_pandas()
    .query("cor > 0.1")
    .sort_values("cor", ascending=False)
    .index
)

scores = []
for gene in tqdm.tqdm(genes_all):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if interaction_file.exists():
        scores_gene = pd.read_pickle(interaction_file).reset_index()
        scores_gene = scores_gene.loc[
            (scores_gene["lost1"] > 2) & (scores_gene["lost2"] > 2)
        ]
        scores_gene2 = scores_gene[["cor", "window1", "window2"]].copy()
        scores_gene2["window1"] = scores_gene2["window1"].astype("category")
        scores_gene2["window2"] = scores_gene2["window2"].astype("category")
        scores_gene["gene"] = gene
        scores_gene["gene"] = pd.Categorical(scores_gene["gene"], categories=genes_all)
        scores.append(scores_gene)

    # if len(scores) > 500:
    #     break
scores = pd.concat(scores)
print(len(scores["gene"].unique()))

# %%
scores["distance"] = np.abs(scores["window1"] - scores["window2"])
# %%
scores_far = scores.query("distance > 1000").copy()
scores_significant = scores_far

# %%
scores_significant.query("gene in @genes_oi").groupby("gene").first().sort_values(
    "cor", ascending=False
).head(10).assign(symbol=lambda x: transcriptome.var.loc[x.index, "symbol"].values)
scores_significant.query("gene in @genes_oi").groupby("gene").size().to_frame(
    "n"
).sort_values("n", ascending=False).head(10).assign(
    symbol=lambda x: transcriptome.var.loc[x.index, "symbol"].values
)

# %%
bins = np.linspace(0, (window[1] - window[0]), 20)
scores_significant["bin"] = pd.Categorical(
    bins[
        np.digitize(
            scores_significant["distance"],
            bins,
        )
    ],
    categories=bins,
)
synergism_distance = pd.DataFrame(
    {
        # "total": np.bincount(
        #     np.digitize(
        #         scores_far["distance"],
        #         bins,
        #     ),
        #     minlength=len(bins),
        # ),
        # "synergistic": np.bincount(
        #     np.digitize(
        #         scores_significant["distance"],
        #         bins,
        #     ),
        #     minlength=len(bins),
        # ),
        "synergistic": scores_significant.groupby("bin")["cor"].mean(),
        "bin": bins,
    }
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_distance["bin"],
    synergism_distance["synergistic"],
    color="black",
)

# %%
scores_significant_melted = pd.melt(
    scores_significant[["window1", "window2", "cor"]],
    value_vars=["window1", "window2"],
    id_vars=["cor"],
)
bins = np.linspace(*window, 20)
scores_significant_melted["bin"] = pd.Categorical(
    bins[
        np.digitize(
            scores_significant_melted["value"],
            bins,
        )
    ],
    categories=bins,
)

scores["bin"] = pd.Categorical(
    bins[
        np.digitize(
            scores["window2"],
            bins,
        )
    ],
    categories=bins,
)
synergism_distance = pd.DataFrame(
    {
        # "total": np.bincount(
        #     np.digitize(
        #         scores_far["distance"],
        #         bins,
        #     ),
        #     minlength=len(bins),
        # ),
        # "synergistic": np.bincount(
        #     np.digitize(
        #         scores_significant["distance"],
        #         bins,
        #     ),
        #     minlength=len(bins),
        # ),
        "synergistic": scores_significant_melted.groupby("bin")["cor"].mean(),
        "synergistic2": scores.groupby("bin")["deltacor2"].mean(),
        "bin": bins,
    }
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_distance["bin"],
    synergism_distance["synergistic"] / np.abs(synergism_distance["synergistic2"]),
    color="black",
)
ax2 = ax.twinx()
ax2.plot(
    synergism_distance["bin"],
    synergism_distance["synergistic"],
    color="orange",
)


# %% [markdown]
# ## HiC distance correspondence
genes_oi = transcriptome.gene_id(["BCL2", "CD74", "CD79A"])

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
gene_scores_raw = {}
for gene in genes_oi:
    # load hic
    promoter = promoters.loc[gene]
    hic, bins_hic = chdm.hic.extract_hic(promoter)
    hic = clean_hic(hic, bins_hic)

    # load scores
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if interaction_file.exists():
        scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
    else:
        raise ValueError("No scores found")
    assert len(scores_oi) > 0

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
                    **compare_contingency(*(matching_oi[["cor", "balanced"]].values.T)),
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
    gene_scores_raw["gene"] = pooling_scores.assign(gene=gene)

# %%
gene_scores = pd.concat(gene_scores)
# %%
gene_scores.groupby("k")["odds"].mean().plot()
# %%
