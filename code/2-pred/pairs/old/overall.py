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
folder_dataset = chd.get_output() / "datasets" / dataset_name

transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")

splitter = "5x5"
regions_name = "100k100k"
prediction_name = "v30"

# fragments
fragments = chd.data.Fragments(folder_dataset / "fragments" / regions_name)

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output() / "pred" / dataset_name / regions_name / splitter / "magic" / prediction_name
)

# %% [markdown]
# ## Transcriptome diffexp

# %%
transcriptome.adata = transcriptome.adata[:, transcriptome.var.index]

# %%
import scanpy as sc

sc.tl.rank_genes_groups(transcriptome.adata, groupby="celltype", use_raw = False)
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

# %%
genes_all = transcriptome.var.index
genes_all = transcriptome.var.query("symbol in ['BCL2']").index

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
        # scores_gene = scores_gene.loc[
        #     (scores_gene["lost1"] > 2) & (scores_gene["lost2"] > 2)
        # ]
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
sns.histplot(scores_far.query("cor > 0.1")["window1"].sample(100000), bins=100)
sns.histplot(scores_far.query("cor < -0.1")["window1"].sample(100000), bins=100)

# %%
scores_significant.query("gene in @genes_oi").groupby("gene").first().sort_values(
    "cor", ascending=False
).head(10).assign(symbol=lambda x: transcriptome.var.loc[x.index, "symbol"].values)
scores_significant.query("gene in @genes_oi").groupby("gene").size().to_frame(
    "n"
).sort_values("n", ascending=False).head(10).assign(
    symbol=lambda x: transcriptome.var.loc[x.index, "symbol"].values
)

# %% [markdown]
# ## Plot distance co-predictivity

# %%
bins = np.linspace(1000, (window[1] - window[0]), 50)
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
        "synergistic": scores_significant.groupby("bin")["cor"].mean(),
        "bin": bins,
    }
)

from statsmodels.nonparametric.smoothers_lowess import lowess

synergism_distance["synergistic_smooth"] = lowess(
    synergism_distance["synergistic"],
    synergism_distance["bin"],
    frac=0.2,
    return_sorted=False,
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.scatter(
    synergism_distance["bin"],
    synergism_distance["synergistic"],
    color="black",
    s=1,
)
ax.plot(
    synergism_distance["bin"],
    synergism_distance["synergistic_smooth"],
    color="red",
)
ax.set_xlim(1000)
ax.set_ylim(0)
ax.set_xticks([1000, 100000, 200000])
ax.xaxis.set_major_formatter(chd.plotting.distance_ticker)
ax.set_ylabel("Average co-predictivity\n(cor between $\\Delta$ cor)")
ax.set_xlabel("Distance")
manuscript.save_figure(fig, "5", "synergism_distance")

# %% [markdown]
# ## Plot position co-predictivity

# %%
scores_significant_melted = pd.melt(
    scores_significant[["window1", "window2", "cor"]],
    value_vars=["window1", "window2"],
    id_vars=["cor"],
)
bins = np.linspace(*window, 100)
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
        "synergistic": scores_significant_melted.groupby("bin")["cor"].mean(),
        "synergistic2": scores.groupby("bin")["deltacor2"].mean(),
        "bin": bins,
    }
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.scatter(
    synergism_distance["bin"], synergism_distance["synergistic"], color="black", s=1
)

# loess
from statsmodels.nonparametric.smoothers_lowess import lowess

synergism_distance["synergistic_smooth"] = lowess(
    synergism_distance["synergistic"],
    synergism_distance["bin"],
    frac=0.2,
    return_sorted=False,
)
ax.plot(
    synergism_distance["bin"],
    synergism_distance["synergistic_smooth"],
    color="red",
)
ax.set_ylim(0)
ax.set_xlim(*window)
ax.xaxis.set_major_formatter(chd.plotting.gene_ticker)
ax.set_ylabel("Average co-predictivity\n(cor between $\\Delta$ cor)")

manuscript.save_figure(fig, "5", "synergism_position")

# %% [markdown]
# ## Plot promoter co-predictivity
# additive_vs_nonadditive_gene_scores = pd.read_pickle(
#     chd.get_output() / "additive_vs_nonadditive_gene_scores.pkl"
# )

# %%
genescores = pd.DataFrame(
    {
        "cor_tss": scores.loc[scores["window1"] == -50.0]
        .query("distance > 1000")
        .groupby("gene")["cor"]
        .mean()
        .fillna(0.0)
    }
)
genescores["diff"] = additive_vs_nonadditive_gene_scores["diff"]
# %%
plt.scatter(genescores["cor_tss"], genescores["diff"], s=1)
# %%
fig, ax = plt.subplots(figsize=(2, 2))
sns.ecdfplot(scores_significant["cor"].values[:1000], label="All pairs")
sns.ecdfplot(
    scores_significant.query("(effect1 * effect2) < 0")["cor"].values[:1000],
    label="Different effect pairs",
)
sns.ecdfplot(genescores["cor_tss"].values[:1000], label="TSS")
sns.ecdfplot(
    genescores.query("diff > 0.05")["cor_tss"].values[:1000],
    label="TSS with perforamce",
)
ax.legend()

# %%
scores_significant["effect_prod"] = (
    scores_significant["effect1"] * scores_significant["effect2"]
)

scores_significant["corabs"] = np.abs(scores_significant["cor"])

# %%
fig, ax = plt.subplots(figsize=(1.5, 1.5))
sns.ecdfplot(
    scores_significant["cor"].values[
        np.random.choice(len(scores_significant), 1000, replace=False)
    ],
    label="All pairs",
    ax=ax,
)
sns.ecdfplot(
    scores_significant.query("effect_prod < 0")["cor"].values[:1000],
    label="Different effect",
    ax=ax,
)
sns.ecdfplot(
    scores_significant.query("effect_prod > 0")["cor"].values[:1000],
    label="Same effect",
    ax=ax,
)
ax.axvline(0.0, color="black", linestyle="--")
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=8)
ax.set_xlim(-0.05, 0.1)
ax.set_xlabel("Co-predictivity (cor $\Delta$ cor)")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_ylabel("Pairs")

manuscript.save_figure(fig, "5", "copredictivity_direction")


# %%
fig, ax = plt.subplots(figsize=(1.5, 1.5))
sns.ecdfplot(genescores["cor_tss"].values[:1000], label="All genes", ax=ax)
sns.ecdfplot(
    genescores.query("diff > 0.05")["cor_tss"].values[:1000],
    label="$\Delta\Delta$cor > 0.05",
    ax=ax,
)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=8)
ax.axvline(0.0, color="black", linestyle="--")
ax.set_xlim(-0.05, 0.1)
ax.set_xlabel("Co-predictivity (cor $\Delta$ cor)")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_ylabel("Pairs with TSS")

manuscript.save_figure(fig, "5", "copredictivity_different")


# %%
x = scores_significant["cor"]

# %%
sns.ecdfplot(x.values[np.random.choice(len(x), 1000, replace=False)])
