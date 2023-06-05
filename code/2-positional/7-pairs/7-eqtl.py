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

import pickle

import scanpy as sc

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")


# %%
device = "cuda:0"
# device = "cpu"

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
prediction_name = "v21"

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

# %%
scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
interaction_file = scores_folder / "interaction.pkl"

promoter = promoters.loc[gene]

# remove the MHC region, messing with everything all the time
if promoter["chr"] == "chr6":
    continue

if interaction_file.exists():
    interaction = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
    interaction = interaction.rename(columns={0: "cor"})

# %%
# lowess residuals
from statsmodels.nonparametric.smoothers_lowess import lowess

b = interaction_windows["cor"]
c = interaction_windows["deltacor1"]

b2 = lowess(
    c,
    b,
    return_sorted=False,
)

# %%
plt.scatter(b, c)
plt.scatter(b, b2)

# %%
plt.scatter(c, b2)
plt.scatter(c, b2 - b)

# %%


# %%

# %%
interaction_oi = interaction.query("distance > 2000").query("cor > 0").copy()
interaction_oi["deltacor_prod"] = np.abs(interaction_oi["deltacor1"]) * np.abs(
    interaction_oi["deltacor2"]
)
# %%
plt.scatter(interaction_oi["deltacor_prod"], interaction_oi["cor"])
# %%
# lowess residuals
from statsmodels.nonparametric.smoothers_lowess import lowess

b = interaction_oi["deltacor_prod"]
c = interaction_oi["cor"]

b2 = lowess(
    c,
    b,
    return_sorted=False,
)

# %%
plt.scatter(b, c)
plt.scatter(b, b2)

# %%
interaction_oi["cor_resid"] = interaction_oi["cor"] - b2

# %%
def calculate_cor_resid(interaction_oi):
    # lowess residuals
    from statsmodels.nonparametric.smoothers_lowess import lowess

    b = interaction_oi["deltacor_prod"]
    c = interaction_oi["cor"]

    b2 = lowess(
        c,
        b,
        return_sorted=False,
    )

    interaction_oi["cor_resid"] = interaction_oi["cor"] - b2
    return interaction_oi


# %%
sns.heatmap(interaction_oi.set_index(["window1", "window2"])["cor"].unstack())
# %%
sns.heatmap(
    interaction_oi.set_index(["window1", "window2"])["cor_resid"].unstack(),
    vmin=-0.1,
    vmax=0.1,
    cmap="RdBu_r",
)

# %%
sns.heatmap(
    interaction_oi.set_index(["window1", "window2"])["cor"].unstack(),
    vmin=-0.1,
    vmax=0.1,
    cmap="RdBu_r",
)

# ### Get SNPs

# %%
motifscan_name = "gwas_immune"

motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
motifscan.n_motifs = len(motifs)
motifs["ix"] = np.arange(motifs.shape[0])

# %%
scores = []

genes_oi = transcriptome.var.index[:3000]
A = []
B = []
C = []
D = []
G = []
for gene in tqdm.tqdm(genes_oi):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    promoter = promoters.loc[gene]

    # remove the MHC region, messing with everything all the time
    if promoter["chr"] == "chr6":
        continue

    if interaction_file.exists():
        interaction = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
        interaction = interaction.rename(columns={0: "cor"})

        gene_ix = transcriptome.var.index.tolist().index(gene)

        indptr_start = gene_ix * (window[1] - window[0])
        indptr_end = (gene_ix + 1) * (window[1] - window[0])

        indptr = motifscan.indptr[indptr_start:indptr_end]
        motif_indices = motifscan.indices[indptr[0] : indptr[-1]]
        position_indices = chd.utils.indptr_to_indices(indptr - indptr[0]) + window[0]

        y = (position_indices[None, :] > interaction["window1"].values[:, None]) & (
            position_indices[None, :] < interaction["window2"].values[:, None]
        )

        windowscores = (
            interaction.query("distance > 1000")
            .sort_values("cor", ascending=False)
            .groupby("window1")
            .mean(numeric_only=True)
        )
        windowscores["start"] = windowscores.index - window_width / 2
        windowscores["end"] = windowscores.index + window_width / 2

        y = (position_indices[None, :] > windowscores["start"].values[:, None]) & (
            position_indices[None, :] < windowscores["end"].values[:, None]
        )
        windowscores["n_snps"] = y.sum(1)

        width = 100
        b = np.zeros(window[1] - window[0])
        c = np.zeros(window[1] - window[0])

        if (len(interaction) > 0) and (len(position_indices) > 0):
            interaction_windows = (
                interaction.query("distance > 1000")
                .groupby("window1")
                .mean(numeric_only=True)
            )
            interaction_windows["cor_abs"] = np.abs(interaction_windows["cor"])
            for window1, windowscores in interaction_windows.iterrows():
                b[
                    int(window1 - width / 2 + window[0]) : int(
                        window1 + width / 2 + window[0]
                    )
                    # ] = windowscores["cor"]
                ] = windowscores["cor_abs"]
                c[
                    int(window1 - width / 2 + window[0]) : int(
                        window1 + width / 2 + window[0]
                    )
                ] = windowscores["deltacor1"]

            a = np.zeros(window[1] - window[0], dtype=bool)
            a[position_indices + window[0]] = True

            A.append(a)
            B.append(b)
            C.append(c)
            G.append(gene)

            # contingency = [
            #     [np.sum(~a & ~b), np.sum(~a & b)],
            #     [np.sum(a & ~b), np.sum(a & b)],
            # ]

            # scores.append(
            #     {
            #         "gene": gene,
            #         "n": len(position_indices),
            #         "n_interact": len(interaction),
            #         "contingency": contingency,
            #     }
            # )

# %%
A = np.stack(A)
B = np.stack(B)
C = np.stack(C)

# %%
plt.scatter(np.log1p(A.sum(1)), np.log1p((C != 0).sum(1)))

# %%
scores = []
for q in tqdm.tqdm(np.linspace(0.95, 1, 20)[:-1]):
    score = {"q": q}

    B_ = B >= np.quantile(B, q)
    contingency = [
        [np.sum(~A & ~B_), np.sum(~A & B_)],
        [np.sum(A & ~B_), np.sum(A & B_)],
    ]
    odds = contingency[0][0] * contingency[1][1] / contingency[0][1] / contingency[1][0]

    score.update({"odds_interaction": odds})

    B_ = -C >= np.quantile(-C, q)
    contingency = [
        [np.sum(~A & ~B_), np.sum(~A & B_)],
        [np.sum(A & ~B_), np.sum(A & B_)],
    ]
    odds = contingency[0][0] * contingency[1][1] / contingency[0][1] / contingency[1][0]

    score.update({"odds_predictivity": odds})

    scores.append(score)

# %%
scores = pd.DataFrame(scores)
fig, ax = plt.subplots()
ax.plot(scores["q"], scores["odds_interaction"], label="interaction")
ax.plot(scores["q"], scores["odds_predictivity"], label="predictivity")
ax.set_yscale("log")
plt.legend()

# %%
import scipy

scores = []
for gene_ix, gene in tqdm.tqdm(enumerate(G)):
    a = A[gene_ix]
    score = {
        "gene": gene,
        "rank_interaction": scipy.stats.rankdata(
            B[gene_ix] / (-C[gene_ix] + 0.0001), method="min"
        )[a].max(),
        "rank_prediction": scipy.stats.rankdata(-C[gene_ix], method="min")[a].max(),
    }

    scores.append(score)

scores = pd.DataFrame(scores)

# %%
fig, ax = plt.subplots()
ax.violinplot(scores["rank_interaction"], positions=[1])
ax.violinplot(scores["rank_prediction"], positions=[2])

# %%
(scores["rank_interaction"] > scores["rank_prediction"]).mean()

# %%
(scores["rank_interaction"] < scores["rank_prediction"]).mean()

# %%
scores["rank_interaction"].median()
# %%
scores["rank_prediction"].median()
# %%


# %%
