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

import pickle

import scanpy as sc

import tqdm.auto as tqdm

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
regions_name, window = "100k100k", np.array([-100000, 100000])
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

# %% [markdown]
# ### Get SNPs

# %%
import chromatinhd.data.associations

# %%
motifscan_name = "gwas_immune"

motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / regions_name / motifscan_name
)
motifscan = chd.data.associations.Associations(motifscan_folder)
association = motifscan.association
# motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
# motifs["ix"] = np.arange(motifs.shape[0])

# association = pd.read_pickle(motifscan_folder / "association.pkl")

# %%
regionpairwindow = chd.models.pred.interpret.RegionPairWindow(prediction.path / "scoring" / "regionpairwindow2")
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(prediction.path / "scoring" / "regionmultiwindow2")

# %%
# Calculate for each gene and window the amount of overlapping SNPs

genewindowscores = []

genes_oi = transcriptome.var.index#[-100:]
for gene in tqdm.tqdm(genes_oi):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    promoter = fragments.regions.coordinates.loc[gene]
    if promoter["chrom"] == "chr6":
        continue

    if gene == "ENSG00000239713":
            continue

    if gene in regionpairwindow.interaction:
        # print(gene)
        interaction = regionpairwindow.interaction[gene].mean("fold").to_dataframe("cor").reset_index()

        interaction["window1_mid"] = regionmultiwindow.design.loc[interaction["window1"], "window_mid"].values
        interaction["window2_mid"] = regionmultiwindow.design.loc[interaction["window2"], "window_mid"].values

        interaction["distance"] = np.abs(interaction["window1_mid"] - interaction["window2_mid"])

        gene_ix = transcriptome.var.index.tolist().index(gene)

        position_indices = (motifscan.coordinates[motifscan.region_indptr[gene_ix]:motifscan.region_indptr[gene_ix+1]] % fragments.regions.region_width) + fragments.regions.window[0]

        interaction["abscor"] = np.abs(interaction["cor"])
        windowscores = (
            interaction.query("distance > 1000")
            .sort_values("cor", ascending=False)
            .groupby("window1")
            .mean(numeric_only=True)
        )
        windowscores.index.name = "window"
        window_width = 100
        windowscores["start"] = windowscores["window1_mid"] - window_width / 2
        windowscores["end"] = windowscores["window1_mid"] + window_width / 2

        y = (position_indices[None, :] > windowscores["start"].values[:, None]) & (
            position_indices[None, :] < windowscores["end"].values[:, None]
        )
        windowscores["n_snps"] = y.sum(1)

        windowscores = windowscores.join(regionmultiwindow.scores["deltacor"].sel_xr(gene).sel(phase="test").mean("fold").to_dataframe("deltacor"))

        genewindowscores.append(windowscores.assign(gene=gene))

        # break

genewindowscores = pd.concat(genewindowscores)

# %%
np.corrcoef(genewindowscores["cor"], genewindowscores["deltacor"])

# %%
np.corrcoef(genewindowscores["cor"], genewindowscores["n_snps"])

# %%
np.corrcoef(genewindowscores["deltacor"], genewindowscores["n_snps"])

# %%
np.corrcoef(genewindowscores["deltacor"], genewindowscores["n_snps"])

# %%
q = 0.9
genewindowscores_oi = genewindowscores
a1 = genewindowscores_oi["deltacor"].values <= np.quantile(
    genewindowscores_oi["deltacor"].values, 1 - q
)
a2 = genewindowscores_oi["cor"].values >= np.quantile(
    genewindowscores_oi["cor"].values, q
)

b = genewindowscores_oi["n_snps"].values > 0
contingency_prediction = [
    [np.sum(a1 & b), np.sum(a1 & ~b)],
    [np.sum(~a1 & b), np.sum(~a1 & ~b)],
]
odds_prediction = (
    contingency_prediction[0][0]
    * contingency_prediction[1][1]
    / contingency_prediction[0][1]
    / contingency_prediction[1][0]
)
captured_prediction = np.sum(a1 & b) / np.sum(b)

contingency_interaction = [
    [np.sum(a2 & b), np.sum(a2 & ~b)],
    [np.sum(~a2 & b), np.sum(~a2 & ~b)],
]
odds_interaction = (
    contingency_interaction[0][0]
    * contingency_interaction[1][1]
    / contingency_interaction[0][1]
    / contingency_interaction[1][0]
)
captured_interaction = np.sum(a2 & b) / np.sum(b)

# %%
odds_interaction

# %%
odds_prediction

# %%
scores = []
for q in [0.1, 0.5, 0.9]:
# for q in [0.925, 0.95, 0.975, 0.99, 0.999]:
    for gene, genewindowscores_oi in genewindowscores.groupby("gene"):
        a1 = genewindowscores_oi["deltacor"].values <= np.quantile(
            genewindowscores_oi["deltacor"].values, 1 - q
        )
        a2 = genewindowscores_oi["cor"].values >= np.quantile(
            genewindowscores_oi["cor"].values, q
        )

        b = genewindowscores_oi["n_snps"].values > 0
        contingency_prediction = [
            [np.sum(a1 & b), np.sum(a1 & ~b)],
            [np.sum(~a1 & b), np.sum(~a1 & ~b)],
        ]
        odds_prediction = (
            contingency_prediction[0][0]
            * contingency_prediction[1][1]
            / contingency_prediction[0][1]
            / contingency_prediction[1][0]
        )
        captured_prediction = np.sum(a1 & b) / np.sum(b)

        contingency_interaction = [
            [np.sum(a2 & b), np.sum(a2 & ~b)],
            [np.sum(~a2 & b), np.sum(~a2 & ~b)],
        ]
        odds_interaction = (
            contingency_interaction[0][0]
            * contingency_interaction[1][1]
            / contingency_interaction[0][1]
            / contingency_interaction[1][0]
        )
        captured_interaction = np.sum(a2 & b) / np.sum(b)

        x3 = genewindowscores_oi["cor"].values / (
            -genewindowscores_oi["deltacor"].values
        )
        a3 = x3 >= np.quantile(x3, q)
        contingency_both = [
            [np.sum(a3 & b), np.sum(a3 & ~b)],
            [np.sum(~a3 & b), np.sum(~a3 & ~b)],
        ]
        odds_both = (
            contingency_both[0][0]
            * contingency_both[1][1]
            / contingency_both[0][1]
            / contingency_both[1][0]
        )
        captured_both = np.sum(a3 & b) / np.sum(b)

        captured_random = []
        for i in range(100):
            captured_random.append(np.sum(np.random.permutation(a3) & b) / np.sum(b))
        captured_random = np.mean(captured_random)

        scores.append(
            {
                "gene": gene,
                "odds_prediction": odds_prediction,
                "odds_interaction": odds_interaction,
                "captured_interaction": captured_interaction,
                "captured_prediction": captured_prediction,
                "captured_both": captured_both,
                "captured_random": captured_random,
                "n_snps": b.sum(),
                "contingency_prediction": contingency_prediction,
                "contingency_interaction": contingency_interaction,
                "contingency_both": contingency_both,
                "q": q,
            }
        )
scores = pd.DataFrame(scores)

# %%
(
    scores.query("q == 0.9").query("n_snps > 0")["captured_interaction"].mean(),
    scores.query("q == 0.9").query("n_snps > 0")["captured_prediction"].mean(),
    scores.query("q == 0.9").query("n_snps > 0")["captured_both"].mean(),
    scores.query("q == 0.9").query("n_snps > 0")["captured_random"].mean(),
)

# %%
qscores = scores.groupby("q").agg(
    {
        "contingency_prediction": lambda x: np.stack(x).sum(0),
        "contingency_interaction": lambda x: np.stack(x).sum(0),
        "contingency_both": lambda x: np.stack(x).sum(0),
    }
)
qscores = qscores.join(
    scores.groupby("q")[
        [
            "captured_interaction",
            "captured_prediction",
            "captured_both",
            "captured_random",
        ]
    ].mean()
)


qscores["odds_prediction"] = qscores["contingency_prediction"].apply(
    lambda x: x[0][0] * x[1][1] / x[0][1] / x[1][0]
)
qscores["odds_interaction"] = qscores["contingency_interaction"].apply(
    lambda x: x[0][0] * x[1][1] / x[0][1] / x[1][0]
)
qscores["odds_both"] = qscores["contingency_both"].apply(
    lambda x: x[0][0] * x[1][1] / x[0][1] / x[1][0]
)
# %%
fig, ax = plt.subplots()
ax.plot(qscores.index, qscores["odds_prediction"])
ax.plot(qscores.index, qscores["odds_interaction"])
ax.plot(qscores.index, qscores["odds_both"])

# %%
qscores["ratio_interaction"] = (
    qscores["captured_interaction"] / qscores["captured_random"]
)
qscores["ratio_prediction"] = (
    qscores["captured_prediction"] / qscores["captured_random"]
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    qscores.index.astype(str),
    qscores["ratio_interaction"],
    label="average co-predictivity",
)
ax.plot(
    qscores.index.astype(str),
    qscores["ratio_prediction"],
    label="predictivity",
)
ax.set_xlabel("q")
ax.set_ylabel("Odds-ratio SNP \nand high score\n(score $\geq$ q(score))")
ax.annotate(
    "average\nco-predictivity\n(cor $\\Delta$cor)",
    (2, qscores.iloc[-3]["ratio_interaction"]),
    xytext=(10, 10),
    ha="right",
    # xycoords="axes fraction",
    textcoords="offset points",
    color=sns.color_palette()[0],
    # arrowprops=dict(arrowstyle="-", color=sns.color_palette()[0]),
)
ax.annotate(
    "predictivity\n($\\Delta$cor)",
    (2, qscores.iloc[-3]["ratio_prediction"]),
    xytext=(5, -20),
    ha="left",
    # xycoords="axes fraction",
    textcoords="offset points",
    color=sns.color_palette()[1],
    # arrowprops=dict(arrowstyle="-", color=sns.color_palette()[1]),
)
ax.set_yscale("log")
ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
# ax.legend()

manuscript.save_figure(fig, "5", "copredictivity_gwas_overlap")


# %%
genescores = pd.DataFrame(
    {
        "cor_cor_snps": genewindowscores.groupby("gene")
        .apply(lambda x: np.corrcoef(x["cor"], x["n_snps"] > 1)[0, 1])
        .dropna()
        .sort_values(),
        "cor": genewindowscores.groupby("gene")["cor"].max(),
        "n_snps": genewindowscores.groupby("gene")["n_snps"].sum(),
        "n_positions": genewindowscores.groupby("gene").size(),
    }
)
genescores["symbol"] = transcriptome.var["symbol"]
# %%
genescores.dropna().sort_values("cor", ascending=False).head(30)
# %%
fig, ax = plt.subplots()
ax.scatter(genescores["n_positions"], genescores["cor"])
# np.corrcoef(genewindowscores.groupby("gene").size(), genewindowscores.groupby("gene")["cor"].mean())
# %%
np.corrcoef(
    genewindowscores.groupby("gene").size(),
    genewindowscores.groupby("gene")["cor"].mean(),
)
