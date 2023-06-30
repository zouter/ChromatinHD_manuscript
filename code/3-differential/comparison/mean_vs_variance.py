# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import IPython

if IPython.get_ipython():
    IPython.get_ipython().magic("load_ext autoreload")
    IPython.get_ipython().magic("autoreload 2")
    IPython.get_ipython().magic("config InlineBackend.figure_format='retina'")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %% [markdown]
# ## Full real

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "KLF4_7"
# dataset_name = "MSGN1_7"
# dataset_name = "morf_20"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %% [markdown]
# ### Load latent space

# %%
# latent = torch.from_numpy(simulation.cell_latent_space).to(torch.float32)

# using transcriptome clustering
# sc.tl.leiden(transcriptome.adata, resolution = 0.1)
# latent = torch.from_numpy(pd.get_dummies(transcriptome.adata.obs["leiden"]).values).to(torch.float)

# loading
latent_name = "leiden_1"
latent_name = "leiden_0.1"
# latent_name = "celltype"
# latent_name = "overexpression"
folder_data_preproc = folder_data / dataset_name
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
latent_torch = torch.from_numpy(latent.values).to(torch.float)

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
cluster_info["color"] = sns.color_palette("husl", latent.shape[1])
transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = pd.Categorical(
    pd.from_dummies(latent).iloc[:, 0]
)

# %%
# method_name = 'v4_128-64-32_30_rep'
# method_name = 'v8_128-64-32'
method_name = "v9_128-64-32"


class Prediction(chd.flow.Flow):
    pass


prediction = Prediction(
    chd.get_output()
    / "prediction_likelihood"
    / dataset_name
    / promoter_name
    / latent_name
    / method_name
)
model = chd.load((prediction.path / "model_0.pkl").open("rb"))

# %%
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

probs_diff = probs - probs.mean(1, keepdims=True)
# mixture_diff = mixtures - mixtures.mean(1, keepdims=True)

# %%
design["gene_ix"] = design["gene_ix"]

# %%
x = (
    (design["coord"].values)
    .astype(int)
    .reshape(
        (
            len(design["gene_ix"].cat.categories),
            len(design["active_latent"].cat.categories),
            len(design["coord"].cat.categories),
        )
    )
)
y = probs_diff

# %%
desired_x = torch.arange(*window)

# %%
probs_interpolated = chd.utils.interpolate_1d(
    desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
).numpy()

# %% [markdown]
# ## Accessibility vs differential

# %%
prob_cutoff = np.log(1.0)

# %%
heights_diff_max = np.log(
    (np.exp(probs) - np.exp(probs.mean(1, keepdims=True))).max(1).flatten()
)

# %%
locusscores = pd.DataFrame(
    {
        "probs_std": probs.std(1).flatten(),
        "probs_diff_max": probs_diff.max(1).flatten(),
        "heights_diff_max": heights_diff_max,
        # "probs_mean":probs.min(1).flatten(),
        "probs_mean": probs.mean(1).flatten(),
        "probs_mean_high": (probs.mean(1) > 0.0).flatten(),
        "prob_high_any": (probs > prob_cutoff).any(1).flatten(),
    }
)

# %%
# probs_mean_bins = np.arange(5, 11)
# probs_mean_bins = np.arange(-2, 4)
probs_mean_bins = np.log(np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]))

# %%
fig, ax = plt.subplots()
locusscores_oi = locusscores.sample(100000)
ax.set_ylim(0, locusscores_oi["probs_diff_max"].quantile(0.95))
ax.scatter(
    np.exp(locusscores_oi["probs_mean"]),
    locusscores_oi["probs_diff_max"],
    s=0.1,
    alpha=0.1,
)
for bin in probs_mean_bins:
    ax.axvline(np.exp(bin))
ax.set_xscale("log")
ax.set_xlabel("Accessibility")
ax.set_xlabel("Maximal fold-change")

# %%
locusscores["probs_mean_bin"] = pd.Categorical(
    np.digitize(locusscores["probs_mean"], probs_mean_bins)
)

# %%
bins = locusscores["probs_mean_bin"].cat.categories

# %%
labels = [
    f"< {round(i, 2)}"
    for i in (np.exp(probs_mean_bins[[0]]) / (window[1] - window[0]) * 100 * 100)
] + [
    f"≥ {round(i, 2)}"
    for i in (np.exp(probs_mean_bins) / (window[1] - window[0]) * 100 * 100)
]
bin_info = pd.DataFrame(
    {
        "bin": bins.tolist(),
        "label": labels,
    }
).set_index("bin")
norm = mpl.colors.Normalize(bins.min() - 1, bins.max())
cmap = mpl.cm.YlOrRd
bin_info["color"] = cmap(norm(bin_info.index)).tolist()

# %%
metric = "probs_diff_max"
metric_label = "Max fold difference in accessibility"
# metric = "heights_diff_max"; metric_label = "Max height difference in accessibility"
# metric = "probs_std"; metric_label = "Std in accessibility"

# %%
fig, axes = plt.subplots(
    len(bins),
    1,
    figsize=(2.5, len(bins) / 4),
    sharex=True,
    sharey=False,
    gridspec_kw={"hspace": 0},
)
for i, (bin, locusscores_bin) in enumerate(locusscores.groupby("probs_mean_bin")):
    ax = axes[i]
    ax.hist(
        np.exp(locusscores_bin[metric]),
        density=True,
        bins=np.logspace(np.log10(1), np.log10(4), 50),
        color="grey",
        # color=bin_info.loc[bin, "color"],
        lw=0,
    )
    ax.set_ylim(0, 2)
    # ax.set_xlim(0, 2)
    ax.set_xscale("log")
    ax.axvline(np.exp(locusscores_bin[metric].mean()), color="#333")
    ax.set_yticks([ax.get_ylim()[-1] / 2])
    ax.set_yticklabels([bin_info.loc[bin, "label"]])
    ax.set_ylabel("")
    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)

ax = axes[int(len(axes) / 2) - 1]
ax.set_ylabel(
    # "Mean accessibility\n $\\frac{\\mathrm{\#\;Tn5\;insertions}}{\\mathrm{100\;nucleotides} \\times \\mathrm{100\;cells}}$",
    "Mean\naccessibility",
    ha="right",
    va="center",
    rotation=0,
)

ax = axes[-1]
ax.set_xlabel(metric_label)
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([1, 2, 3, 4])

manuscript.save_figure(fig, "4", "mean_vs_variance_relationship")

# %%
fig, ax = plt.subplots(figsize=(4, 4))
norm = mpl.colors.Normalize(bins.min(), bins.max())
cmap = mpl.cm.YlOrRd
ax.set_xlim(0.0, 1.0)
for i, (bin, locusscores_bin) in enumerate(locusscores.groupby("probs_mean_bin")):
    sns.ecdfplot(locusscores_bin[metric], ax=ax, color=cmap(norm(bin)))
ax.set_xlabel(metric_label)

# %% [markdown]
# ### Per gene

# %%
# gene_id = "AAK1"
gene_id = "FOS"
gene_id = "IRF7"
gene_id = "CD40"
gene_id = "CD74"
# gene_id = "TYMP"
# gene_id = "PAX5"
# gene_id = "Neurod2"
# gene_id = "Vtn"
# gene_id = "CEMIP"
gene_oi = transcriptome.gene_ix(gene_id)

# %%
locusscores = pd.DataFrame(
    {
        # "expprobs_std":np.log(np.exp(mixtures)[gene_oi].std(0).flatten()),
        "probs_std": probs[gene_oi].std(0).flatten(),
        "probs_diff_max": probs_diff[gene_oi].max(0).flatten(),
        "probs_mean": probs[gene_oi].mean(0).flatten(),
        "probs_mean_high": (probs[gene_oi].mean(0) > 0.0).flatten(),
        "prob_high_any": (probs[gene_oi] > prob_cutoff).any(0).flatten(),
    }
)
# locusscores = locusscores.query("rho_high_any").sample(100000)
# locusscores = locusscores.query("probs_mean_high").sample(100000)

# %%
fig, ax = plt.subplots()
locusscores_oi = locusscores
ax.scatter(locusscores_oi["probs_mean"], locusscores_oi["probs_diff_max"])
for bin in probs_mean_bins:
    ax.axvline(bin)

# %%
probs_mean_bins = np.arange(-2, 4)

# %%
locusscores["probs_mean_bin"] = pd.Categorical(
    np.digitize(locusscores["probs_mean"], probs_mean_bins)
)

# %%
bins = locusscores["probs_mean_bin"].cat.categories

# %%
bin_info = pd.DataFrame(
    {
        "bin": bins,
    }
).set_index("bin")
norm = mpl.colors.Normalize(bins.min() - 1, bins.max())
cmap = mpl.cm.YlOrRd
bin_info["color"] = cmap(norm(bin_info.index)).tolist()

# %%
metric = "probs_diff_max"
metric_label = "Max log-fold difference in accessibility compared to mean"
# metric = "probs_std"; metric_label = "Std in accessibility"

# %%
fig, axes = plt.subplots(
    len(bins),
    1,
    figsize=(5, len(bins) / 3),
    sharex=True,
    sharey=False,
    gridspec_kw={"hspace": 0},
)
for i, (bin, locusscores_bin) in enumerate(locusscores.groupby("probs_mean_bin")):
    ax = axes[i]
    sns.histplot(
        locusscores_bin[metric],
        ax=ax,
        stat="density",
        bins=np.linspace(0.0, 4.0, 50),
        color=bin_info.loc[bin, "color"],
        lw=0,
    )
    ax.set_ylim(0, 4)
    # ax.set_xlim(0, 1)
    ax.axvline(locusscores_bin[metric].median(), color="red")
    ax.set_yticks([2])
    ax.set_yticklabels([bin])
    ax.set_ylabel("")
# axes[-1].set_ylabel("density")
axes[-1].set_xlabel(metric_label)
axes[int(len(axes) / 2)].set_ylabel(
    "Mean\naccessibility", ha="right", va="center", rotation=0
)

# %%
fig, ax = plt.subplots(figsize=(4, 4))
norm = mpl.colors.Normalize(bins.min(), bins.max())
cmap = mpl.cm.YlOrRd
ax.set_xlim(0.0, 1.0)
for i, (bin, locusscores_bin) in enumerate(locusscores.groupby("probs_mean_bin")):
    sns.ecdfplot(locusscores_bin[metric], ax=ax, color=cmap(norm(bin)))

# %% [markdown]
# ## Enrichment

# %%
# mixture_interpolated = probs_interpolated - probs_interpolated.mean(-1, keepdims = True)
mixture_interpolated = probs_interpolated

# %%
# cluster_oi = "Lymphoma"
# cluster_oi = "pDCs"
# cluster_oi = "Monocytes"
# cluster_oi = "cDCs"
cluster_oi = "CD4 T"
# cluster_oi = "leiden_0"
# cluster_oi = "CD8 T"
# cluster_oi = "B"
cluster_ix = cluster_info.index.tolist().index(cluster_oi)

# %%
# k = 100
# probs_mean_pooled = (
#     torch.nn.functional.max_pool1d(
#         torch.from_numpy(probs_interpolated), k, stride=1, padding=(k // 2,)
#     )[..., :-1]
#     .mean(1)
#     .numpy()
# )

probs_mean = probs_interpolated.mean(1)

clusterprobs_diff = mixture_interpolated[:, cluster_ix] - probs_mean
# clusterprobs_diff = mixture_interpolated[:, cluster_ix] - probs_mean

# %%
n_cuts = 10
q = np.linspace(0, 1, n_cuts + 1)[1:-1]

# %%
probs_mean_cuts = np.quantile(probs_mean.flatten(), q)
clusterprobs_diff_cuts = np.quantile(clusterprobs_diff.flatten(), q)

# probs_mean_cuts = np.linspace(
#     np.quantile(probs_mean.flatten(), 0.01),
#     np.quantile(probs_mean.flatten(), 0.99),
#     n_cuts - 1,
# )
# probs_mean_cuts = np.linspace(1, 3, n_cuts-1)
# probs_mean_cuts = np.linspace(5, 12, n_cuts-1)
probs_mean_cuts = np.log(np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]))


clusterprobs_diff_cuts = np.linspace(
    np.quantile(clusterprobs_diff.flatten(), 0.01),
    np.quantile(clusterprobs_diff.flatten(), 0.999),
    n_cuts - 1,
)
# clusterprobs_diff_cuts = np.linspace(-np.log(4), np.log(4), n_cuts-1)

# %%
bin = np.digitize(probs_mean.flatten(), probs_mean_cuts, right=True) + (
    np.digitize(clusterprobs_diff.flatten(), clusterprobs_diff_cuts, right=True)
) * (n_cuts)

# %%
ix = np.random.choice(np.prod(clusterprobs_diff.shape), 100000)
fig, ax = plt.subplots()
ax.scatter(
    np.exp(probs_mean.flatten()[ix]),
    np.exp(clusterprobs_diff.flatten()[ix]),
    c=bin[ix],
    s=1,
)
ax.set_xscale("log")
ax.set_yscale("log")
for cut in probs_mean_cuts:
    ax.axvline(np.exp(cut), color="#33333333")
for cut in clusterprobs_diff_cuts:
    ax.axhline(np.exp(cut), color="#33333333")

# %%
bins_bg = np.reshape(np.bincount(bin, minlength=n_cuts**2), (n_cuts, n_cuts))

# %%
fig, ax = plt.subplots()
norm = mpl.colors.LogNorm()
sns.heatmap(bins_bg, ax=ax, norm=norm)
# ax.invert_xaxis()
ax.invert_yaxis()

# %%
motifscan_name = "cutoff_0001"
# motifscan_name = "onek1k_0.2"
# motifscan_name = "gwas_immune"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)

# %%
# motif_ix_oi = motifscan.motifs.index.tolist().index(
#     motifscan.motifs.loc[motifscan.motifs.index.str.contains("SPI1")].index[0]
# )
# motif_ix_oi = motifscan.motifs.index.tolist().index(
#     motifscan.motifs.loc[motifscan.motifs.index.str.contains("TCF7")].index[0]
# )
# motif_ix_oi = motifscan.motifs.index.tolist().index(motifscan.motifs.loc[motifscan.motifs.index.str.contains("IRF4")].index[0])
motif_ix_oi = motifscan.motifs.index.tolist().index(
    motifscan.motifs.loc[motifscan.motifs.index.str.contains("RUNX3")].index[0]
)
# motif_ix_oi = motifscan.motifs.index.tolist().index(motifscan.motifs.loc[motifscan.motifs.index.str.contains("dc")].index[0])
# motif_ix_oi = motifscan.motifs.index.tolist().index(motifscan.motifs.loc[motifscan.motifs.index.str.contains("Rheumatoid arthritis")].index[0])

# %%
position_indices = np.repeat(
    np.arange(fragments.n_genes * (window[1] - window[0])), np.diff(motifscan.indptr)
)[(motifscan.indices == motif_ix_oi)]
position_chosen = np.zeros(fragments.n_genes * (window[1] - window[0]), dtype=bool)
position_chosen[position_indices] = True

# %%
bins_oi = np.reshape(
    np.bincount(bin[position_indices], minlength=n_cuts**2), (n_cuts, n_cuts)
)

# %%
odds = bins_oi / bins_bg

# %%
fig, ax = plt.subplots()
norm = mpl.colors.LogNorm()
sns.heatmap(bins_bg / bins_bg.sum(1, keepdims=True), ax=ax, norm=norm)
# ax.invert_xaxis()
ax.invert_yaxis()

# %%
fig, ax = plt.subplots()
sns.heatmap(odds, ax=ax)
ax.invert_yaxis()

# %%
fig, ax = plt.subplots()
norm = mpl.colors.LogNorm()
sns.heatmap(bins_oi, ax=ax, norm=norm)
ax.invert_yaxis()

# %% [markdown]
# ### Enrichment overall

# %%
motifs = motifscan.motifs

# %%
motifclustermapping = pd.DataFrame(
    [
        [motifs.loc[motifs.index.str.contains("TCF7")].index[0], ["CD4 T", "CD8 T"]],
        [motifs.loc[motifs.index.str.contains("IRF8")].index[0], ["cDCs"]],
        [motifs.loc[motifs.index.str.contains("IRF4")].index[0], ["cDCs"]],
        [
            motifs.loc[motifs.index.str.contains("CEBPB")].index[0],
            ["Monocytes", "cDCs"],
        ],
        [motifs.loc[motifs.index.str.contains("GATA4")].index[0], ["CD4 T"]],
        [
            motifs.loc[motifs.index.str.contains("HNF6_HUMAN.H11MO.0.B")].index[0],
            ["CD4 T"],
        ],
        [motifs.loc[motifs.index.str.contains("RARA")].index[0], ["MAIT"]],
        [motifs.loc[motifs.index.str.contains("PEBB")].index[0], ["NK"]],
        [motifs.loc[motifs.index.str.contains("RUNX2")].index[0], ["NK"]],
        [motifs.loc[motifs.index.str.contains("RUNX1")].index[0], ["CD8 T"]],
        [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["CD8 T"]],
        [
            motifs.loc[motifs.index.str.contains("TBX21_HUMAN.H11MO.0.A")].index[0],
            ["NK"],
        ],
        [
            motifs.loc[motifs.index.str.contains("SPI1")].index[0],
            ["Monocytes", "B", "cDCs"],
        ],
        [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["B"]],
        [motifs.loc[motifs.index.str.contains("NFKB2")].index[0], ["B"]],
        [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["B", "pDCs"]],  # TCF3
        [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],
        [motifs.loc[motifs.index.str.contains("FOS")].index[0], ["cDCs"]],
        [motifs.loc[motifs.index.str.contains("IRF1")].index[0], ["cDCs"]],
    ],
    columns=["motif", "clusters"],
).set_index("motif")
motifclustermapping = (
    motifclustermapping.explode("clusters")
    .rename(columns={"clusters": "cluster"})
    .reset_index()[["cluster", "motif"]]
)

# %%
clusterprobs_diff = mixture_interpolated - probs_mean[:, None]

# %%
n_cuts = 10
q = np.linspace(0, 1, n_cuts + 1)[1:-1]

# %%
# probs_mean_cuts = np.linspace(
#     np.quantile(probs_mean.flatten(), 0.01),
#     np.quantile(probs_mean.flatten(), 0.99),
#     n_cuts - 1,
# )
probs_mean_bins = pd.DataFrame(
    {"cut": np.log(np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]))}
)
labels = [
    f"< {round(i, 2)}"
    for i in (
        np.exp(probs_mean_bins["cut"].values[[0]]) / (window[1] - window[0]) * 100 * 100
    )
] + [
    f"≥ {round(i, 2)}"
    for i in (
        np.exp(probs_mean_bins["cut"].values[:-1]) / (window[1] - window[0]) * 100 * 100
    )
]
probs_mean_bins["label"] = labels

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": np.log(np.array([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, np.inf]))}
)
clusterprobs_diff_bins["label"] = ["<⅛", "⅛-¼", "¼-½", "½-1", "1-2", "2-4", "4-8", ">8"]
clusterprobs_diff_bins["label"] = ["<⅛", ">⅛", ">¼", ">½", ">1", ">2", ">4", ">8"]
clusterprobs_diff_bins["label"] = ["", "⅛", "¼", "½", "1", "2", "4", "8"]

# clusterprobs_diff_cuts = np.linspace(
#     np.quantile(clusterprobs_diff.flatten(), 0.01),
#     np.quantile(clusterprobs_diff.flatten(), 0.999),
#     n_cuts - 1,
# )

# %%
bin = np.digitize(probs_mean[:, None, :], probs_mean_bins["cut"], right=True) + (
    np.digitize(clusterprobs_diff, clusterprobs_diff_bins["cut"], right=True)
) * (len(probs_mean_bins))

# %%
n = np.bincount(
    bin.flatten(), minlength=len(probs_mean_bins) * len(clusterprobs_diff_bins)
)

# %%
sns.heatmap(np.reshape(np.log(n), (len(probs_mean_bins), len(clusterprobs_diff_bins))))

# %%
odds_cluster_motifs = []
contingencies_cluster_motifs = []
for _, (cluster, motif) in tqdm.tqdm(
    motifclustermapping[["cluster", "motif"]].iterrows()
):
    cluster_ix = cluster_info.loc[cluster, "dimension"]
    motif_ix_oi = motifs.index.tolist().index(motif)

    position_indices = np.repeat(
        np.arange(fragments.n_genes * (window[1] - window[0])),
        np.diff(motifscan.indptr),
    )[(motifscan.indices == motif_ix_oi)]
    position_chosen = np.zeros(fragments.n_genes * (window[1] - window[0]), dtype=bool)
    position_chosen[position_indices] = True

    bin_oi = bin[:, cluster_ix]
    bins_bg = np.reshape(
        np.bincount(
            bin_oi.flatten(),
            minlength=len(probs_mean_bins) * len(clusterprobs_diff_bins),
        ),
        (len(probs_mean_bins), len(clusterprobs_diff_bins)),
    )
    bins_oi = np.reshape(
        np.bincount(
            bin_oi.flatten()[position_indices],
            minlength=len(probs_mean_bins) * len(clusterprobs_diff_bins),
        ),
        (len(probs_mean_bins), len(clusterprobs_diff_bins)),
    )
    odds = (bins_oi / bins_bg) / (position_chosen.sum() / len(bin_oi.flatten()))
    odds_cluster_motifs.append(odds)

    cont = np.stack(
        [
            bins_oi,
            bins_bg,
            np.ones_like(bins_oi) * position_chosen.sum(),
            np.ones_like(bins_oi) * len(bin_oi.flatten()),
        ],
    )
    contingencies_cluster_motifs.append(cont)

# %% [markdown]
# ### Common contingency

# %%
weights = np.array([cont[0].sum() for cont in contingencies_cluster_motifs])
weights = weights / weights.sum()
weights[:] = 1.0
contingencies = np.stack(contingencies_cluster_motifs)
contingency = np.sum(contingencies * weights[:, None, None, None], 0)
odds = (contingency[0] / contingency[1]) / (contingency[2] / contingency[3])
odds[np.isnan(odds)] = 1.0

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))
cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=np.log(0.125), vmax=np.log(8.0))
ax.matshow(np.log(odds).T, cmap=cmap, norm=norm)

# no y-ticks for figure 4
# ax.set_ylabel("Mean accessibility (??)")
# ax.set_yticks(np.arange(len(probs_mean_bins)))
# ax.set_yticklabels(probs_mean_bins["label"])
ax.set_yticks([])
ax.tick_params(axis="y", length=5, which="minor")
ax.set_yticks([-0.5] + list(np.arange(len(clusterprobs_diff_bins)) + 0.5), minor=True)

ax.tick_params(
    axis="x", rotation=0, bottom=True, top=False, labelbottom=True, labeltop=False
)
ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)) - 0.5)
ax.set_xticklabels(clusterprobs_diff_bins["label"])

manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment")

# %%
fig_colorbar = plt.figure(figsize=(0.1, 1.5))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="vertical")
colorbar.set_label("Odds motif enrichment")
colorbar.set_ticks(np.log([0.125, 1, 8]))
colorbar.set_ticklabels(["⅛", "1", "8"])
manuscript.save_figure(fig_colorbar, "4", "colorbar_odds")

# %%
fig = chd.grid.Figure(chd.grid.Wrap(padding_width=0.1, padding_height=0.0))

cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=np.log(0.125), vmax=np.log(8.0))

for (_, (cluster, motif)), odds in zip(
    motifclustermapping[["cluster", "motif"]].iterrows(), odds_cluster_motifs
):
    panel, ax = fig.main.add(chd.grid.Panel((0.8, 0.8)))
    ax.matshow(np.log(odds).T, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    if motifscan.motifs.loc[motif]["gene"] in transcriptome.var.index:
        symbol = transcriptome.symbol(motifscan.motifs.loc[motif]["gene"])
    else:
        symbol = motifscan.motifs.loc[motif]["gene_label"]

    ax.set_title(f"{cluster} {symbol}", fontsize=8)

# set ticks for bottom left
panel, ax = fig.main.get_bottom_left_corner()
ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.tick_params(
    axis="x", rotation=0, bottom=True, top=False, labelbottom=True, labeltop=False
)
ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)) - 0.5)
ax.set_xticklabels(clusterprobs_diff_bins["label"])

fig.plot()

manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment_individual")


# %% [markdown]
# ### Individual contingencies mean

# %%
odds = np.stack(odds_cluster_motifs)
odds[np.isnan(odds)] = 1.0
odds[odds == (-np.inf)] = 1.0
odds[odds == (np.inf)] = 1.0
odds[odds == 0] = 1.0

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))
cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=np.log(0.25), vmax=np.log(4.0))
ax.matshow(np.log(odds).mean(0).T, cmap=cmap, norm=norm)
ax.set_ylabel("Mean accessibility (??)")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])

# manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment")
