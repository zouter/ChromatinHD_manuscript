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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rank gene-motif correspondence

# This notebook

# %%
import IPython

if IPython.get_ipython():
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
    IPython.get_ipython().run_line_magic(
        "config", "InlineBackend.figure_format='retina'"
    )

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %% [markdown]
# ## Data

# %% [markdown]
# ### Dataset

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "brain"
# dataset_name = "alzheimer"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
# transcriptome = None
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
fragments.create_cut_data()

# %%
fragments.obs["lib"] = torch.bincount(
    fragments.cut_local_cell_ix, minlength=fragments.n_cells
).numpy()

# %% [markdown]
# ### Latent space

# %%
# loading
# latent_name = "leiden_1"
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

fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])
if transcriptome is not None:
    transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = fragments.obs[
        "cluster"
    ] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# ### Prediction

# %%
method_name = "v9_128-64-32"


class Prediction(chd.flow.Flow):
    pass


prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_likelihood"
    / dataset_name
    / promoter_name
    / latent_name
    / method_name
)

# %%
# create base pair ranking
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

probs_diff = probs - probs.mean(1, keepdims=True)
design["gene_ix"] = design["gene_ix"]
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
desired_x = torch.arange(*window)
probs_interpolated = chd.utils.interpolate_1d(
    desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
).numpy()

prob_cutoff = np.log(1.0)

basepair_ranking = probs_interpolated - probs_interpolated.mean(1, keepdims=True)
basepair_ranking[probs_interpolated < prob_cutoff] = -np.inf

# %% [markdown]
# ### Motifscan

# %%
motifscan_name = "cutoff_0001"
# motifscan_name = "cutoff_001"
# motifscan_name = "onek1k_0.2"
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_cns"
# motifscan_name = "gtex_immune"

# %%
motifscan_folder = (
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)
motifscan = chd.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
motifscan.n_motifs = len(motifs)
motifs["ix"] = np.arange(motifs.shape[0])

# %%
# count number of motifs per gene
promoters["n"] = (
    np.diff(motifscan.indptr).reshape((len(promoters), (window[1] - window[0]))).sum(1)
)

# %% [markdown]
# ## Extract motif data

# %%
# motifs_oi = pd.DataFrame([], columns = ["motif", "clusters"]).set_index("motif")

motifs_oi = pd.DataFrame(
    [
        # [motifs.loc[motifs.index.str.contains("SPI1")].index[0], ["Monocytes"]],
        # [motifs.loc[motifs.index.str.contains("TCF7")].index[0], ["T"]],
        # [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["T"]],
        # [motifs.loc[motifs.index.str.contains("SNAI1")].index[0], ["Lymphoma"]],
        # [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], ["Lymphoma"]],
        # [motifs.loc[motifs.index.str.contains("PO5F1")].index[0], ["Lymphoma"]],
        # [motifs.loc[motifs.index.str.contains("NFKB")].index[0], ["Lymphoma"]],
        # [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["Lymphoma"]],
        # [motifs.loc[motifs.index.str.contains("Chronic lymphocytic leukemia")].index[0], ["Lymphoma"]],
        # [motifs.loc[motifs.index.str.contains("TFE2")].index[0], ["pDCs"]],
        # [motifs.loc[motifs.index.str.contains("BHA15")].index[0], ["pDCs"]],
        # [motifs.loc[motifs.index.str.contains("BC11A")].index[0], ["pDCs"]],
        [motifs.loc[motifs.index.str.contains("IRF1")].index[0], ["cDCs"]],
    ],
    columns=["motif", "clusters"],
).set_index("motif")
motifs_oi["target"] = motifs_oi.index.str.split("_").str[0]

# %%
basepair_ranking_zerod = basepair_ranking.copy()
basepair_ranking_zerod[basepair_ranking_zerod < 0] = 0

basepair_ranking_zerod = basepair_ranking.copy()
basepair_ranking_zerod[basepair_ranking_zerod < 0] = 0

# %%
motifdata = []
scores = []
for gene_ix, gene in tqdm.tqdm(enumerate(promoters.index)):
    indptr_start = gene_ix * (window[1] - window[0])
    indptr_end = (gene_ix + 1) * (window[1] - window[0])

    motif_indices = motifscan.indices[
        motifscan.indptr[indptr_start] : motifscan.indptr[indptr_end]
    ]
    position_indices = chd.utils.indptr_to_indices(
        motifscan.indptr[indptr_start : indptr_end + 1]
    )

    assert len(motif_indices) == len(position_indices)

    for motif in motifs_oi.index:
        motif_ix = motifs.loc[motif, "ix"]
        clusters_ixs = cluster_info.loc[motifs_oi.loc[motif, "clusters"], "dimension"]
        positions_oi = position_indices[motif_indices == motif_ix]

        ranking_gene = basepair_ranking_zerod[gene_ix, clusters_ixs[0]]
        n_differential_positions = (ranking_gene > np.log(1.5)).sum()
        if len(positions_oi) > 0:
            area = chd.utils.ecdf.relative_area_between_ecdfs(
                ranking_gene, ranking_gene[positions_oi]
            )
        else:
            area = 0.0
        scores.append(
            [gene_ix, motif, area, n_differential_positions, len(positions_oi)]
        )
scores = pd.DataFrame(
    scores, columns=["gene_ix", "motif", "area", "n_differential_positions", "n_hits"]
).set_index(["gene_ix", "motif"])
scores["symbol"] = transcriptome.var.iloc[scores.index.get_level_values("gene_ix")][
    "symbol"
].values


# %%
(
    scores.sort_values("area", ascending=False)
    .query("n_differential_positions > 1000")
    .query("n_hits > 3")
).head(20)

# %%
sc.pl.umap(transcriptome.adata, color=["cluster", transcriptome.gene_id("CD86")])

# %%
# scores.query("symbol == 'SIAH2'")
# scores.query("symbol == 'BCL2'")

# %%
celltype_oi = "cDCs"
sc.tl.rank_genes_groups(transcriptome.adata, "cluster")
sc.get.rank_genes_groups_df(transcriptome.adata, celltype_oi).assign(
    symbol=lambda x: transcriptome.symbol(x["names"]).values
).set_index("symbol")
# %%
scores["lfc"] = (
    sc.get.rank_genes_groups_df(transcriptome.adata, celltype_oi)
    .assign(symbol=lambda x: transcriptome.symbol(x["names"]).values)
    .set_index("symbol")
    .reindex(scores["symbol"])["logfoldchanges"]
    .values
)

# %%
scores.groupby(["gene_ix", "symbol"]).min().sort_values("area", ascending=False).query(
    "n_hits > 2.0"
).query("n_differential_positions > 3000").query("lfc > 1").head(20).style.bar()

# %% [markdown]
# ## Individual target gene

# %%
gene_oi = transcriptome.gene_id("IL4R")
gene_ix = transcriptome.var.index.get_loc(gene_oi)

# cluster_oi = "Lymphoma"
cluster_oi = "Lymphoma cycling"
cluster_ix = cluster_info.loc[cluster_oi, "dimension"]

motifs_oi = [
    motifs.loc[motifs.index.str.contains("SNAI1")].index[0],
    # motifs.loc[motifs.index.str.contains("PO2F2")].index[0],
    # motifs.loc[motifs.index.str.contains("PO5F1")].index[0],
    # motifs.loc[motifs.index.str.contains("NFKB")].index[0],
    # motifs.loc[motifs.index.str.contains("TFE2")].index[0],
]
motif_ixs = motifs.loc[motifs_oi, "ix"]


# %%
indptr_start = gene_ix * (window[1] - window[0])
indptr_end = (gene_ix + 1) * (window[1] - window[0])

motif_indices = motifscan.indices[
    motifscan.indptr[indptr_start] : motifscan.indptr[indptr_end]
]
position_indices = chd.utils.indptr_to_indices(
    motifscan.indptr[indptr_start : indptr_end + 1]
)

positions_oi = position_indices[np.isin(motif_indices, motif_ixs)]

sns.ecdfplot(basepair_ranking_zerod[gene_ix, cluster_ix])
sns.ecdfplot(basepair_ranking_zerod[gene_ix, cluster_ix, positions_oi])

chd.utils.ecdf.relative_area_between_ecdfs(
    basepair_ranking_zerod[gene_ix, cluster_ix],
    basepair_ranking_zerod[gene_ix, cluster_ix, positions_oi],
)

# %%
peak_scores_dir = (
    chd.get_output()
    / "prediction_differential"
    / dataset_name
    / promoter_name
    / latent_name
    / "scanpy"
    # / "macs2_improved"
    / "encode_screen"
    # / "macs2_leiden_0.1_merged"
)

peakresult = pickle.load((peak_scores_dir / "slices.pkl").open("rb"))

basepair_ranking_peak = peakresult.position_ranked.reshape(
    (
        peakresult.n_genes,
        peakresult.n_clusters,
        peakresult.window[1] - peakresult.window[0],
    )
)

# %%
basepair_ranking_zerod_gene = basepair_ranking_zerod[gene_ix, cluster_ix]
basepair_ranking_zerod_gene_oi = basepair_ranking_zerod_gene[positions_oi]

basepair_ranking_peak_gene = basepair_ranking_peak[gene_ix, cluster_ix]
basepair_ranking_peak_gene_oi = basepair_ranking_peak_gene[positions_oi]

# %%
fig, ax = plt.subplots()
sns.ecdfplot(basepair_ranking_zerod_gene)
sns.ecdfplot(basepair_ranking_zerod_gene_oi)

fig, ax = plt.subplots()
sns.ecdfplot(basepair_ranking_peak_gene)
sns.ecdfplot(basepair_ranking_peak_gene_oi)

# %%
(
    chd.utils.ecdf.relative_area_between_ecdfs(
        basepair_ranking_zerod_gene, basepair_ranking_zerod_gene_oi
    ),
    chd.utils.ecdf.relative_area_between_ecdfs(
        basepair_ranking_peak_gene, basepair_ranking_peak_gene_oi
    ),
)

# %%
random = []
for i in range(1000):
    random.append(
        chd.utils.ecdf.area_between_ecdfs(
            basepair_ranking_zerod_gene,
            np.random.choice(
                basepair_ranking_zerod_gene, len(positions_oi), replace=True
            ),
        )
    )
random = np.array(random)
fig, ax = plt.subplots()
sns.ecdfplot(random)
ax.axvline(
    chd.utils.ecdf.area_between_ecdfs(
        basepair_ranking_zerod_gene, basepair_ranking_zerod_gene_oi
    ),
    color="red",
)
print(
    (
        random
        >= chd.utils.ecdf.area_between_ecdfs(
            basepair_ranking_zerod_gene, basepair_ranking_zerod_gene_oi
        )
    ).mean()
)


# %%
random = []
for i in range(1000):
    random.append(
        chd.utils.ecdf.area_between_ecdfs(
            basepair_ranking_peak[gene_ix, cluster_ix],
            np.random.choice(
                basepair_ranking_peak[gene_ix, cluster_ix],
                len(positions_oi),
                replace=True,
            ),
        )
    )
random = np.array(random)
fig, ax = plt.subplots()
sns.ecdfplot(random)
ax.axvline(
    chd.utils.ecdf.area_between_ecdfs(
        basepair_ranking_peak[gene_ix, cluster_ix],
        basepair_ranking_peak[gene_ix, cluster_ix, positions_oi],
    ),
    color="red",
)
print(
    (
        random
        >= chd.utils.ecdf.area_between_ecdfs(
            basepair_ranking_peak[gene_ix, cluster_ix],
            basepair_ranking_peak[gene_ix, cluster_ix, positions_oi],
        )
    ).mean()
)


# %%
