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
# motifscan_name = "cutoff_0001"
# motifscan_name = "cutoff_001"
# motifscan_name = "onek1k_0.2"
motifscan_name = "gwas_immune"
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
# %%
motifscan_name = "gwas_immune"

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on="snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

# %%
associations_genes = {}
for promoter in tqdm.tqdm(promoters.itertuples(), total=len(promoters)):
    association_oi = association.loc[
        (association["chr"] == promoter.chr)
        & (association["pos"] >= promoter.start)
        & (association["pos"] < promoter.end)
    ].copy()

    association_oi["position"] = (
        association_oi["pos"] - promoter.tss
    ) * promoter.strand

    associations_genes[promoter.Index] = association_oi


# %% [markdown]
# ### Relative to peaks

# %%
import chromatinhd.peakcounts

# peakcaller = "cellranger"
# peakcaller = "macs2"
# peakcaller = "macs2_improved";peakcaller_label = "MACS2"
# peakcaller = "encode_screen";peakcaller_label = "ENCODE SCREEN"

peakcaller = "macs2_leiden_0.1_merged"

# peakcaller = "rolling_500";peakcaller_label = "Sliding window 500"

# peakcaller = "rolling_100"

# diffexp = "signac"
diffexp = "scanpy"
# diffexp = "scanpy_wilcoxon"

# %%
scores_dir = (
    chd.get_output()
    / "prediction_differential"
    / dataset_name
    / promoter_name
    / latent_name
    / diffexp
    / peakcaller
)
peakresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %%
peakresult.get_slicescores()

# %%
scores_dir = prediction.path / "scoring" / peakcaller / diffexp
regionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %%
region_position_chosen = regionresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), (window[1] - window[0]))
)
peak_position_chosen = peakresult.position_chosen.reshape(
    (fragments.n_genes, len(cluster_info), (window[1] - window[0]))
)
# %%
clusters_oi = cluster_info["dimension"].values
# clusters_oi = cluster_info.query("cluster == 'B'")["dimension"].values
clusters_oi = cluster_info.query("cluster == 'Lymphoma'")["dimension"].values

genescores = []
for gene, association_oi in tqdm.tqdm(associations_genes.items()):
    gene_ix = promoters.index.get_loc(gene)

    association_oi["peak"] = peak_position_chosen[
        gene_ix, :, association_oi["position"] - window[0] - 1
    ][:, clusters_oi].any(1)

    association_oi["region"] = region_position_chosen[
        gene_ix, :, association_oi["position"] - window[0] - 1
    ][:, clusters_oi].any(1)

    # association_oi = association_oi.loc[
    #     association_oi["disease/trait"].str.contains("leuke")
    # ]

    genescores.append(
        {
            "gene": gene,
            "symbol": transcriptome.symbol(gene),
            "peak": association_oi["peak"].sum(),
            "region": association_oi["region"].sum(),
            "n": association_oi.shape[0],
        }
    )
genescores = pd.DataFrame(genescores)
# %%
genescores.query("n < 10").sort_values("region", ascending=False).head(30)
genescores.query("n == 2").query("peak <region").sort_values(
    "region", ascending=False
).head(30)
# %%
genescores["peak"].mean(), genescores["region"].mean()

# %%
symbol = "NFKB2"
gene_id = transcriptome.gene_id(symbol)
gene_ix = transcriptome.gene_ix(symbol)

fig, ax = plt.subplots()
ax.plot(peak_position_chosen[gene_ix].any(0))
ax.plot(region_position_chosen[gene_ix].any(0))

association_oi = associations_genes[gene_id]
ax.scatter(
    association_oi["position"] - window[0],
    [1] * association_oi.shape[0],
    s=10,
    alpha=1,
    color="red",
    zorder=10,
)

# %%
