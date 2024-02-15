# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    default = "pbmc10k",
    # default = "GSE198467_H3K27ac",
    # default = "brain",
    # default = "pbmc10k"
)
parser.add_argument("--promoter_name", default = "10k10k")
# parser.add_argument("--latent_name", default = "celltype")
parser.add_argument("--latent_name", default = "leiden_0.1")
parser.add_argument("--method_name", default = 'v9_128-64-32')

try:
    get_ipython().__class__.__name__
    in_jupyter = True
except:
    in_jupyter = False
globals().update(vars(parser.parse_args("" if in_jupyter else None)))

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / dataset_name


# %% [markdown]
# ### Load data

# %%
class Prediction(chd.flow.Flow):
    pass
prediction = Prediction(chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / method_name)
# model = chd.load((prediction.path / "model_0.pkl").open("rb"))

# %%
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

probs_diff = probs - probs.mean(1, keepdims=True)

# %%
design["gene_ix"] = design["gene_ix"]

# %%
window = {
    "10k10k":np.array([-10000, 10000])
}[promoter_name]
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# Interpolate probs for individual positions

# %%
x = (design["coord"].values).astype(int).reshape((len(design["gene_ix"].cat.categories), len(design["active_latent"].cat.categories), len(design["coord"].cat.categories)))
desired_x = torch.arange(*window)
probs_interpolated = chd.utils.interpolate_1d(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)).numpy()

# %%
prob_cutoff = np.log(1.0)
# prob_cutoff = np.log(1e-5)

# basepair_ranking = probs_interpolated
basepair_ranking = probs_interpolated - probs_interpolated.mean(
    1, keepdims=True
)
basepair_ranking[probs_interpolated < prob_cutoff] = -np.inf

# %%
import scipy

# %%
basepair_ranking_flattened = basepair_ranking.transpose(0, 2, 1).flatten().reshape(-1, basepair_ranking.shape[1])
basepair_ranking_flattened[np.isinf(basepair_ranking_flattened)] = 0.
basepair_ranking_flattened[basepair_ranking_flattened < 0] = 0.
# basepair_ranked_flattened = scipy.stats.rankdata(basepair_ranking_flattened, axis = 0)

# %% [markdown]
# ## QTLs

# %%
qtlscan_name = "gwas_immune"
# motifscan_name = "gtex_immune"
# motifscan_name = "onek1k_0.2"

# %%
qtlscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name / qtlscan_name
qtlscan = chd.data.Motifscan(qtlscan_folder)

# %%
motifscan_name = "cutoff_001"

# %%
window_size = window[1] - window[0]

# %%
gene_ix = transcriptome.gene_ix("IL1B")

# %%
# plt.plot(basepair_ranked_flattened[:, 0][(gene_ix * window_size):((gene_ix+1) * window_size)])

# %%
qtlscan.indices_position = chd.utils.numpy.indptr_to_indices(qtlscan.indptr)

# %%
qtlscan.motifs["ix"] = np.arange(len(qtlscan.motifs))

# %%
# motif_ix = qtlscan.motifs.loc["Hodgkin's lymphoma", "ix"]
# motif_ix = qtlscan.motifs.loc["Interleukin-6 levels", "ix"]
motif_ix = qtlscan.motifs.loc["Systemic lupus erythematosus", "ix"]
# motif_ix = qtlscan.motifs.loc["Chronic lymphocytic leukemia", "ix"]
# motif_ix = qtlscan.motifs.loc["Type 1 diabetes", "ix"]
# motif_ix = qtlscan.motifs.loc["Crohn's disease", "ix"]

# %%
# qtl_ranked = basepair_ranked_flattened[qtlscan.indices_position[qtlscan.indices == motif_ix]]

# %%
forbidden_genes = transcriptome.var["ix"][transcriptome.var["symbol"].str.startswith("HLA")]
forbidden_gene_selection = np.ones(fragments.n_genes, dtype = bool)
forbidden_gene_selection[forbidden_genes] = 0

forbidden_ix = np.ones(fragments.n_genes * window_size, dtype = bool)
for gene_ix in forbidden_genes:
    forbidden_ix[np.arange(gene_ix * window_size, (gene_ix + 1) * window_size)] = 0

# %%
qtlscan.motifs

# %%
qtl_positions = qtlscan.indices_position[qtlscan.indices == motif_ix]
qtl_selection = np.zeros(fragments.n_genes * window_size, dtype = bool)
qtl_selection[qtl_positions] = True
print(qtl_selection.sum())
qtl_selection[~forbidden_ix] = False
print(qtl_selection.sum())
qtl_ranked = basepair_ranking_flattened[qtl_selection]

qtl_mean_rank = qtl_ranked.mean(0)
basepair_mean_rank = basepair_ranking_flattened[forbidden_ix].mean(0)

# basepair_mean_rank = basepair_ranked_flattened.mean(0)

# %%
gene_ix = transcriptome.gene_ix("TYMP")
plt.plot(probs_interpolated[gene_ix, 4])
plt.plot(probs_interpolated[gene_ix, 0])

# %%
scores = pd.DataFrame(
    {"score":qtl_ranked[:, 4],
    "gene":pd.Series(transcriptome.var.index[qtl_positions[qtl_selection[qtl_positions]] // window_size])
    }
)
scores["symbol"] = transcriptome.symbol(scores["gene"]).values

grouped_scores = pd.DataFrame({
    "best_score":scores.sort_values("score", ascending = False).groupby("gene")["score"].max(),
    "scores":scores.groupby("gene").apply(lambda x:[x["score"].sort_values(ascending = False).round(2).astype(str)]),
    "n":scores.sort_values("score", ascending = False).groupby("gene").size(),
})
grouped_scores["symbol"] = transcriptome.symbol(grouped_scores.index).values
grouped_scores.sort_values("best_score", ascending = False)

# %%
# sns.histplot(basepair_ranked_flattened[:, 0][:100000], bins = 10, stat = "density")
# sns.histplot(qtl_ranked_oi[:, 0], bins = 10, color = "red", stat = "density")

# %%
(qtl_mean_rank - basepair_mean_rank) / (basepair_mean_rank*2)

# %%
pd.Series(qtl_mean_rank / basepair_mean_rank, index = cluster_info.index).plot(kind = "barh")

# %%
qtl_ranked = basepair_ranking_flattened[qtlscan.indices_position[qtlscan.indices == motif_ix]]

# %%
import torch_scatter

# %%
qtl_mean_rank_gene = torch_scatter.segment_mean_coo(torch.from_numpy(qtl_ranked), torch.from_numpy(qtl_positions // window_size), dim_size = fragments.n_genes)

# %%
basepair_mean_rank_gene = torch_scatter.segment_mean_coo(torch.from_numpy(basepair_ranking_flattened), torch.repeat_interleave(torch.arange(fragments.n_genes), window_size), dim_size = fragments.n_genes)

# %%
gene_ix = transcriptome.gene_ix("POU2AF1")

# %%
qtl_mean_rank_gene[gene_ix]

# %%
basepair_mean_rank_gene[gene_ix]

# %%
qtl_mean_rank_gene[

# %%
forbidden_gene_selection

# %%
(qtl_mean_rank_gene[forbidden_gene_selection] / (basepair_mean_rank_gene[forbidden_gene_selection]+1e-10)).mean(0)

# %%
qtl_mean_rank_gene.shape

# %%
(qtl_mean_rank_gene / (basepair_mean_rank_gene+1e-1)).mean(0)

# %%
qtl_mean_rank_gene.shape

# %%
forbidden_genes

# %%
np.isin(qtl_positions // window_size, forbidden_genes)

# %%
pd.Series(transcriptome.var.index[qtlscan.indices_position[qtlscan.indices == motif_ix] // window_size]).value_counts()

# %%

# %%

# %%
