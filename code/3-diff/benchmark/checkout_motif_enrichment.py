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
import peakfreeatac as pfa
import tempfile

# %% [markdown]
# ## Data

# %% [markdown]
# ### Dataset

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "brain"
# dataset_name = "alzheimer"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
fragments.create_cut_data()

# %%
torch.bincount(fragments.cut_local_cell_ix, minlength = fragments.n_cells).numpy().shape

# %%
fragments.obs["lib"] = torch.bincount(fragments.cut_local_cell_ix, minlength = fragments.n_cells).numpy()

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
transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %%
cluster_info

# %%
cluster_info["lib"] = fragments.obs.groupby("cluster")["lib"].sum().values

# %% [markdown]
# ### Prediction

# %%
method_name = 'v4_128-64-32_30_rep'
class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / method_name)

# %%
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
probs2 = pickle.load((prediction.path / "probs2.pkl").open("rb"))
mixtures = pickle.load((prediction.path / "mixtures.pkl").open("rb"))
rhos = pickle.load((prediction.path / "rhos.pkl").open("rb"))
rho_deltas = pickle.load((prediction.path / "rho_deltas.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

# %% [markdown]
# ### Motifscan

# %%
# motifscan_name = "cutoff_0001"
motifscan_name = "onek1k_0.2"
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
motifscan = pfa.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
motifscan.n_motifs = len(motifs)
motifs["ix"] = np.arange(motifs.shape[0])

# %%
# count number of motifs per gene
transcriptome.var["n"] = np.diff(motifscan.indptr).reshape((len(transcriptome.var), (window[1] - window[0]))).sum(1)

# %%
# distribution
plt.plot(np.diff(motifscan.indptr).reshape((len(transcriptome.var), (window[1] - window[0]))).sum(0))

# %%
scores_dir = (prediction.path / "scoring" / "significant_up" / motifscan_name)
motifscores = pd.read_pickle(scores_dir / "motifscores.pkl")
scores = pd.read_pickle(scores_dir / "scores.pkl")
print(scores["n_position"])

# %%
peaks_name = "cellranger"
# peaks_name = "macs2"
# peaks_name = "macs2_improved"
# peaks_name = "rolling_500"

# %%
scores_dir = (prediction.path / "scoring" / peaks_name / motifscan_name)
motifscores_all = pd.read_pickle(scores_dir / "motifscores_all.pkl")
scores = pd.read_pickle(scores_dir / "scores.pkl")
# genemotifscores_all = pd.read_pickle(scores_dir / "genemotifscores_all.pkl")
print(scores["n_position"])

# %%
x = motifscores_all.query("cluster == 'pDCs'").sort_values("logodds_peak", ascending = False)

# %%
motifscores_all.sort_values("logodds_region", ascending = False).head(10)

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(motifscores_all["in_peak"], motifscores_all["in_region"], c = motifscores_all["cluster"].astype("category").cat.codes)
ax.axline([0, 0], slope = 1)

# %%
(motifscores_all["in_region"] - motifscores_all["in_peak"]).mean()

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(motifscores_all["logodds_peak"], motifscores_all["logodds_region"])
ax.axline([0, 0], slope = 1)

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
# ax.scatter(
#     np.exp(motifscores_all["logodds_peak"]),
#     np.exp(motifscores["logodds"]),
#     s = 1
# )

motifscores_oi = motifscores_all
# motifscores_oi = motifscores_all.query("(qval_peak < 0.05) or (qval_region < 0.05)")
ax.scatter(
    np.exp(motifscores_oi["logodds_peak"]),
    np.exp(motifscores_oi["logodds_region"]),
    s = 1
)

ax.set_ylim(1/4, 4)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(1/4, 4)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["¼", "½", "1", "2", "4"])

for i, label in zip([1/2, 1/np.sqrt(2), np.sqrt(2), 2], ["½", r"$\frac{1}{\sqrt{2}}$", "$\sqrt{2}$", "2"]):
    intercept = 1
    slope = i
    ax.axline((1, slope * 1), (intercept*2, slope * 2), color = "grey", dashes = (1, 1))
    
    if i > 1:
        x = 4
        y = intercept + slope * i
        ax.text(x, y, label, fontsize = 8)
    # ax.text(np.sqrt(1/i), np.sqrt(i), label, fontsize = 8)
ax.axvline(1, color = "grey")
ax.axhline(1, color = "grey")
ax.set_xlabel("Odds-ratio differential peaks")
ax.set_ylabel("Odds-ratio\ndifferential\nChromatinHD\nregions", rotation = 0, va = "center", ha = "right")

linreg = scipy.stats.linregress(motifscores_oi["logodds_region"], motifscores_oi["logodds_peak"])
slope = linreg.slope
intercept = linreg.intercept
print(1/slope)

ax.axline((np.exp(0), np.exp(intercept)), (np.exp(1), np.exp(1/slope)), color = "orange")

# %%
import peakfreeatac.grid


# %%
def plot_motifscores(ax, motifscores_all):
    ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
    ax.scatter(
        np.exp(motifscores_all["logodds_region"]),
        np.exp(motifscores_all["logodds_peak"]),
        s = 1
    )

    ax.set_ylim(1/4, 4)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["¼", "½", "1", "2", "4"])

    ax.set_xlim(1/4, 4)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["¼", "½", "1", "2", "4"])

    for i, label in zip([1/2, 1/np.sqrt(2), np.sqrt(2), 2], ["½", r"$\frac{1}{\sqrt{2}}$", "$\sqrt{2}$", "2"]):
        intercept = 1
        slope = i
        ax.axline((1, slope * 1), (intercept*2, slope * 2), color = "grey", dashes = (1, 1))

        if i > 1:
            x = 4
            y = intercept + slope * i
            ax.text(x, y, label, fontsize = 8)
        # ax.text(np.sqrt(1/i), np.sqrt(i), label, fontsize = 8)
    ax.axvline(1, color = "grey")
    ax.axhline(1, color = "grey")
    # ax.set_xlabel("Odds-ratio differential peaks")
    # ax.set_ylabel("Odds-ratio\ndifferential\nChromatinHD\nregions", rotation = 0, va = "center", ha = "right")
    
main = pfa.grid.Wrap()
fig = pfa.grid.Figure(main)

for cluster in cluster_info.index:
    ax_ = main.add(pfa.grid.Ax((2, 2)))
    
    ax_.ax.set_title(cluster)
    
    motifscores_cluster = motifscores_all.query("cluster == @cluster")
    motifscores_cluster = motifscores_cluster.query("(qval_peak < 0.05) or (qval_region < 0.05)")
    plot_motifscores(ax_.ax, motifscores_cluster)
    
    if len(motifscores_cluster) > 1:
        import scipy.stats
        linreg = scipy.stats.linregress(motifscores_cluster["logodds_region"], motifscores_cluster["logodds_peak"])
        slope = linreg.slope
        intercept = linreg.intercept

        ax_.ax.axline((np.exp(0), np.exp(intercept)), (np.exp(1), np.exp(slope)), color = "orange")
    
        print(1/slope)
fig.plot()

# %%
motifscores_all.query("cluster == 'leiden_0'").sort_values("logodds_region")

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.set_aspect(1)
ax.axline([0, 0], slope = 1, color = "#333333", zorder = 0)
# ax.scatter(
#     np.exp(motifscores_all["logodds_peak"]),
#     np.exp(motifscores["logodds"]),
#     s = 1
# )
ax.scatter(
    np.exp(motifscores_all["logodds_peak"]),
    np.exp(motifscores_all["logodds_region"]),
    s = 1
)

ax.set_ylim(1/4, 4)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["¼", "½", "1", "2", "4"])

ax.set_xlim(1/4, 4)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["¼", "½", "1", "2", "4"])

for i, label in zip([1/2, 1/np.sqrt(2), np.sqrt(2), 2], ["½", r"$\frac{1}{\sqrt{2}}$", "$\sqrt{2}$", "2"]):
    intercept = 1
    slope = i
    ax.axline((1, slope * 1), (intercept*2, slope * 2), color = "grey", dashes = (1, 1))
    
    if i > 1:
        x = 4
        y = intercept + slope * i
        ax.text(x, y, label, fontsize = 8)
    # ax.text(np.sqrt(1/i), np.sqrt(i), label, fontsize = 8)
ax.axvline(1, color = "grey")
ax.axhline(1, color = "grey")
ax.set_xlabel("Odds-ratio differential peaks")
ax.set_ylabel("Odds-ratio\ndifferential\nChromatinHD\nregions", rotation = 0, va = "center", ha = "right")

# %%
print("same # of positions, all motifs odds:", np.exp((motifscores_all["logodds_region"] - motifscores_all["logodds_peak"]).mean()))
print("significant positions, all motifs odds:", np.exp((motifscores["logodds"] - motifscores_all["logodds_peak"]).mean()))
print("same # of positions, all motifs in region:", (motifscores_all["in_region"]).mean() / (motifscores_all["in_peak"]).mean())
print("significant positions, all motifs in region:", (motifscores["in"]).mean() / (motifscores_all["in_peak"]).mean())

# %%
motifscores_all["in_region"].plot(kind = "hist")
motifscores_all["in_peak"].plot(kind = "hist")

# %%
