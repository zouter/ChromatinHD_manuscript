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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %% [markdown]
# ## Get the dataset

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k_leiden_0.1"
folder_data_preproc = folder_data / dataset_name

# %%
fragments = chd.data.Fragments(folder_data_preproc / "fragments")

# %% [markdown]
# ## Creating data

# %% [markdown]
# ### Size

# %%
(
    5000000  # number of SNPs
    * 1000  # number of patients
    * 2  # homo/hetero/homo alt
    * 8  # bits
    / 1024
    / 1024
    / 1024  # gbs
)

# %%
outcome = torch.tensor([[0, 1, 0, 1, 1, 0]])

# %%
outcome

# %% [markdown]
# ## Create loaders

# %%
# which clusters to look at
# the loader will always select the same set of clusters
clusters_oi = np.array([0, 1, 2, 3])

# %%
# which positions (i.e. SNPs) to look at
positions_oi = np.array([10000])

# %%
# import chromatinhd.loaders.chunkfragments

# %%
import chromatinhd.loaders.fragments

# n_cells_step = 1000
# n_genes_step = 1000

n_cells_step = 100
n_genes_step = 50

# n_cells_step = 2000
# n_genes_step = 500

loaders_train = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments": fragments, "cellxgene_batch_size": n_cells_step * n_genes_step},
    n_workers=20,
    shuffle_on_iter=True,
)
minibatches_train = chd.loaders.minibatching.create_bins_random(
    cells_train,
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step=n_genes_step,
    n_cells_step=n_cells_step,
    use_all=True,
    permute_genes=False,
)
loaders_train.initialize(minibatches_train)

loaders_validation = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments": fragments, "cellxgene_batch_size": n_cells_step * n_genes_step},
    n_workers=5,
)
minibatches_validation = chd.loaders.minibatching.create_bins_random(
    cells_validation,
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step=n_genes_step,
    n_cells_step=n_cells_step,
    use_all=True,
    permute_genes=False,
)
loaders_validation.initialize(minibatches_validation)

# %% [markdown]
# ## Model

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
fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# ### Create model

# %%
reflatent = torch.eye(n_latent_dimensions).to(torch.float)
reflatent_idx = torch.from_numpy(np.where(latent.values)[1])

# %%
import chromatinhd.models.likelihood.v9 as vae_model

# model = vae_model.Decoding(fragments, torch.from_numpy(latent.values), nbins = (128, 64, 32, ))
model = vae_model.Decoding(fragments, torch.from_numpy(latent.values), nbins=(32,))

# model = pickle.load((chd.get_output() / "prediction_likelihood/GSE198467_H3K27ac/10k10k/leiden_0.1/v9_128-64-32/model_0.pkl").open("rb"))

# %% [markdown]
# ### Prior distribution

# %%
device = "cuda"
model = model.to(device)

# %%
fragments.create_cut_data()

# %%
gene_oi = 0
# gene_oi = promoters["symbol"].tolist().index("Neurod1")
# gene_oi = int(transcriptome.gene_ix("Neurod1"));gene_id = transcriptome.var.index[gene_oi]
# gene_oi = int(transcriptome.gene_ix("IL1B"));gene_id = transcriptome.var.index[gene_oi]
# gene_oi = "ENSG00000005379";gene_ix = int(transcriptome.gene_ix(transcriptome.symbol(gene_oi)))

# %%
model.n_genes = fragments.n_genes

# %%
bc = torch.bincount(fragments.genemapping)
(bc / bc.sum())[gene_oi] * fragments.n_genes

# %%
import torch

# %% jp-MarkdownHeadingCollapsed=true
model = model.to(device)

## Plot prior distribution
fig, axes = plt.subplots(
    latent.shape[1], 1, figsize=(20, 1 * latent.shape[1]), sharex=True, sharey=True
)

probs = []

pseudocoordinates = torch.linspace(0, 1, 1000).to(device)

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1]) / 2
binsize = binmids[1] - binmids[0]

fragments_oi_all = fragments.cut_local_gene_ix == gene_oi
print(fragments_oi_all.sum())
for i, ax in zip(range(latent.shape[1]), axes):
    lib_all = model.libsize.cpu().numpy()

    cells_oi = torch.where(latent_torch[:, i])[0]
    lib_oi = model.libsize[cells_oi].cpu().numpy()
    n_cells = latent_torch[:, i].sum()

    color = cluster_info.iloc[i]["color"]
    fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (
        fragments.cut_local_gene_ix == gene_oi
    )

    bincounts, _ = np.histogram(
        fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins
    )
    ax.bar(
        binmids, bincounts / n_cells * len(bins), width=binsize, color="#888888", lw=0
    )

    # Plot initial posterior distribution
    pseudolatent = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device)
    pseudolatent[:, i] = 1.0

    prob = model.evaluate_pseudo(
        pseudocoordinates.to(device), latent=pseudolatent.to(device), gene_oi=gene_oi
    )
    ax.plot(
        pseudocoordinates.cpu().numpy(),
        np.exp(prob),
        label=i,
        color=color,
        lw=2,
        zorder=20,
    )
    ax.plot(
        pseudocoordinates.cpu().numpy(),
        np.exp(prob),
        label=i,
        color="#FFFFFFFF",
        lw=3,
        zorder=10,
    )

    # ax.set_ylabel(f"{cluster_info.iloc[i]['label']}\n# fragments={fragments_oi.sum()}\n {int(latent_torch[:, i].sum())}", rotation = 0, ha = "right", va = "center")
    ax.set_ylabel(
        f"{cluster_info.iloc[i]['label']}\n freq={fragments_oi.sum()/n_cells}",
        rotation=0,
        ha="right",
        va="center",
    )
    ax.set_ylim(0, 40)

    probs.append(prob)
probs = np.stack(probs)

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

nbins = np.array(model.mixture.transform.nbins)
bincuts = np.concatenate([[0], np.cumsum(nbins)])
binmids = bincuts[:-1] + nbins / 2

ax = main[0, 0] = chd.grid.Ax((10, 0.25))
ax = ax.ax
plotdata = (model.mixture.transform.unnormalized_heights.data.cpu().numpy())[[gene_oi]]
ax.imshow(plotdata, aspect="auto")
ax.set_yticks([])
for b in bincuts:
    ax.axvline(b - 0.5, color="black", lw=0.5)
ax.set_xlim(0 - 0.5, plotdata.shape[1] - 0.5)
ax.set_xticks([])
ax.set_ylabel("$h_0$", rotation=0, ha="right", va="center")

ax = main[1, 0] = chd.grid.Ax(dim=(10, n_latent_dimensions * 0.25))
ax = ax.ax
plotdata = model.decoder.logit_weight.data[gene_oi].cpu().numpy()
ax.imshow(
    plotdata, aspect="auto", cmap=mpl.cm.RdBu_r, vmax=np.log(2), vmin=np.log(1 / 2)
)
ax.set_yticks(range(len(cluster_info)))
ax.set_yticklabels(cluster_info.index, rotation=0, ha="right")
for b in bincuts:
    ax.axvline(b - 0.5, color="black", lw=0.5)
ax.set_xlim(-0.5, plotdata.shape[1] - 0.5)

ax.set_xticks(bincuts - 0.5, minor=True)
ax.set_xticks(binmids - 0.5)
ax.set_xticklabels(nbins)
ax.xaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=5, which="minor")
ax.set_ylabel("$\Delta h$", rotation=0, ha="right", va="center")

ax.set_xlabel("Resolution")

fig.plot()

# %% [markdown]
# ### Train

# %%
device = "cuda"
# device = "cpu"

# %%
# model.mixture.transform.unnormalized_heights.data = model.mixture.transform.unnormalized_heights.data.to(torch.float64)
# model.mixture.transform.unnormalized_widths.data = model.mixture.transform.unnormalized_widths.data.to(torch.float64)
# model.decoder.logit_weight.data = model.decoder.logit_weight.data.to(torch.float64)
# model.reflatent = model.reflatent.to(torch.float64)

# %%
optimizer = chd.optim.SparseDenseAdam(
    model.parameters_sparse(), model.parameters_dense(autoextend=True), lr=1e-2
)

# %%
loaders_train.restart()
loaders_validation.restart()
import gc

gc.collect()
torch.cuda.empty_cache()

# %%
import torch_scatter


class GeneLikelihoodHook:
    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.likelihood_mixture = []
        self.likelihood_counts = []

    def start(self):
        self.likelihood_mixture_checkpoint = np.zeros(self.n_genes)
        self.likelihood_counts_checkpoint = np.zeros(self.n_genes)
        return {}

    def run_individual(self, model, data):
        self.likelihood_mixture_checkpoint[data.genes_oi] += (
            torch_scatter.scatter_sum(
                model.track["likelihood"], data.cut_local_gene_ix, dim_size=data.n_genes
            )
            .detach()
            .cpu()
            .numpy()
        )

    def finish(self):
        self.likelihood_mixture.append(self.likelihood_mixture_checkpoint)


hook_genelikelihood = GeneLikelihoodHook(fragments.n_genes)
hooks = [hook_genelikelihood]

# %%
# torch.autograd.set_detect_anomaly(True)

# %%
# minibatches_train

# %%
# with torch.autograd.detect_anomaly():
model = model.to(device).train()
loaders_train.restart()
loaders_validation.restart()
trainer = chd.train.Trainer(
    model,
    loaders_train,
    loaders_validation,
    optimizer,
    n_epochs=50,
    checkpoint_every_epoch=1,
    optimize_every_step=1,
    hooks_checkpoint=hooks,
)
trainer.train()

# %%
likelihood_mixture = pd.DataFrame(
    np.vstack(hook_genelikelihood.likelihood_mixture), columns=fragments.var.index
).T

# %%
scores = (
    (likelihood_mixture.iloc[:, -1] - likelihood_mixture[0])
    .sort_values()
    .to_frame("lr")
)
scores["label"] = transcriptome.symbol(scores.index)

# %%
scores.tail(20)

# %%
pickle.dump(model.to("cpu"), open("./model.pkl", "wb"))

# %%
model = pickle.load(open("./model.pkl", "rb"))

# %% [markdown]
# ## Inference single gene

# %%
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_64-32" / "model_2.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30_freescale" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30_freescale_scalelik" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30_freescale_scalelik_laplace" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30_laplace0.05" / "model_0.pkl").open("rb"))

# %%
sns.histplot(model.decoder.rho_weight.weight.data.numpy().flatten())

# %%
z = model.decoder.logit_weight.weight.data.numpy().flatten()

# %%
import scipy.stats

# %%
sns.histplot(model.decoder.logit_weight.weight.data.numpy().flatten()[:100])

# %%
scipy.stats.laplace.fit(z)

# %%
# model.mixture_delta_p_scale = torch.tensor(0., device = device)
# model.rho_delta_p_scale = torch.tensor(0., device = device)
# model.mixture_delta_p_scale_dist = "normal"

# model.mixture_delta_p_scale
# model.rho_delta_p_scale

# %%
device = "cuda"
model = model.to(device).eval()

# %%
transcriptome.var.head(10)

# %%
gene_oi = 0

# symbol = "DNAH5" # !!
# symbol = "ABHD12B"
# symbol = "HOXB8" # !!
# symbol = "CALB1" # !
# symbol = "DLL3" # !!
# symbol = "HOXD4" # !
# symbol = "GLI3"
# symbol = "CAPN2" # !!
# symbol = "LRRTM4"
# symbol = "SHROOM3"
# symbol = "CALD1"
# symbol = "ADAM28" # !
# symbol = "SLIT3" # !
# symbol = "PDGFRA" # !
# symbol = "HAPLN1" # !
# symbol = "C3orf52" #!?
# symbol = "PHC2" # !!
# symbol = "AFF3"
# symbol = "EFNB2"
# symbol = "RPL4"
# symbol = "HSPA8"
# symbol = "NOTCH2"
# symbol = "STK38L"
# symbol = "FREM1"

# symbol = "SOX5"
symbol = "CD3D"
symbol = "IL1B"
symbol = "SAT1"

gene_oi = int(transcriptome.gene_ix(symbol))
gene_id = transcriptome.var.index[gene_oi]

# %% jp-MarkdownHeadingCollapsed=true
model = model.to(device)

## Plot prior distribution
fig, axes = plt.subplots(
    latent.shape[1], 1, figsize=(20, 1 * latent.shape[1]), sharex=True, sharey=True
)

probs = []

pseudocoordinates = torch.linspace(0, 1, 1000).to(device)

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1]) / 2
binsize = binmids[1] - binmids[0]

fragments_oi_all = fragments.cut_local_gene_ix == gene_oi
print(fragments_oi_all.sum())
for i, ax in zip(range(latent.shape[1]), axes):
    lib_all = model.libsize.cpu().numpy()

    cells_oi = torch.where(latent_torch[:, i])[0]
    lib_oi = model.libsize[cells_oi].cpu().numpy()
    n_cells = latent_torch[:, i].sum()

    color = cluster_info.iloc[i]["color"]
    fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (
        fragments.cut_local_gene_ix == gene_oi
    )

    bincounts, _ = np.histogram(
        fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins
    )
    ax.bar(
        binmids, bincounts / n_cells * len(bins), width=binsize, color="#888888", lw=0
    )

    # Plot initial posterior distribution
    pseudolatent = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device)
    pseudolatent[:, i] = 1.0

    prob = model.evaluate_pseudo(
        pseudocoordinates.to(device), latent=pseudolatent.to(device), gene_oi=gene_oi
    )
    ax.plot(
        pseudocoordinates.cpu().numpy(),
        np.exp(prob),
        label=i,
        color=color,
        lw=2,
        zorder=20,
    )
    ax.plot(
        pseudocoordinates.cpu().numpy(),
        np.exp(prob),
        label=i,
        color="#FFFFFFFF",
        lw=3,
        zorder=10,
    )

    # ax.set_ylabel(f"{cluster_info.iloc[i]['label']}\n# fragments={fragments_oi.sum()}\n {int(latent_torch[:, i].sum())}", rotation = 0, ha = "right", va = "center")
    ax.set_ylabel(
        f"{cluster_info.iloc[i]['label']}\n freq={fragments_oi.sum()/n_cells}",
        rotation=0,
        ha="right",
        va="center",
    )
    ax.set_ylim(0, 40)

    probs.append(prob)
probs = np.stack(probs)

# %%
sns.heatmap(probs, cmap=mpl.cm.RdBu_r)

# %%
probs_diff = probs - probs.mean(0, keepdims=True)

sns.heatmap(probs_diff, cmap=mpl.cm.RdBu_r, center=0.0)
