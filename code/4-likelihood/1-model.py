# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

# %% [markdown]
# ## Simulation

# %%
from simulate import Simulation

# %%
simulation = Simulation(n_cells = 1000, n_genes = 30)

# %%
window = simulation.window

# %%
coordinates, mapping, cell_latent_space= torch.from_numpy(simulation.coordinates).to(torch.float32), torch.from_numpy(simulation.mapping), torch.from_numpy(simulation.cell_latent_space).to(torch.float32)

# %%
cellmapping = mapping[:, 0]
genemapping = mapping[:, 1] 

# %%
## Plot prior distribution
gene_ix = 0
fig, (ax) = plt.subplots(1, 1, figsize=(40, 2))
for dim in range(cell_latent_space.shape[1]):
    fragments_oi = (cell_latent_space[:, dim] > 0)[cellmapping] & (genemapping == gene_ix)
    ax.hist(coordinates.cpu().numpy()[fragments_oi, 0], bins=1000, range = window, histtype = "step")

# %%
# library sizes
sns.histplot(torch.bincount(cellmapping, minlength = simulation.n_cells).numpy())

# %%
# gene means
sns.histplot(torch.bincount(genemapping, minlength = simulation.n_genes).numpy())

# %% [markdown]
# Create fragments

# %%
import chromatinhd as chd
import pathlib
import tempfile

# %%
fragments = chd.data.Fragments(path = pathlib.Path(tempfile.TemporaryDirectory().name))

# %%
# need to make sure the order of fragments is cellxgene
order = np.argsort((simulation.mapping[:, 0] * simulation.n_genes) + simulation.mapping[:, 1])

fragments.coordinates = torch.from_numpy(simulation.coordinates)[order]
fragments.mapping = torch.from_numpy(simulation.mapping)[order]
fragments.var = pd.DataFrame({"gene":np.arange(simulation.n_genes)}).set_index("gene")
fragments.obs = pd.DataFrame({"cell":np.arange(simulation.n_cells)}).set_index("cell")
fragments.create_cellxgene_indptr()

# %% [markdown]
# ## Single gene real

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments_original = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
# one gene
symbol = "IL1B"
# symbol = "Neurod1"
gene_id = transcriptome.gene_id(symbol)
gene_ix = fragments_original.var.loc[gene_id]["ix"]

fragments = chd.data.Fragments(path = pathlib.Path(tempfile.TemporaryDirectory().name))

fragments_oi = fragments_original.genemapping == gene_ix

fragments.coordinates = fragments_original.coordinates[fragments_oi]
fragments.mapping = fragments_original.mapping[fragments_oi]
fragments.mapping[:, 1] = 0
fragments.var = pd.DataFrame({"gene":[1]}).set_index("gene")
fragments.obs = pd.DataFrame({"cell":np.arange(fragments_original.n_cells)}).set_index("cell")
fragments.create_cellxgene_indptr()

# %% [markdown]
# ## Two genes real

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments_original = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
# one gene
# symbols = ["IL1B", "FOSB"]
# symbols = ["FOSB", "IL1B"]
symbols = transcriptome.var.symbol[:100].tolist() + ["IL1B", "FOSB"]
# symbol = "Neurod1"
gene_ids = transcriptome.gene_id(symbols)
gene_ixs = fragments_original.var.loc[gene_ids]["ix"]

fragments = chd.data.Fragments(path = pathlib.Path(tempfile.TemporaryDirectory().name))

fragments_oi = torch.isin(fragments_original.genemapping, torch.from_numpy(gene_ixs.values))

fragments.coordinates = fragments_original.coordinates[fragments_oi]
fragments.mapping = fragments_original.mapping[fragments_oi]
mapper = np.zeros(fragments_original.n_genes)
mapper[gene_ixs] = np.arange(len(gene_ixs))
fragments.mapping[:, 1] = torch.from_numpy(mapper[fragments_original.mapping[fragments_oi][:, 1]])

cellxgene = fragments.mapping[:, 0] * len(gene_ids) + fragments.mapping[:, 1]
order = torch.argsort(cellxgene)
fragments.coordinates = fragments.coordinates[order]
fragments.mapping = fragments.mapping[order]

fragments.var = pd.DataFrame({"gene":range(len(gene_ids))}).set_index("gene")
fragments.obs = pd.DataFrame({"cell":np.arange(fragments_original.n_cells)}).set_index("cell")
fragments.create_cellxgene_indptr()

# %%
fragments.window = window
fragments.create_cut_data()

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
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %% [markdown]
# ## Create loaders

# %%
fragments.window = window
fragments.create_cut_data()

# %%
cells_train = np.arange(0, int(fragments.n_cells * 9 / 10))
cells_validation = np.arange(int(fragments.n_cells * 9 / 10), fragments.n_cells)

# %%
import chromatinhd.loaders.fragments
# n_cells_step = 1000
# n_genes_step = 1000

n_cells_step = 100
n_genes_step = 5000

# n_cells_step = 2000
# n_genes_step = 500

loaders_train = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
    n_workers = 20,
    shuffle_on_iter = True
)
minibatches_train = chd.loaders.minibatching.create_bins_random(
    cells_train,
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step = n_genes_step,
    n_cells_step=n_cells_step,
    use_all = True,
    permute_genes = False
)
loaders_train.initialize(minibatches_train)

loaders_validation = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
    n_workers = 5
)
minibatches_validation = chd.loaders.minibatching.create_bins_random(cells_validation, np.arange(fragments.n_genes), fragments.n_genes, n_genes_step = n_genes_step, n_cells_step=n_cells_step, use_all = True, permute_genes = False)
loaders_validation.initialize(minibatches_validation)

# %%
# device = "cuda"
# data = loaders_train.pull()
# data = data.to(device)

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
transcriptome.obs["cluster"] = transcriptome.adata.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# ### Create model

# %%
reflatent = torch.eye(n_latent_dimensions).to(torch.float)
reflatent_idx = torch.from_numpy(np.where(latent.values)[1])

# %%
# import chromatinhd.models.vae.v3 as vae_model
# model = vae_model.Decoding(fragments, latent, n_bins = 50)

# import chromatinhd.models.likelihood.v2 as vae_model
# model = vae_model.Decoding(fragments, latent, n_components = 64)

# import chromatinhd.models.likelihood.v4 as vae_model
# model = vae_model.Decoding(fragments, torch.from_numpy(latent.values), nbins = (64, 32, ))


import chromatinhd.models.likelihood.v8 as vae_model
# import chromatinhd.models.likelihood.v5 as vae_model
# model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (256, 128, 64, ))
# model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (64, 32, ))
# model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (64, ))
# model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (128, ))
model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (128, 64, 32, 16, 8, 4))
# model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (32, 64, 128, ))
# model = vae_model.Decoding(fragments, reflatent, reflatent_idx, nbins = (128, ))

# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v2_baseline" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v2" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_64_1l" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_64-32" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30_rep" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_256-128-64-32" / "model_0.pkl").open("rb"))
# model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v5_128-64-32" / "model_0.pkl").open("rb"))

# %%
design = chd.utils.crossing(
    # pd.DataFrame({"gene_ix":[2]}),
    pd.DataFrame({"gene_ix":np.arange(fragments.n_genes)}),
    pd.DataFrame({"coord":np.linspace(0, 1, 1000)}),
    pd.DataFrame({"reflatent":[0, 1]})
)

# %%
sns.heatmap((model.mixture.transform.unnormalized_heights.data.cpu().numpy()))

# %%
device = "cpu"

# %%
import chromatinhd.models.likelihood.v5.quadratic

# %% [markdown]
# ### Prior distribution

# %%
device = "cuda"
model = model.to(device)

# %%
fragments.create_cut_data()

# %%
# gene_oi = symbols.index("CCL4")
gene_oi = int(transcriptome.gene_ix("CCL4"));gene_id = transcriptome.var.index[gene_oi]
# gene_oi = "ENSG00000005379";gene_ix = int(transcriptome.gene_ix(transcriptome.symbol(gene_oi)))

# %%
model.n_genes = fragments.n_genes

# %%
bc = torch.bincount(fragments.genemapping)
bc / bc.sum()

# %% jp-MarkdownHeadingCollapsed=true tags=[]
model = model.to(device)

## Plot prior distribution
fig, axes = plt.subplots(latent.shape[1], 1, figsize=(20, 1*latent.shape[1]), sharex = True, sharey = True)

prob_mixtures = []
rho_deltas = []
rhos = []

pseudocoordinates = torch.linspace(0, 1, 500).to(device)

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]

fragments_oi_all = (fragments.cut_local_gene_ix == gene_oi)
print(fragments_oi_all.sum())
for i, ax in zip(range(latent.shape[1]), axes):
    lib_all = model.libsize.cpu().numpy()
    
    cells_oi = torch.where(latent_torch[:, i])[0]
    lib_oi = model.libsize[cells_oi].cpu().numpy()
    n_cells = latent_torch[:, i].sum()
    
    color = cluster_info.iloc[i]["color"]
    fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi)
    
    bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins)
    ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0)

    # Plot initial posterior distribution
    pseudolatent = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device)
    pseudolatent[:, i] = 1.
    
    likelihood_mixture, rho_delta, rho, _, _ = model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_oi = gene_oi)
    prob_mixture = likelihood_mixture.numpy()
    print(np.trapz(np.exp(prob_mixture), np.linspace(0, 1, len(prob_mixture))))
    rho_delta = rho_delta.numpy()
    rho = rho.numpy()
    ax.plot(pseudocoordinates.cpu().numpy(), np.exp(prob_mixture), label = i, color = color, lw = 2, zorder = 20)
    ax.plot(pseudocoordinates.cpu().numpy(), np.exp(prob_mixture), label = i, color = "#FFFFFFFF", lw = 3, zorder = 10)
    
    # ax.set_ylabel(f"{cluster_info.iloc[i]['label']}\n# fragments={fragments_oi.sum()}\n {int(latent_torch[:, i].sum())}", rotation = 0, ha = "right", va = "center")
    ax.set_ylabel(f"{cluster_info.iloc[i]['label']}\n freq={fragments_oi.sum()/n_cells}", rotation = 0, ha = "right", va = "center")
    ax.set_ylim(0, 10)
    
    prob_mixtures.append(prob_mixture)
    rho_deltas.append(rho_delta)
    rhos.append(rho)

# %%
prob_mixtures = np.stack(prob_mixtures)

# %%
plt.plot(np.exp(prob_mixtures[5] - prob_mixtures.mean(0)))

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

nbins = np.array(model.mixture.transform.nbins)
bincuts = np.concatenate([[0], np.cumsum(nbins)])
binmids = bincuts[:-1] + nbins/2

ax = main[0, 0] = chd.grid.Ax((10, 0.25))
ax = ax.ax
plotdata = (model.mixture.transform.unnormalized_heights.data.cpu().numpy())[[gene_oi]]
ax.imshow(plotdata, aspect = "auto")
ax.set_yticks([])
for b in bincuts:
    ax.axvline(b-0.5, color = "black", lw = 0.5)
ax.set_xlim(0-0.5, plotdata.shape[1]-0.5)
ax.set_xticks([])
ax.set_ylabel("$h_0$", rotation = 0, ha = "right", va = "center")

ax = main[1, 0] = chd.grid.Ax(dim = (10, n_latent_dimensions * 0.25))
ax = ax.ax
plotdata = (model.decoder.logit_weight[gene_oi].data.cpu().numpy())
ax.imshow(plotdata, aspect = "auto", cmap = mpl.cm.RdBu_r, vmax = np.log(2), vmin = np.log(1/2))
ax.set_yticks(range(len(cluster_info)))
ax.set_yticklabels(cluster_info.index, rotation = 0, ha = "right")
for b in bincuts:
    ax.axvline(b-0.5, color = "black", lw = 0.5)
ax.set_xlim(-0.5, plotdata.shape[1]-0.5)

ax.set_xticks(bincuts-0.5, minor = True)
ax.set_xticks(binmids-0.5)
ax.set_xticklabels(nbins)
ax.xaxis.set_tick_params(length = 0)
ax.xaxis.set_tick_params(length = 5, which = "minor")
ax.set_ylabel("$\Delta h$", rotation = 0, ha = "right", va = "center")

ax.set_xlabel("Resolution")

fig.plot()

# %% [markdown] tags=[]
# ### Train

# %%
device = "cuda"
# device = "cpu"

# %%
optimizer = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-2)

# %%
loaders_train.restart()
loaders_validation.restart()
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
import torch_scatter
class GeneLikelihoodHook():
    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.likelihood_mixture = []
        self.likelihood_counts = []
        
    def start(self):
        self.likelihood_mixture_checkpoint = np.zeros(self.n_genes)
        self.likelihood_counts_checkpoint = np.zeros(self.n_genes)
        return {}
        
    def run_individual(self, model, data):
        self.likelihood_mixture_checkpoint[data.genes_oi] += torch_scatter.scatter_sum(model.track["likelihood_mixture"], data.cut_local_gene_ix, dim_size = data.n_genes).detach().cpu().numpy()
        
    def finish(self):
        self.likelihood_mixture.append(self.likelihood_mixture_checkpoint)
        
hook_genelikelihood = GeneLikelihoodHook(fragments.n_genes)
hooks = [hook_genelikelihood]

# %%
# torch.autograd.set_detect_anomaly(True)

# %%
# with torch.autograd.detect_anomaly():
model = model.to(device).train()
loaders_train.restart()
loaders_validation.restart()
trainer = chd.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 10, checkpoint_every_epoch=1, optimize_every_step = 1, hooks_checkpoint = hooks)
trainer.train()

# %%
likelihood_mixture = pd.DataFrame(np.vstack(hook_genelikelihood.likelihood_mixture), columns = fragments.var.index).T
likelihood_counts = pd.DataFrame(np.vstack(hook_genelikelihood.likelihood_counts), columns = fragments.var.index).T

# %%
scores = (likelihood_mixture.iloc[:, -1] - likelihood_mixture[0]).sort_values().to_frame("lr")
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
model = pickle.load((chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / "v4_128-64-32_30" / "model_0.pkl").open("rb"))
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
    
gene_oi = int(transcriptome.gene_ix(symbol));gene_id = transcriptome.var.index[gene_oi]

# %% jp-MarkdownHeadingCollapsed=true tags=[]
## Plot prior distribution
# fig, axes = plt.subplots(latent.shape[1], 1, figsize=(20, 1*latent.shape[1]), sharex = True, sharey = True)

prob_mixtures = []
rho_deltas = []
rhos = []

pseudocoordinates = torch.linspace(0, 1, 1000).to(device)

fragments_oi_all = (fragments.cut_local_gene_ix == gene_oi)
for i, ax in zip(range(latent.shape[1]), axes):
    # color = cluster_info.loc[i]["color"]
    # fragments_oi = (latent[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi)
    # ax.hist(fragments.cut_coordinates[fragments_oi_all].cpu().numpy(), bins=200, range = (0, 1), lw = 1, density = True, histtype = "step", ec = "#333333FF", zorder = 10)
    # ax.hist(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=200, range = (0, 1), lw = 0, density = True, zorder = 0, histtype = "bar", color = "#333333AA")

    # Plot initial posterior distribution
    pseudolatent = torch.zeros((1, latent.shape[1])).to(device)
    pseudolatent[:, i] = 1.
    
    likelihood_mixture, rho_delta, rho = model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_oi = gene_oi)
    prob_mixture = likelihood_mixture.numpy()
    rho_delta = rho_delta.numpy()
    rho = rho.numpy()
    
    # ax.plot(pseudocoordinates.cpu().numpy(), np.exp(prob_mixture), label = i, color = color, lw = 2, zorder = 20)
    # ax.plot(pseudocoordinates.cpu().numpy(), np.exp(prob_mixture), label = i, color = "#FFFFFFFF", lw = 3, zorder = 10)
    # ax.set_ylabel(f"{cluster_info.loc[i]['label']}\nn={fragments_oi.sum()}\n {int(latent[:, i].sum())}", rotation = 0, ha = "right", va = "center")
    # ax.set_ylim(0, 20)
    
    prob_mixtures.append(prob_mixture)
    rho_deltas.append(rho_delta)
    rhos.append(rho)

# %%
mixture = np.stack(prob_mixtures)
sns.heatmap(mixture, cmap = mpl.cm.RdBu_r)

# %%
mixture_diff = mixture - mixture.mean(0, keepdims = True)
mixture_diff = mixture_diff - mixture_diff.mean(1, keepdims = True)

sns.heatmap(mixture_diff, cmap = mpl.cm.RdBu_r, center = 0.)

# %%
rho_deltas = np.stack(rho_deltas)
rho_deltas = rho_deltas# - rho_deltas.mean()

# %%
rhos = np.stack(rhos)

# %%
total_diff = mixture_diff + rho_deltas

# %%
sns.heatmap(total_diff, cmap = mpl.cm.RdBu_r, center = 0)

# %%
rho_cutoff = np.log(1.0)
# rho_cutoff = -np.inf

mask = (rhos > rho_cutoff)
total_diff[~mask] = np.nan
mixture_diff[~mask] = np.nan
X = total_diff.copy()
X[np.isnan(X)] = 0.

# %%
rhos.shape

# %%
fig, (ax0, ax1) = plt.subplots(2, 1, figsize = (7, 7))

sns.heatmap(rhos, ax = ax0)
ax0.set_yticks(np.arange(len(cluster_info)))
ax0.set_yticklabels(cluster_info["label"], rotation = 0, ha = "right")

sns.heatmap(total_diff, cmap = mpl.cm.RdBu_r, center = 0, ax = ax1)
ax1.set_yticks(np.arange(len(cluster_info))+0.5)
ax1.set_yticklabels(cluster_info["label"], rotation = 0, ha = "right")
""

# %%
import sklearn.decomposition

# %%
pca = sklearn.decomposition.PCA()
Z = pca.fit_transform(X)

# %%
transcriptome.adata.obs["grouping"] = pd.Categorical(pd.from_dummies(pd.DataFrame(latent.numpy())).values[:, 0])
# transcriptome.adata.obs["grouping"] = transcriptome.adata.obs["overexpressed"]
sc.pl.umap(transcriptome.adata, color = ["grouping", gene_id], legend_loc = "on data")

# %%
# add outcome to cluster info
cluster_info["outcome"] = pd.DataFrame({"cluster":transcriptome.adata.obs["grouping"].cat.codes, "outcome":sc.get.obs_df(transcriptome.adata, gene_id)}).groupby("cluster").mean()["outcome"]

# %%
fig, ax = plt.subplots()

import adjustText
plotdata = pd.concat([
    cluster_info,
    pd.DataFrame({"PC1":Z[:, 0], "PC2":Z[:, 1]})
], axis = 1)

outcome_cmap = mpl.cm.viridis
outcome_norm = mpl.colors.Normalize()
ax.scatter(
    plotdata["PC1"],
    plotdata["PC2"],
    c = outcome_cmap(outcome_norm(plotdata["outcome"]))
)
texts = []
for _, row in plotdata.iterrows():
    texts.append(ax.text(
        row["PC1"],
        row["PC2"],
        s = row["label"],
        ha = "center",
        va = "center",
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.2')
    ))
adjustText.adjust_text(texts)

# %% [markdown]
# ## Inference all genes + all latent dimensions

# %%
method_name = 'v4_128-64-32_30_rep'
class Prediction(chd.flow.Flow):
    pass
prediction = Prediction(chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / method_name)
model = pickle.load((prediction.path / "model_0.pkl").open("rb"))

# %%
# device = "cpu"
device = "cuda"
model = model.to(device).eval()

# %%
design_gene = pd.DataFrame({"gene_ix":np.arange(fragments.n_genes)})
# design_gene = pd.DataFrame({"gene_ix":[gene_oi]})
design_latent = pd.DataFrame({"active_latent":np.arange(latent.shape[1])})
design_coord = pd.DataFrame({"coord":np.arange(window[0], window[1]+1, step = 25)})
design = chd.utils.crossing(design_gene, design_latent, design_coord)
design["batch"] = np.floor(np.arange(design.shape[0]) / 10000).astype(int)

# %%
mixtures = []
rho_deltas = []
rhos = []
probs = []
probs2 = []
for _, design_subset in tqdm.tqdm(design.groupby("batch")):
    pseudocoordinates = torch.from_numpy(design_subset["coord"].values).to(device)
    pseudocoordinates = (pseudocoordinates - window[0]) / (window[1] - window[0])
    pseudolatent = torch.nn.functional.one_hot(torch.from_numpy(design_subset["active_latent"].values).to(device), latent.shape[1]).to(torch.float)
    gene_ix = torch.from_numpy(design_subset["gene_ix"].values).to(device)
    
    mixture, rho_delta, rho, prob, prob2 = model.evaluate_pseudo2(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_ix = gene_ix)
   
    mixtures.append(mixture.numpy())
    rho_deltas.append(rho_delta.numpy())
    rhos.append(rho.numpy())
    probs.append(prob.numpy())
    probs2.append(prob2.numpy())
mixtures = np.hstack(mixtures)
rho_deltas = np.hstack(rho_deltas)
rhos = np.hstack(rhos)
probs = np.hstack(probs)
probs2 = np.hstack(probs2)

# %%
mixtures = mixtures.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
rho_deltas = rho_deltas.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
rhos = rhos.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
probs = probs.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
probs2 = probs2.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))

mixture_diff = mixtures - mixtures.mean(-2, keepdims = True)
probs_diff = mixture_diff + rho_deltas# - rho_deltas.mean(-2, keepdims = True)

# %%
prob_cutoff = np.log(1.0)
# rho_cutoff = -np.inf

# %%
mask = probs2 > prob_cutoff

# %%
probs_diff_masked = probs_diff.copy()
probs_diff_masked[~mask] = np.nan
mixture_diff_masked = mixture_diff.copy()
mixture_diff_masked[~mask] = np.nan

X = mixture_diff_masked.copy()
X[np.isnan(X)] = 0.

# %%
pickle.dump(probs2, (prediction.path / "probs2.pkl").open("wb"))
pickle.dump(probs, (prediction.path / "probs.pkl").open("wb"))
pickle.dump(mixtures, (prediction.path / "mixtures.pkl").open("wb"))
pickle.dump(rhos, (prediction.path / "rhos.pkl").open("wb"))
pickle.dump(rho_deltas, (prediction.path / "rho_deltas.pkl").open("wb"))

# %%
design["gene_ix"] = design["gene_ix"].astype("category")
design["active_latent"] = design["active_latent"].astype("category")
design["batch"] = design["batch"].astype("category")
design["coord"] = design["coord"].astype("category")
pickle.dump(design, (prediction.path / "design.pkl").open("wb"))

# %%
X2 = np.clip(X, np.quantile(X, 0.01, 0, keepdims = True), np.quantile(X, 0.99, 0, keepdims = True))

# %%
pd.DataFrame({
    "prob_diff":(X2**2).mean(-1).mean(-1),
    "gene":fragments.var.index,
    "symbol":transcriptome.var["symbol"]
}).sort_values("prob_diff", ascending = False).head(20)

cluster_oi = 0
pd.DataFrame({
    "prob_diff":(X2**2).mean(-1)[..., cluster_oi],
    "gene":fragments.var.index,
    "symbol":transcriptome.var["symbol"]
}).sort_values("prob_diff", ascending = False).head(20)

# %%
# gene_oi = 0; symbol = transcriptome.var.iloc[gene_oi]["symbol"];gene_id = transcriptome.var.index[gene_oi]

# symbol = "Csf1"
# symbol = "THOC3"
# symbol = "TCF4"
# symbol = "ANXA4"
symbol = "ADGRG5"
# symbol = "FREM1"
# symbol = "RUNX3"
# symbol = "PLAUR"
symbol = "Neurod2"
gene_oi = int(transcriptome.gene_ix(symbol));gene_id = transcriptome.var.index[gene_oi]

# %%
sns.heatmap(probs_diff_masked[gene_oi], cmap = mpl.cm.RdBu_r, center = 0)

# %% [markdown]
# ## Single base pair interpolation

# %%
x = (design["coord"].values).astype(int).reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
y = probs_diff


# %%
def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    a = (fp[...,1:] - fp[...,:-1]) / (xp[...,1:] - xp[...,:-1])
    b = fp[..., :-1] - (a.mul(xp[..., :-1]) )

    indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
    indices = torch.clamp(indices, 0, a.shape[-1] - 1)
    slope = a.index_select(a.ndim-1, indices)
    intercept = b.index_select(a.ndim-1, indices)
    return x * slope + intercept


# %%
desired_x = torch.arange(*window)

# %%
probs_diff_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs_diff)).numpy()
rhos_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(rhos)).numpy()
probs_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)).numpy()

# %%
sns.heatmap(probs_diff_interpolated[gene_oi], cmap = mpl.cm.RdBu_r, center = 0.)

# %%
rho_cutoff = np.log(1.)
# rho_cutoff = np.log(0.5)
# rho_cutoff = -np.inf

# %%
probs_diff_interpolated_masked = probs_diff_interpolated.copy()
mask_interpolated = (rhos_interpolated > rho_cutoff)
probs_diff_interpolated_masked[~mask_interpolated] = np.nan

# %%
sns.heatmap(probs_diff_interpolated_masked[gene_oi], cmap = mpl.cm.RdBu_r, center = 0.)

# %%
fig, ax = plt.subplots()
for cluster in cluster_info.index:
    ax.plot(probs_diff_interpolated_masked[gene_oi][cluster], color = cluster_info.loc[cluster, "color"])

# %% [markdown]
# ## Length of differential stretches

# %%
cutoff = 0.5

# %%
slices = []
for cluster_ix in range(len(cluster_info)):
    positions_oi = (probs_diff_interpolated_masked[:, cluster_ix] >= cutoff)
    gene_ixs, positions = np.where(positions_oi)
    groups = np.hstack([0, np.cumsum((np.diff(gene_ixs * ((window[1] - window[0]) + 1) + positions) != 1))])
    cuts = np.where(np.hstack([True, (np.diff(groups)[:-1] != 0), True]))[0]
    
    position_slices = np.vstack((positions[cuts[:-1]], positions[cuts[1:]-1])).T
    gene_ixs_slices = gene_ixs[cuts[:-1]]
    slices.append(pd.DataFrame({
        "left":position_slices[:, 0],
        "right":position_slices[:, 1],
        "gene_ix":gene_ixs_slices,
        "cluster_ix":cluster_ix
    }))
slices = pd.concat(slices)

# %%
slices["length"] = slices["right"] - slices["left"]

# %%
bins = 10**np.linspace(0, 4, 50)

# %%
bincounts = np.histogram(slices["length"], bins)[0]

# %%
fig, ax = plt.subplots()
# ax.bar(bins[:-1], bincounts)
ax.hist(slices["length"], bins=bins)
ax.set_xscale("log")

# %% [markdown]
# ## Enrichment

# %%
# pmc10k cellranger
# cluster_ix = 0; perc = 0.00576642
# cluster_ix = 1; perc = 0.02668863
# cluster_ix = 2; perc = 0.00349143
# cluster_ix = 3; perc = 0.00685638
cluster_ix = 4; perc = 0.00892843
# cluster_ix = 5; perc = 0.00451207

# pmc10k macs2
cluster_ix = 0; perc = 0.00292257
cluster_ix = 1; perc = 0.01063146

# e18brain
cluster_ix = 0; perc = 0.00203288
# cluster_ix = 1; perc = 0.00140263
# cluster_ix = 2; perc = 0.00453203
# cluster_ix = 3; perc = 0.00046647

# e18brain window_500
# cluster_ix = 2; perc = 0.047746

cutoff = np.quantile(probs_diff_interpolated_masked[:, cluster_ix], 1-perc)

# %%
positions_oi = (probs_diff_interpolated_masked[:, cluster_ix] >= cutoff)

# %%
gene_ixs, positions = np.where(positions_oi)
groups = np.hstack([0, np.cumsum((np.diff(gene_ixs * ((window[1] - window[0]) + 1) + positions) != 1))])
cuts = np.where(np.hstack([True, (np.diff(groups)[:-1] != 0), True]))[0]

# %%
position_slices = np.vstack((positions[cuts[:-1]], positions[cuts[1:]-1])).T
gene_ixs_slices = gene_ixs[cuts[:-1]]

# %%
sns.histplot(position_slices[:, 0], bins = np.linspace(0, (window[1] - window[0]), 20))
sns.histplot(position_slices[:, 1], bins = np.linspace(0, (window[1] - window[0]), 20))

# %%
pd.DataFrame({
    "selected":(np.bincount(gene_ixs, minlength = fragments.n_genes)),
    "gene":fragments.var.index,
    "symbol":transcriptome.var["symbol"]
}).sort_values("selected", ascending = False).head(10)

# %% [markdown]
# GC content and background

# %%
onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb")).flatten(0, 1)


# %%
def count_gc(relative_starts, relative_end, gene_ixs, onehot_promoters):
    gc = []
    n = 0
    for relative_start, relative_end, gene_ix in tqdm.tqdm(zip(relative_starts, relative_end, gene_ixs)):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        gc.append(onehot_promoters[start_ix:end_ix, [1, 2]].sum() / (end_ix - start_ix + 1e-5))
        
    gc = torch.hstack(gc).numpy()
    
    return gc


# %%
promoter_gc = count_gc(torch.ones(fragments.n_genes, dtype = torch.int) * window[0], torch.ones(fragments.n_genes, dtype = torch.int) * window[1], torch.arange(fragments.n_genes), onehot_promoters)
sns.histplot(
    promoter_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(promoter_gc.mean(), color = "blue")

window_oi_gc = count_gc(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, onehot_promoters)
sns.histplot(
    window_oi_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_oi_gc.mean(), color = "orange")

# %%
n_random = 100
n_select_random = 10
position_slices_repeated = position_slices.repeat(n_random, 0)
random_position_slices = np.zeros_like(position_slices_repeated)
random_position_slices[:, 0] = np.random.randint(np.ones(position_slices_repeated.shape[0]) * window[0], np.ones(position_slices_repeated.shape[0]) * window[1] - (position_slices_repeated[:, 1] - position_slices_repeated[:, 0]))
random_position_slices[:, 1] = random_position_slices[:, 0] + (position_slices_repeated[:, 1] - position_slices_repeated[:, 0])
random_gene_ixs_slices = np.random.randint(fragments.n_genes, size = random_position_slices.shape[0])

# %%
window_random_gc = count_gc(random_position_slices[:, 0], random_position_slices[:, 1], random_gene_ixs_slices, onehot_promoters)

# %%
np.repeat(np.arange(position_slices.shape[0]), n_select_random).shape

# %%
random_difference = np.abs((window_random_gc.reshape((position_slices.shape[0], n_random)) - window_oi_gc[:, None]))

chosen_background = np.argsort(random_difference, axis = 1)[:, :n_select_random].flatten()
chosen_background_idx = np.repeat(np.arange(position_slices.shape[0]), n_select_random) * n_random + chosen_background

background_position_slices = random_position_slices[chosen_background_idx]
background_gene_ixs_slices = random_gene_ixs_slices[chosen_background_idx]

# %%
plt.scatter(window_oi_gc, window_random_gc[chosen_background_idx[::n_select_random]])

# %%
window_background_gc = count_gc(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, onehot_promoters)

# %%
sns.histplot(
    promoter_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(promoter_gc.mean(), color = "blue")

sns.histplot(
    window_oi_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_oi_gc.mean(), color = "orange")

sns.histplot(
    window_random_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_random_gc.mean(), color = "green")

sns.histplot(
    window_background_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_background_gc.mean(), color = "red")

# %%
i = 100
plt.plot(probs_diff_interpolated_masked[gene_ixs_slices[i], cluster_ix, (position_slices[i,0]-10):(position_slices[i, 1]+10)])
plt.axhline(cutoff)

# %%
motifscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name / "cutoff_0001"
motifscan = chd.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))


# %%
def count_motifs(relative_starts, relative_end, gene_ixs, motifscan_indices, motifscan_indptr):
    motif_indices = []
    n = 0
    for relative_start, relative_end, gene_ix in tqdm.tqdm(zip(relative_starts, relative_end, gene_ixs)):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        motif_indices.append(motifscan_indices[motifscan_indptr[start_ix]:motifscan_indptr[end_ix]])
        n += relative_end - relative_start
    motif_indices = np.hstack(motif_indices)
    motif_counts = np.bincount(motif_indices, minlength = len(motifs))
    
    return motif_counts, n


# %%
motif_counts, n = count_motifs(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, motifscan.indices, motifscan.indptr)

# %%
n

# %%
motif_counts2, n2 = count_motifs(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, motifscan.indices, motifscan.indptr)
# motif_counts2, n2 = np.bincount(motifscan.indices, minlength = motifs_oi.shape[0]), fragments.n_genes * (window[1] - window[0])

# %%
n/n2

# %%
n / (fragments.n_genes * (window[1] - window[0]))

# %%
contingencies = np.stack([
    np.stack([n2 - motif_counts2, motif_counts2]),
    np.stack([n - motif_counts, motif_counts]),
]).transpose(2, 0, 1)
import scipy.stats
odds_conditional = []
for cont in contingencies:
    odds_conditional.append(scipy.stats.contingency.odds_ratio(cont + 1, kind='conditional').statistic)

motifscores = pd.DataFrame({
    "odds":((motif_counts+1) / (n+1)) / ((motif_counts2+1) / (n2+1)),
    "odds_conditional":odds_conditional,
    "motif":motifs.index
}).set_index("motif")
motifscores["logodds"] = np.log(odds_conditional)

# %%
motifscores.loc[motifscores.index.str.contains("SPI")]

# %%
motifscores.sort_values("odds", ascending = False)

# %%
motifscores.to_csv(chd.get_output() / "a.csv")

# %%
sns.histplot(motifscores["odds"])

# %%
motifscores

# %%
transcriptome.adata.obs["leiden"] = pd.Categorical(pd.from_dummies(pd.DataFrame(latent.numpy())).values[:, 0])
sc.pl.umap(transcriptome.adata, color = ["leiden", gene_id], legend_loc = "on data")

# %% [markdown]
# ## Gene investigation

# %%
pd.DataFrame({
    "prob_diff":(X2**2).mean(-1).mean(-1),
    "gene":fragments.var.index,
    "symbol":transcriptome.var["symbol"]
}).sort_values("prob_diff", ascending = False).head(20)

cluster_oi = 0
pd.DataFrame({
    "prob_diff":(X2**2).mean(-1)[..., cluster_oi],
    "gene":fragments.var.index,
    "symbol":transcriptome.var["symbol"]
}).sort_values("prob_diff", ascending = False).head(20)

# %%
# gene_oi = 0; symbol = transcriptome.var.iloc[gene_oi]["symbol"];gene_id = transcriptome.var.index[gene_oi]

# symbol = "THOC3"
# symbol = "ITPKB"
# symbol = "TCF4"
symbol = "ANXA4"
# symbol = "FOSB"
gene_oi = int(transcriptome.gene_ix(symbol));gene_id = transcriptome.var.index[gene_oi]

# %%
sc.pl.umap(transcriptome.adata, color = ["cluster", "celltype", gene_id], legend_loc = "on data")
# sc.pl.umap(transcriptome.adata, color = ["gene_overexpressed", "cluster", gene_id])

# %%
sns.heatmap(probs_diff_interpolated_masked[gene_oi], cmap = mpl.cm.RdBu_r, center = 0., yticklabels = cluster_info["label"])

# %%
peaks_name = "cellranger"
motifscan_name = "cutoff_0001"

# %%
scores_dir = (prediction.path / "scoring" / peaks_name / motifscan_name)
motifscores_all = pd.read_pickle(scores_dir / "motifscores_all.pkl")

# %%
motifs_oi = pd.DataFrame([
    [motifs.loc[motifs.index.str.contains("TF7L2")].index[0], [0]],
    [motifs.loc[motifs.index.str.contains("TF7L1")].index[0], [0]],
    [motifs.loc[motifs.index.str.contains("GATA3")].index[0], [0]],
    [motifs.loc[motifs.index.str.contains("PEBB")].index[0], [2]],
    [motifs.loc[motifs.index.str.contains("SPI1")].index[0], [1]],
    [motifs.loc[motifs.index.str.contains("PO2F2")].index[0], [3]],
    [motifs.loc[motifs.index.str.contains("BHA15")].index[0], [4]],
], columns = ["motif", "clusters"]).set_index("motif")
motifs_oi["ix"] = motifs.loc[motifs_oi.index, "ix"].values
assert len(motifs_oi) == len(motifs_oi.index.unique())
motifs_oi["color"] = sns.color_palette(n_colors = len(motifs_oi))
motifs_oi["label"] = motifs.loc[motifs_oi.index, "gene_label"]

# %%
indptr_start = gene_oi * (window[1] - window[0])
indptr_end = (gene_oi + 1) * (window[1] - window[0])

# %%
motifdata = []
for motif in motifs_oi.index:
    motif_ix = motifs.loc[motif, "ix"]
    for pos in range(indptr_start, indptr_end):
        pos_indices = motifscan.indices[motifscan.indptr[pos]:motifscan.indptr[pos+1]]
        if motif_ix in pos_indices:
            motifdata.append({"position":pos - indptr_start + window[0], "motif":motif})
motifdata = pd.DataFrame(motifdata)
print(len(motifdata))

# %%
n_clusters = latent.shape[1]

# %%
plotdata_expression = sc.get.obs_df(transcriptome.adata, [gene_id, "latent"]).rename(columns = {gene_id:"expression", "latent":"cluster"})

# %%
plotdata_atac = design.query("gene_ix == @gene_oi").copy().rename(columns = {"active_latent":"cluster"}).set_index(["coord", "cluster"]).drop(columns = ["batch", "gene_ix"])
plotdata_atac["prob"] = probs[gene_oi].flatten()
plotdata_atac["prob_diff"] = plotdata_atac["prob"] - plotdata_atac.groupby("coord")["prob"].mean()
plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

# %%
plotdata_genome = plotdata_atac
plotdata_genome_mean = plotdata_atac_mean

# plotdata_genome = plotdata_peaks.loc["macs2"]
# plotdata_genome_mean = plotdata_peaks_mean.loc["macs2"]

# %%
import chromatinhd.grid
main = chd.grid.Grid(2, 3, padding_width = 0.1, padding_height = 0.5)
fig = chd.grid.Figure(main)

padding_height = 0.001

wrap_expression = chd.grid.Wrap(ncol = 1, padding_height = padding_height)
wrap_expression.set_title("RNA-seq")
main[0, 0] = wrap_expression

wrap_genome = chd.grid.Wrap(ncol = 1, padding_height = padding_height)
wrap_genome.set_title("ATAC-seq cuts")
main[0, 1] = wrap_genome

wrap_motiflegend = chd.grid.Wrap(ncol = 1,padding_height = padding_height)
wrap_motiflegend.set_title("Motifs")
main[0, 2] = wrap_motiflegend

wrap_peaks = chd.grid.Wrap(ncol = 1, padding_height = padding_height)
main[1, 1] = wrap_peaks

fragments_oi_all = (fragments.cut_local_gene_ix == gene_oi)
cut_coordinates_all = fragments.cut_coordinates[fragments_oi_all].cpu().numpy()
cut_coordinates_all = cut_coordinates_all * (window[1] - window[0]) + window[0]

panel_height = 0.5
panel_width = 10

show_atac_diff = False
norm_atac_diff = mpl.colors.Normalize(-1, 1., clip = True)

lim_expression = (0, plotdata_expression.groupby("cluster")["expression"].mean().max() * 1.2)

for cluster_ix in range(n_clusters):
    ax_expression = chd.grid.Ax((0.3, panel_height))
    wrap_expression.add(ax_expression)
    
    ax_genome = chd.grid.Ax((panel_width, panel_height))
    wrap_genome.add(ax_genome)
    
    ax_motiflegend = chd.grid.Ax((1, panel_height))
    wrap_motiflegend.add(ax_motiflegend)
    
    # genome    
    ax = ax_genome.ax
    ax.set_ylim(0, 20)
    
    ax.set_xlim(*window)
    
    # empirical distribution of atac-seq cuts    
    fragments_oi = (latent[fragments.cut_local_cell_ix, cluster_ix] != 0) & (fragments.cut_local_gene_ix == gene_oi)
    cut_coordinates = fragments.cut_coordinates[fragments_oi].cpu().numpy()
    cut_coordinates = cut_coordinates * (window[1] - window[0]) + window[0]
    
    n_bins = 250
    cuts = np.linspace(*window, n_bins)
    bincounts, bins = np.histogram(cut_coordinates, bins = cuts, range = window)
    binmids = bins[:-1] + (bins[:-1] - bins[1:]) /2
    bindensity = bincounts/bincounts.sum() * n_bins
    ax.fill_between(binmids, bindensity, step="post", alpha=0.2, color = "#333")

    if show_atac_diff:
        # posterior distribution of atac-seq cuts
        plotdata_genome_cluster = plotdata_genome.xs(cluster_ix, level = "cluster")
        ax.plot(plotdata_genome_mean.index, np.exp(plotdata_genome_mean["prob"]), color = "black", lw = 1, zorder = 1, linestyle = "dashed")
        ax.plot(plotdata_genome_cluster.index, np.exp(plotdata_genome_cluster["prob"]), color = "black", lw = 1, zorder = 1)
        polygon = ax.fill_between(plotdata_genome_mean.index, np.exp(plotdata_genome_mean["prob"]), np.exp(plotdata_genome_cluster["prob"]), color = color, zorder = 0)

        # up/down gradient
        verts = np.vstack([p.vertices for p in polygon.get_paths()])
        c = plotdata_genome_cluster["prob_diff"].values
        c[c == np.inf] = 0.
        gradient = ax.imshow(
            c.reshape(1, -1),
            cmap='RdBu_r',
            aspect='auto',
            extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()],
            zorder = 25,
            norm = norm_atac_diff
        )
        gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)
        polygon.set_alpha(0)
    
    ax.set_yticks([])
    ax.set_xticks([])
    
    # motifs
    motifs_oi_cluster = motifs_oi.loc[[cluster_ix in clusters for clusters in motifs_oi["clusters"]]]
    motifdata_cluster = motifdata.loc[motifdata["motif"].isin(motifs_oi_cluster.index)]
    for _, z in motifdata_cluster.iterrows():
        ax.axvline(z["position"], color = motifs_oi.loc[z["motif"], "color"], zorder = 100)
        ax.scatter(z["position"], 0, color = motifs_oi.loc[z["motif"], "color"], zorder = 100, clip_on = False)
    
    # expression  
    ax = ax_expression.ax
    sns.despine(ax = ax, left = False, right = True, top = True, bottom = False)
    
    ax.set_yticks([])
    ax.set_xticks([])
    
    plotdata_expression_cluster = plotdata_expression.loc[plotdata_expression["cluster"] == cluster_ix]
    ax.bar([0.], [plotdata_expression_cluster["expression"].mean()], color = "grey")
    ax.set_ylim(*lim_expression)
    
    ax.set_ylabel(f"{cluster_info.loc[cluster_ix]['label']}\n{fragments_oi.sum()} fragments\n {int(latent[:, cluster_ix].sum())} cells", rotation = 0, ha = "right", va = "center")
    
    # motif legend
    n = len(motifs_oi_cluster)
    ax = ax_motiflegend.ax
    ax.axis("off")
    ax.set_xlim(-0.5, 2)
    ax.set_ylim((n+4)/2-4, (n+4)/2)
    
    for i, (motif, motif_info) in enumerate(motifs_oi_cluster.iterrows()):
        ax.scatter([0], [i], color = motif_info["color"])
        ax.text(0.1, i, s = motif_info["label"], ha = "left", va = "center")
        
# peaks
ax_peaks = chd.grid.Ax((panel_width, panel_height))
wrap_peaks.add(ax_peaks)
ax = ax_peaks.ax
ax.set_xlim(*window)
for _, peak in peaks.iterrows():
    y = peak_methods.loc[peak["method"], "ix"]
    rect = mpl.patches.Rectangle((peak["start"], y), peak["end"] - peak["start"], 1, fc = "#333")
    ax.add_patch(rect)
ax.set_ylim(0, peak_methods["ix"].max() + 1)
ax.set_yticks(peak_methods["ix"] + 0.5)
ax.set_yticklabels(peak_methods.index)
ax.set_ylabel("Peaks")

# set x ticks of genome
        
wrap_genome.elements[-1].ax.set_xticks([*window, 0])
wrap_genome.elements[-1].ax.set_xlabel("Distance to TSS")
    
fig.plot()
