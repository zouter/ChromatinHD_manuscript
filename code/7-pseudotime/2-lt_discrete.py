#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import gc
import torch
import torch_scatter
import pickle
import pathlib
import tempfile
import scipy.stats
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm.auto as tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set_style('ticks')

import chromatinhd as chd
import chromatinhd.loaders.fragments
import chromatinhd.models.likelihood.v9 as vae_model

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc_backup"
promoter_name, window = "10k10k", np.array([-10000, 10000])

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

df = pd.read_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv", index_col = 0)
df['quantile'] = pd.qcut(df['latent_time'], q=10, labels=False)
latent = pd.get_dummies(df['quantile'], prefix='quantile')
latent_torch = torch.from_numpy(latent.values).to(torch.float)

cluster_info = pd.DataFrame()
cluster_info['cluster'] = list(latent.columns)
cluster_info['label'] = list(latent.columns)
cluster_info['dimension'] = range(latent.shape[-1])
cluster_info["color"] = sns.color_palette("husl", latent.shape[1])
cluster_info.set_index('cluster', inplace=True)

fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)
fragments.window = window
fragments.obs.index.name = "cell"
fragments.obs["cluster"] = df['quantile'] # order of cells/barcodes is identical
fragments.create_cut_data()

# delete transcriptome object eventually
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
transcriptome.var.index = transcriptome.var["Accession"]
transcriptome.var.index.name = "gene"

folds = pd.read_pickle(folder_data_preproc / "fragments_myeloid" / promoter_name / "folds.pkl")
cells_train = folds[0]['cells_train']
cells_validation = folds[0]['cells_validation']

# %% 
n_cells_step = 100
n_genes_step = 50

loaders_train = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments": fragments, "cellxgene_batch_size": n_cells_step * n_genes_step},
    n_workers = 20,
    shuffle_on_iter = True
)
minibatches_train = chd.loaders.minibatching.create_bins_random(
    cells_train,
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step = n_genes_step,
    n_cells_step = n_cells_step,
    use_all = True,
    permute_genes = False
)
loaders_train.initialize(minibatches_train)

loaders_validation = chromatinhd.loaders.pool.LoaderPool(
    chromatinhd.loaders.fragments.Fragments,
    {"fragments": fragments, "cellxgene_batch_size": n_cells_step * n_genes_step},
    n_workers = 5
)
minibatches_validation = chd.loaders.minibatching.create_bins_random(
    cells_validation,
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step = n_genes_step,
    n_cells_step = n_cells_step,
    use_all = True,
    permute_genes = False
)
loaders_validation.initialize(minibatches_validation)

# TODO move this to separate script
# latent_name = "celltype"
# latent_folder = folder_data_preproc / "latent"
# latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
# latent_torch = torch.from_numpy(latent.values).to(torch.float)
# latent.shape[-1] = latent.shape[-1]
# cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
# cluster_info["color"] = sns.color_palette("husl", latent.shape[1])
# fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vae_model.Decoding(fragments, torch.from_numpy(latent.values), nbins = (128, 64, 32, ))
model = model.to(device)
model.n_genes = fragments.n_genes

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

nbins = np.array(model.mixture.transform.nbins)
bincuts = np.concatenate([[0], np.cumsum(nbins)])
binmids = bincuts[:-1] + nbins/2

gene_oi = 0
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

ax = main[1, 0] = chd.grid.Ax(dim = (10, latent.shape[-1] * 0.25))
ax = ax.ax
plotdata = (model.decoder.logit_weight.data[gene_oi].cpu().numpy())
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

# %%
# ### Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(autoextend=True), lr = 1e-2)
loaders_train.restart()
loaders_validation.restart()

gc.collect()
torch.cuda.empty_cache()

# %%
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
        self.likelihood_mixture_checkpoint[data.genes_oi] += torch_scatter.scatter_sum(model.track["likelihood"], data.cut_local_gene_ix, dim_size = data.n_genes).detach().cpu().numpy()
        
    def finish(self):
        self.likelihood_mixture.append(self.likelihood_mixture_checkpoint)
        
hook_genelikelihood = GeneLikelihoodHook(fragments.n_genes)
hooks = [hook_genelikelihood]

# %%
model = model.to(device).train()
loaders_train.restart()
loaders_validation.restart()
trainer = chd.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 50, checkpoint_every_epoch=1, optimize_every_step = 1, hooks_checkpoint = hooks)
trainer.train()

# %%
likelihood_mixture = pd.DataFrame(np.vstack(hook_genelikelihood.likelihood_mixture), columns = fragments.var.index).T
scores = (likelihood_mixture.iloc[:, -1] - likelihood_mixture[0]).sort_values().to_frame("lr")
scores["label"] = transcriptome.var.loc[scores.index]['symbol']

# %%
pickle.dump(model.to("cpu"), open("./2-lt_discrete.pkl", "wb"))

#%%
# def plot_distribution(latent, latent_torch, cluster_info, fragments, gene_oi, model, device, dir_plots):
#     fig, axes = plt.subplots(latent.shape[1], 1, figsize=(20, 1*latent.shape[1]), sharex = True, sharey = True) #

#     probs = [] #
#     pseudocoordinates = torch.linspace(0, 1, 1000).to(device) #
#     bins = np.linspace(0, 1, 500) #
#     binmids = (bins[1:] + bins[:-1])/2 #
#     binsize = binmids[1] - binmids[0] #
#     fragments_oi_all = (fragments.cut_local_gene_ix == gene_oi) #

#     gene_id = fragments.var.index[gene_oi] #

#     for i, ax in zip(range(latent.shape[1]), axes): #
        
#         n_cells = latent_torch[:, i].sum() #
#         fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi) #
        
#         bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins) #
#         ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0) #

#         pseudolatent = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device) #
#         pseudolatent[:, i] = 1. #
#         prob = model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_oi = gene_oi) #
    
#         ax.plot(pseudocoordinates.cpu().numpy(), np.exp(prob), label = i, color = "#0000FF", lw = 2, zorder = 20) #
#         ax.plot(pseudocoordinates.cpu().numpy(), np.exp(prob), label = i, color = "#FFFFFF", lw = 3, zorder = 10) #
        
#         ax.set_ylabel(f"{cluster_info.iloc[i]['label']}\n freq={fragments_oi.sum()/n_cells}", rotation = 0, ha = "right", va = "center") #
        
#         probs.append(prob) #

#     plt.savefig(dir_plots / (gene_id + ".png")) #

#     probs = np.stack(probs) #
#     return probs #

# def plot_pseudo_quantile(probs, latent_torch, cluster_info, gene_oi, fragments, dir_plots):
#     gene_id = fragments.var.index[gene_oi]
#     fragments_oi_all = (fragments.cut_local_gene_ix == gene_oi)

#     bins = np.linspace(0, 1, 500)
#     binmids = (bins[1:] + bins[:-1])/2
#     binsize = binmids[1] - binmids[0]
#     pseudocoordinates = torch.linspace(0, 1, 1000)

#     fig, axes = plt.subplots(probs.shape[0], 1, figsize=(20, 1*probs.shape[0]), sharex = True, sharey = True)
#     for i, ax in zip(range(probs.shape[0]), axes):
#         n_cells = latent_torch[:, i].sum()
#         fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi)
#         bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins)

#         ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0)
#         ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#0000FF", lw = 2, zorder = 20)
#         ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#FFFFFF", lw = 3, zorder = 10)
#         ax.set_ylabel(f"{cluster_info.iloc[i]['label']}\n freq={fragments_oi.sum()/n_cells}", rotation = 0, ha = "right", va = "center")

#     plt.savefig(dir_plots / (gene_id + ".png"))

def evaluate_pseudo_quantile(latent, gene_oi, model, device):
    pseudocoordinates = torch.linspace(0, 1, 1000).to(device)
    probs = []
    for i in range(latent.shape[1]):
        pseudolatent = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device)
        pseudolatent[:, i] = 1.
        prob = model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_oi = gene_oi)
        probs.append(prob)
    probs = np.stack(probs)
    return probs

# %%
model = pickle.load(open("models/2-lt_discrete.pkl", "rb"))

# %%
# ## Inference single gene
sns.histplot(model.decoder.rho_weight.weight.data.numpy().flatten())
z = model.decoder.logit_weight.weight.data.numpy().flatten()
sns.histplot(z[:100])
scipy.stats.laplace.fit(z)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

#%%
model = model.to(device)

# %%
dir_csv = folder_data_preproc / 'likelihood_quantile_myeloid'
os.makedirs(dir_csv, exist_ok=True)

pseudocoordinates = torch.linspace(0, 1, 1000).to(device)

for gene_oi in range(len(promoters)):
    print(gene_oi)
    gene_id = fragments.var.index[gene_oi]
    probs = evaluate_pseudo_quantile(latent, gene_oi, model, device)
    probs_df = pd.DataFrame(np.exp(probs), columns = pseudocoordinates.tolist(), index = cluster_info.index)
    probs_df.to_csv(dir_csv / f"{gene_id}.csv")

print("Done \n")
# %%
# sns.heatmap(probs, cmap = mpl.cm.RdBu_r)
# probs_diff = probs - probs.mean(0, keepdims = True)
# sns.heatmap(probs_diff, cmap = mpl.cm.RdBu_r, center = 0.)

# # %%
# plt.figure(figsize=(8, 8))
# heatmap = plt.pcolor(probs_df, cmap='YlOrRd')
# plt.colorbar(heatmap)
# plt.xticks([0, len(probs_df.columns)-1], ['0', '1'])
# plt.yticks(np.arange(len(probs_df.index)) + 0.5, probs_df.index)
# plt.show()
