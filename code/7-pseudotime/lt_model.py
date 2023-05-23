#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import torch
import pickle
import pathlib
import tempfile
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.loaders.fragments
import chromatinhd.models.likelihood_pseudotime.v1 as likelihood_model

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

#%%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

#%%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
folds = pd.read_pickle(folder_data_preproc / "fragments_myeloid" / promoter_name / "folds.pkl")
fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)
fragments.window = window
window_width = window[1] - window[0]

#%%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# ## Create loaders
#%%
fragments.create_cut_data()

#%%
cells_train = folds[0]['cells_train']
cells_validation = folds[0]['cells_validation']

n_cells_step = 100
n_genes_step = 5000

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
minibatches_validation = chd.loaders.minibatching.create_bins_random(
    cells_validation,
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step = n_genes_step,
    n_cells_step=n_cells_step,
    use_all = True,
    permute_genes = False
)
loaders_validation.initialize(minibatches_validation)

#%%
data = loaders_train.pull()

#%%
data.cut_coordinates

#%%
data.cut_coordinates.shape

#%%
data.cut_local_cell_ix

#%%
data.cut_local_cell_ix.shape

#%%
data.cut_local_gene_ix

#%%
data.cut_local_gene_ix.shape

#%%
data.cut_local_cellxgene_ix

#%%
data.cut_local_cellxgene_ix.shape

#%%
data.cut_local_cellxgene_ix.unique()

#%%
data.cut_local_cellxgene_ix.unique().shape

#%%
# torch works by default in float32
latent_name = "latent_time"
latent_folder = folder_data_preproc / "latent"
df = pd.read_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv", index_col = 0)
latent = df['latent_time'].values.astype(np.float32)
latent = torch.from_numpy(latent)

#%%
# ### Create model
nbins = (128, 64, 32, )
model = likelihood_model.Model(fragments, latent, nbins = nbins)

#%%
optimizer = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(autoextend=True), lr = 1e-2)

#%%
# with torch.autograd.detect_anomaly():
model = model.to("cpu").train()
loaders_train.restart()
loaders_validation.restart()
trainer = chd.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 50, checkpoint_every_epoch=1, optimize_every_step = 1)
trainer.train()

# model = model.to("cpu").train()
# model.forward(data)

#%%
gene_name = 'MPO'
gene_ix = transcriptome.gene_ix(gene_name) # double check if this is the correct gene_ix

#%%
cumulative_sum_list = [0] + [sum(nbins[:i+1]) for i in range(len(nbins))]

fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # create a 3x1 grid of subplots

for i in range(len(nbins)):
    x = cumulative_sum_list[i]
    y = cumulative_sum_list[i+1]
    print(x, y)
    heatmap = axs[i].pcolor(model.decoder.delta_height_slope.data[:, x:y], cmap=plt.cm.RdBu)

fig.colorbar(heatmap, ax=axs)  # add a colorbar for each subplot
plt.savefig(folder_data_preproc / "plots" / ("test" + ".pdf"))


#%%
cumulative_sum_list = [0] + [sum(nbins[:i+1]) for i in range(len(nbins))]

fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # create a 3x1 grid of subplots

for i in range(len(nbins)):
    x = cumulative_sum_list[i]
    y = cumulative_sum_list[i+1]
    print(x, y)
    heatmap = axs[i].pcolor(torch.abs(model.decoder.delta_height_slope.data[:, x:y]), cmap=plt.cm.Blues)

fig.colorbar(heatmap, ax=axs)  # add a colorbar for each subplot
plt.savefig(folder_data_preproc / "plots" / ("test2" + ".pdf"))

#%%
cumulative_sum_list = [0] + [sum(nbins[:i+1]) for i in range(len(nbins))]
for i in range(len(nbins)):
    x = cumulative_sum_list[i]
    y = cumulative_sum_list[i+1]
    print(x, y)
    plt.figure()
    plt.plot(torch.abs(model.decoder.delta_height_slope.data[:, x:y]).mean(0))
    plt.savefig(folder_data_preproc / "plots" / ("delta_height_slope_" + str(nbins[i]) + ".pdf"))
plt.show() 

#%%
trainer.trace.plot()
pd.DataFrame(trainer.trace.train_steps).groupby("epoch").mean()["loss"].plot()
pd.DataFrame(trainer.trace.validation_steps).groupby("epoch").mean()["loss"].plot()

# %%
