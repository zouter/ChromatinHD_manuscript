#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import torch
import torch_scatter
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('ticks')

#%%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

#%%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)
fragments.window = window
fragments.create_cut_data()

folds = pd.read_pickle(folder_data_preproc / "fragments_myeloid" / promoter_name / "folds.pkl")

# torch works by default in float32
df_latent = pd.read_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv", index_col = 0)
latent = torch.tensor(df_latent['latent_time'].values.astype(np.float32))

#%%
n_cells_step = 100
n_genes_step = 5000

#%%
for index, fold in enumerate(folds):

    loaders_train = chromatinhd.loaders.pool.LoaderPool(
        chromatinhd.loaders.fragments.Fragments,
        {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
        n_workers = 20,
        shuffle_on_iter = True
    )
    minibatches_train = chd.loaders.minibatching.create_bins_random(
        fold['cells_train'], #TODO: change this
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
        fold['cells_validation'], #TODO: change this
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
    # print(data.cut_coordinates)
    # print(data.cut_coordinates.shape)
    # print(data.cut_local_cell_ix)
    # print(data.cut_local_cell_ix.shape)
    # print(data.cut_local_gene_ix)
    # print(data.cut_local_gene_ix.shape)
    # print(data.cut_local_cellxgene_ix)
    # print(data.cut_local_cellxgene_ix.shape)
    # print(data.cut_local_cellxgene_ix.unique())
    # print(data.cut_local_cellxgene_ix.unique().shape)
    # print(torch.bincount(data.cut_local_cell_ix))
    # print(torch.bincount(data.cut_local_gene_ix))
    # print(torch.bincount(data.cut_local_cellxgene_ix))
    # print(data.cut_local_cell_ix.unique().shape)

    #%%
    nbins = (256, )
    nbins = (128, 64, 32, )
    nbins = (128, )
    model_name = f"{'_'.join(str(n) for n in nbins)}_fold_{index}" #TODO: include fold number

    #%%
    model = likelihood_model.Model(fragments, latent, nbins = nbins)
    
    # %%
    model.forward(data)

    #%%
    optimizer = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(autoextend=True), lr = 1e-2)

    #%%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()
    loaders_train.restart()
    loaders_validation.restart()
    trainer = chd.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 50, checkpoint_every_epoch=1, optimize_every_step = 1)
    trainer.train()

    pickle.dump(model.to("cpu"), open(f"./3-lt_continuous_{model_name}.pkl", "wb"))

    #%%
    likelihood_per_gene = torch.zeros(fragments.n_genes)
    for data in loaders_validation:
        with torch.no_grad():
            model.forward(data)
        loaders_validation.submit_next()

        cut_gene_ix = data.genes_oi_torch[data.cut_local_gene_ix]
        torch_scatter.scatter_add(model.track["likelihood"], cut_gene_ix, out = likelihood_per_gene).detach().cpu()

    np.savetxt(folder_data_preproc / f'3-lt_continuous_{model_name}_likelihood_per_gene.csv', likelihood_per_gene.numpy(), delimiter=',')

print("Training complete, models saved")

#%%