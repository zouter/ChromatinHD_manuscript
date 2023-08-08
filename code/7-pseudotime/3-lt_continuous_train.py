#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import sys
import ast
import torch
import pickle
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import chromatinhd as chd
import chromatinhd.loaders.fragments

#%%
print('Start of 3-lt_continuous_train.py')

if len(sys.argv) > 2 and sys.argv[2] == 'external':
    locals().update(ast.literal_eval(sys.argv[1]))
    external = True
else:
    dataset_name_sub = "MV2"
    dataset_name = "myeloid"
    nbins = (128, 64, 32, )
    model_type = 'sigmoid'
    external = False

if model_type == 'linear':
    import chromatinhd.models.likelihood_pseudotime.v1 as likelihood_model
elif model_type == 'sigmoid':
    import chromatinhd.models.likelihood_pseudotime.v2 as likelihood_model

folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

fragment_dir = folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}"
df_latent_file = folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv"

#%%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

fragments = chd.data.Fragments(fragment_dir / promoter_name)
fragments.window = window
fragments.create_cut_data()

folds = pd.read_pickle(fragment_dir / promoter_name / "folds.pkl")

# torch works by default in float32
df_latent = pd.read_csv(df_latent_file, index_col = 0)
df_latent['pr'] = (df_latent['latent_time'].rank() - 1) / (len(df_latent) - 1)
latent = torch.tensor(df_latent['pr'].values.astype(np.float32))

#%%
n_cells_step = 100
n_genes_step = 5000

#%%
for index, fold in enumerate(folds):
    print(f"Training fold {index}")
    model_name = f"{dataset_name_sub}_{dataset_name}_{model_type}_{'_'.join(str(n) for n in nbins)}_fold_{index}"
    model_name_pickle = folder_data_preproc / f"models/{model_name}.pkl"
    print(model_name_pickle)

    loaders_train = chromatinhd.loaders.pool.LoaderPool(
        chromatinhd.loaders.fragments.Fragments,
        {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
        n_workers = 20,
        shuffle_on_iter = True
    )
    minibatches_train = chd.loaders.minibatching.create_bins_random(
        fold['cells_train'],
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
        fold['cells_validation'],
        np.arange(fragments.n_genes),
        fragments.n_genes,
        n_genes_step = n_genes_step,
        n_cells_step=n_cells_step,
        use_all = True,
        permute_genes = False
    )
    loaders_validation.initialize(minibatches_validation)

    data = loaders_train.pull()

    model = likelihood_model.Model(fragments, latent, nbins = nbins)
    model.forward(data)

    optimizer = chd.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(autoextend=True), lr = 1e-2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()
    loaders_train.restart()
    loaders_validation.restart()
    trainer = chd.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 50, checkpoint_every_epoch=1, optimize_every_step = 1)
    trainer.train()

    pickle.dump(model.to("cpu"), open(model_name_pickle, "wb"))
    
    # likelihood_per_gene = torch.zeros(fragments.n_genes)
    # for data in loaders_validation:
    #     with torch.no_grad():
    #         model.forward(data)
    #     loaders_validation.submit_next()

    #     cut_gene_ix = data.genes_oi_torch[data.cut_local_gene_ix]
    #     torch_scatter.scatter_add(model.track["likelihood"], cut_gene_ix, out = likelihood_per_gene).detach().cpu() # check dict key

    # np.savetxt(folder_data_preproc / f'3-lt_continuous_{model_name}_likelihood_per_gene.csv', likelihood_per_gene.numpy(), delimiter=',')

print('End of 3-lt_continuous_train.py')

#%%
# trainer.trace.plot()