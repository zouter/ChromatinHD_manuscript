import chromatinhd.loaders.fragments
import chromatinhd.models
import chromatinhd.models.likelihood.v2
import chromatinhd.models.likelihood.v4
import chromatinhd.models.likelihood.v5
import chromatinhd.models.likelihood.v8
import chromatinhd as chd

import pickle
import numpy as np
import torch

import copy

# n_cells_step = 1000
# n_genes_step = 1000

n_cells_step = 100
n_genes_step = 5000


def get_design(dataset_name, latent_name, fragments):
    folder_root = chd.get_output()
    folder_data = folder_root / "data"
    folder_data_preproc = folder_data / dataset_name
    latent_folder = folder_data_preproc / "latent"
    cell_latent_space = torch.from_numpy(
        pickle.load((latent_folder / (latent_name + ".pkl")).open("rb")).values
    ).to(torch.float)

    general_model_parameters = {"fragments": fragments}

    general_loader_parameters = {
        "fragments": fragments,
        "cellxgene_batch_size": n_cells_step * n_genes_step,
    }

    fragments.create_cut_data()

    design = {}

    import chromatinhd.models.likelihood.v9

    design["v9_128-64-32"] = {
        "model_cls": chromatinhd.models.likelihood.v9.Decoding,
        "model_parameters": {
            **general_model_parameters,
            "latent": cell_latent_space,
            "nbins": (
                128,
                64,
                32,
            ),
        },
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": {
            **{
                k: general_loader_parameters[k]
                for k in ["fragments", "cellxgene_batch_size"]
            }
        },
        "n_epoch": 50,
    }

    import chromatinhd.models.likelihood.v11

    reflatent = torch.eye(cell_latent_space.shape[1]).to(torch.float64)
    reflatent_idx = torch.from_numpy(np.where(cell_latent_space)[1])

    design["v11_128-64-32"] = {
        "model_cls": chromatinhd.models.likelihood.v11.Decoding,
        "model_parameters": {
            **general_model_parameters,
            "latent": cell_latent_space,
            "nbins": (
                128,
                64,
                32,
            ),
        },
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": {
            **{
                k: general_loader_parameters[k]
                for k in ["fragments", "cellxgene_batch_size"]
            }
        },
        "n_epoch": 50,
    }

    return design


import chromatinhd as chd
import chromatinhd.loaders.minibatching


def get_folds_training(fragments, folds):
    for fold in folds:
        minibatcher = chd.loaders.minibatching.Minibatcher(
            fold["cells_train"],
            np.arange(fragments.n_genes),
            fragments.n_genes,
            n_genes_step=n_genes_step,
            n_cells_step=n_cells_step,
        )
        minibatches_train_sets = [
            {
                "tasks": minibatcher.create_minibatches(
                    use_all=True, rg=np.random.RandomState(i), permute_genes=False
                )
            }
            for i in range(10)
        ]
        fold["minibatches_train_sets"] = minibatches_train_sets

        rg = np.random.RandomState(0)
        fold["minibatches_validation"] = chd.loaders.minibatching.create_bins_ordered(
            fold["cells_validation"],
            np.arange(fragments.n_genes),
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            permute_genes=False,
            rg=rg,
        )
        fold["minibatches_validation_trace"] = fold["minibatches_validation"][:8]
    return folds
