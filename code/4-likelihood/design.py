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

    # design["v2"] = {
    #     "model_cls": chromatinhd.models.likelihood.v2.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v2_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v2.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }

    # design["v2_64c"] = {
    #     "model_cls": chromatinhd.models.likelihood.v2.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "n_components": 64,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v2_64c_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v2.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "n_components": 64,
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }

    # # design["v2_64c_2l"] = {
    # #     "model_cls":chromatinhd.models.likelihood.v2.Decoding,
    # #     "model_parameters": {**general_model_parameters, "cell_latent_space":cell_latent_space, "n_components":64, "decoder_n_layers":2},
    # #     "loader_cls":chromatinhd.loaders.fragments.Fragments,
    # #     "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size"]}}
    # # }

    # design["v4"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_256"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (256,),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_64"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (64,),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_64_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (64,),
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_64_1l"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (64,),
    #         "decoder_n_layers": 1,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_64_1l_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (64,),
    #         "decoder_n_layers": 1,
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_64_2l"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (64,),
    #         "decoder_n_layers": 2,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_64_2l_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (64,),
    #         "decoder_n_layers": 2,
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_32"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (32,),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_32_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (32,),
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_16"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (16,),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_16_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (16,),
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_32-16"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (32, 16),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }
    # design["v4_32-16_baseline"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (32, 16),
    #         "baseline": True,
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }

    # design["v4_128-64-32"] = {
    #     "model_cls": chromatinhd.models.likelihood.v4.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "cell_latent_space": cell_latent_space,
    #         "nbins": (
    #             128,
    #             64,
    #             32,
    #         ),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    # }

    # design["v4_256-128-64-32"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v4_256-128-64-32"]["model_parameters"].update({"nbins": (256, 128, 64, 32)})

    # design["v4_64-32"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v4_64-32"]["model_parameters"].update({"nbins": (256, 128, 64, 32)})

    # design["v4_32"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v4_32"]["model_parameters"].update({"nbins": (32,)})

    # design["v4_64"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v4_64"]["model_parameters"].update({"nbins": (64,)})

    # design["v4_128"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v4_128"]["model_parameters"].update({"nbins": (128,)})

    # design["v4_128-64-32_30"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v4_128-64-32_30"]["n_epoch"] = 30

    # design["v4_128-64-32_rep"] = copy.deepcopy(design["v4_128-64-32"])

    # design["v4_128-64-32_30_freescale"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_freescale"]["model_parameters"].update(
    #     {"mixture_delta_p_scale_free": True, "rho_delta_p_scale_free": True}
    # )

    # design["v4_128-64-32_30_freescale_laplace"] = copy.deepcopy(
    #     design["v4_128-64-32_30"]
    # )
    # design["v4_128-64-32_30_freescale_laplace"]["model_parameters"].update(
    #     {
    #         "mixture_delta_p_scale_free": True,
    #         "rho_delta_p_scale_free": True,
    #         "mixture_delta_p_scale_dist": "laplace",
    #     }
    # )

    # design["v4_128-64-32_30_scalelik"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_scalelik"]["model_parameters"].update(
    #     {"scale_likelihood": True}
    # )

    # design["v4_128-64-32_30_freescale_scalelik"] = copy.deepcopy(
    #     design["v4_128-64-32_30"]
    # )
    # design["v4_128-64-32_30_freescale_scalelik"]["model_parameters"].update(
    #     {
    #         "mixture_delta_p_scale_free": True,
    #         "rho_delta_p_scale_free": True,
    #         "scale_likelihood": True,
    #     }
    # )

    # design["v4_128-64-32_30_freescale_scalelik_laplace"] = copy.deepcopy(
    #     design["v4_128-64-32_30"]
    # )
    # design["v4_128-64-32_30_freescale_scalelik_laplace"]["model_parameters"].update(
    #     {
    #         "mixture_delta_p_scale_free": True,
    #         "rho_delta_p_scale_free": True,
    #         "scale_likelihood": True,
    #         "mixture_delta_p_scale_dist": "laplace",
    #     }
    # )

    # design["v4_128-64-32_30_laplace0.05"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_laplace0.05"]["model_parameters"].update(
    #     {
    #         "scale_likelihood": True,
    #         "mixture_delta_p_scale_dist": "laplace",
    #         "mixture_delta_p_scale": 0.05,
    #     }
    # )

    # design["v4_128-64-32_30_laplace0.1"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_laplace0.1"]["model_parameters"].update(
    #     {
    #         "scale_likelihood": True,
    #         "mixture_delta_p_scale_dist": "laplace",
    #         "mixture_delta_p_scale": 0.1,
    #     }
    # )

    # design["v4_128-64-32_30_laplace1.0"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_laplace1.0"]["model_parameters"].update(
    #     {
    #         "scale_likelihood": True,
    #         "mixture_delta_p_scale_dist": "laplace",
    #         "mixture_delta_p_scale": 1.0,
    #     }
    # )

    # design["v4_128-64-32_30_normal0.05"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_normal0.05"]["model_parameters"].update(
    #     {
    #         "scale_likelihood": True,
    #         "mixture_delta_p_scale_dist": "normal",
    #         "mixture_delta_p_scale": 0.05,
    #     }
    # )

    # design["v4_128-64-32_30_rep"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v4_128-64-32_30_rep"]["model_parameters"].update({})

    # design["v5_128-64-32_30_rep"] = copy.deepcopy(design["v4_128-64-32_30"])
    # design["v5_128-64-32_30_rep"][
    #     "model_cls"
    # ] = chromatinhd.models.likelihood.v5.Decoding
    # design["v5_128-64-32_30_rep"]["model_parameters"].update({})

    # design["v5_128-64-32"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v5_128-64-32"]["model_cls"] = chromatinhd.models.likelihood.v5.Decoding
    # design["v5_128-64-32"]["model_parameters"].update({})

    # design["v5_128-64-32"] = copy.deepcopy(design["v4_128-64-32"])
    # design["v5_128-64-32"]["model_cls"] = chromatinhd.models.likelihood.v5.Decoding
    # design["v5_128-64-32"]["model_parameters"].update({})

    # n_latent_dimensions = cell_latent_space.shape[1]
    # print(cell_latent_space.shape)
    # reflatent = torch.eye(n_latent_dimensions).to(torch.float64)
    # reflatent_idx = torch.from_numpy(np.where(cell_latent_space)[1])

    # design["v8_128-64-32"] = {
    #     "model_cls": chromatinhd.models.likelihood.v8.Decoding,
    #     "model_parameters": {
    #         **general_model_parameters,
    #         "reflatent": reflatent,
    #         "reflatent_idx": reflatent_idx,
    #         "nbins": (
    #             128,
    #             64,
    #             32,
    #         ),
    #     },
    #     "loader_cls": chromatinhd.loaders.fragments.Fragments,
    #     "loader_parameters": {
    #         **{
    #             k: general_loader_parameters[k]
    #             for k in ["fragments", "cellxgene_batch_size"]
    #         }
    #     },
    #     "n_epoch": 50,
    # }

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
        # rg = np.random.RandomState(0)
        # fold["minibatches_train"] = chd.loaders.minibatching.create_bins_random(
        #     fold["cells_train"],
        #     list(fold["genes_train"]) + list(fold["genes_validation"]),
        #     n_cells_step=n_cells_step,
        #     n_genes_step=n_genes_step,
        #     n_genes_total=fragments.n_genes,
        #     use_all = True,
        #     rg=rg,
        # )
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
            list(fold["genes_train"]) + list(fold["genes_validation"]),
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            permute_genes=False,
            rg=rg,
        )
        fold["minibatches_validation_trace"] = fold["minibatches_validation"][:8]
    return folds


def get_folds_inference(fragments, folds):
    for fold in folds:
        cells_train = list(fold["cells_train"])[:200]
        genes_train = list(fold["genes_train"])
        cells_validation = list(fold["cells_validation"])
        # genes_validation = list(fold["genes_validation"])

        rg = np.random.RandomState(0)

        minibatches = chd.loaders.minibatching.create_bins_ordered(
            cells_train + cells_validation,
            genes_train,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            rg=rg,
        )
        fold["minibatches"] = minibatches

        fold["phases"] = {
            "train": [cells_train, genes_train],
            "validation": [cells_validation, genes_train],
        }
    return folds


def get_folds_test(fragments, folds):
    for fold in folds:
        cells_test = list(fold["cells_test"])
        genes_test = list(fold["genes_test"])

        rg = np.random.RandomState(0)

        minibatches = chd.loaders.minibatching.create_bins_ordered(
            cells_test,
            genes_test,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            rg=rg,
        )
        fold["minibatches"] = minibatches

        fold["phases"] = {"test": [cells_test, genes_test]}
    return folds
