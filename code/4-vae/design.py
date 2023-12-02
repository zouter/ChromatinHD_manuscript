import chromatinhd.loaders.fragments
import chromatinhd.loaders.fragmentmotif
import chromatinhd.models
import chromatinhd.models.vae.v4
import chromatinhd.models.vae.v5
import chromatinhd.models.vae.v6

import pickle
import copy
import numpy as np

n_cells_step = 200
n_regions_step = 5000


def get_design(fragments):
    general_model_parameters = {"fragments": fragments}

    general_loader_parameters = {
        "fragments": fragments,
        "cellxregion_batch_size": n_cells_step * n_regions_step,
    }

    design = {}
    design["v4"] = {
        "model_cls": chromatinhd.models.vae.v4.VAE,
        "model_parameters": {**general_model_parameters},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": {**{k: general_loader_parameters[k] for k in ["fragments", "cellxregion_batch_size"]}},
    }
    design["v4_baseline"] = {
        "model_cls": chromatinhd.models.vae.v4.VAE,
        "model_parameters": {**general_model_parameters, "baseline": True},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": {**{k: general_loader_parameters[k] for k in ["fragments", "cellxregion_batch_size"]}},
    }
    design["v4_1freq"] = copy.deepcopy(design["v4"])
    design["v4_1freq"]["model_parameters"].update({})

    design["v5"] = {
        "model_cls": chromatinhd.models.vae.v5.VAE,
        "model_parameters": {**general_model_parameters},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": {**{k: general_loader_parameters[k] for k in ["fragments", "cellxregion_batch_size"]}},
    }

    design["v5_baseline"] = copy.deepcopy(design["v5"])
    design["v5_baseline"]["model_parameters"].update({"baseline": True})

    design["v5_1decoder"] = copy.deepcopy(design["v5"])
    design["v5_1decoder"]["model_parameters"].update({"decoder_n_layers": 1})

    design["v5_32"] = copy.deepcopy(design["v5"])
    design["v5_32"]["model_parameters"].update({"n_encoder_bins": 32, "n_decoder_bins": (32,)})
    design["v5_8"] = copy.deepcopy(design["v5"])
    design["v5_8"]["model_parameters"].update({"n_encoder_bins": 8, "n_decoder_bins": (8,)})

    design["v5_norescale"] = copy.deepcopy(design["v5"])
    design["v5_norescale"]["model_parameters"].update({"rescale": False})

    design["v5_encoder32"] = copy.deepcopy(design["v5"])
    design["v5_encoder32"]["model_parameters"].update({"encoder_n_hidden_dimensions": 32})
    design["v5_norescale"]["model_parameters"].update({"rescale": False})

    design["v5_regularizefragmentcounts"] = copy.deepcopy(design["v5"])
    design["v5_regularizefragmentcounts"]["model_parameters"].update({"regularize_fragmentcounts": True})

    design["v5_regularizefragmentcounts_400epoch"] = copy.deepcopy(design["v5"])
    design["v5_regularizefragmentcounts_400epoch"]["model_parameters"].update({"regularize_fragmentcounts": True})
    design["v5_regularizefragmentcounts_400epoch"]["n_epochs"] = 400

    design["v6"] = {
        "model_cls": chromatinhd.models.vae.v6.VAE,
        "model_parameters": {**general_model_parameters},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": {**{k: general_loader_parameters[k] for k in ["fragments", "cellxregion_batch_size"]}},
    }

    design["v5_s0.8"] = copy.deepcopy(design["v5"])
    design["v5_s0.8"]["model_parameters"].update({"fragments_scale": 0.8})
    design["v5_s0.5"] = copy.deepcopy(design["v5"])
    design["v5_s0.5"]["model_parameters"].update({"fragments_scale": 0.5})
    design["v5_s0.3"] = copy.deepcopy(design["v5"])
    design["v5_s0.3"]["model_parameters"].update({"fragments_scale": 0.3})

    design["v5_mixtureautoscale"] = copy.deepcopy(design["v5"])
    design["v5_mixtureautoscale"]["model_parameters"].update({"mixtureautoscale": True})
    design["v5_s0.3"]["model_parameters"].update({"fragments_scale": 0.3})

    design["v5_mixturescale0.1"] = copy.deepcopy(design["v5"])
    design["v5_mixturescale0.1"]["model_parameters"].update({"mixturescale": 0.1})
    design["v5_s0.3"]["model_parameters"].update({"fragments_scale": 0.3})

    design["v5_mixturelaplace"] = copy.deepcopy(design["v5"])
    design["v5_mixturelaplace"]["model_parameters"].update({"mixturedistribution": "laplace"})

    design["v5_2encoder"] = copy.deepcopy(design["v5"])
    design["v5_2encoder"]["model_parameters"].update({"encoder_n_layers": 2})

    design["pca_50"] = {
        "model_cls": chromatinhd.models.vae.peakcount_pca.Model,
        "model_parameters": {},
        "loader_cls": chromatinhd.loaders.peakcounts.Peakcounts,
        "loader_parameters": {},
    }

    return design


import chromatinhd.models.vae.peakcount_pca


def get_design_peakcount(fragments, peakcounts):
    design = {}
    design["pca_50"] = {
        "model_cls": chromatinhd.models.vae.peakcount_pca.Model,
        "model_parameters": {},
        "loader_cls": chromatinhd.loaders.peakcounts.Peakcounts,
        "loader_parameters": {"fragments": fragments, "peakcounts": peakcounts},
    }
    design["pca_20"] = copy.deepcopy(design["pca_50"])
    design["pca_20"]["model_parameters"].update({"n_components": 20})

    design["pca_200"] = copy.deepcopy(design["pca_50"])
    design["pca_200"]["model_parameters"].update({"n_components": 200})

    design["pca_5"] = copy.deepcopy(design["pca_50"])
    design["pca_5"]["model_parameters"].update({"n_components": 5})

    return design


import chromatinhd as chd
import chromatinhd.loaders.minibatches


def get_folds_training(fragments, folds):
    for fold in folds:
        # train
        rg = np.random.RandomState(0)
        minibatcher = chd.loaders.minibatching.Minibatcher(
            fold["cells_train"],
            np.arange(fragments.n_regions),
            fragments.n_genes,
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
        )
        minibatches_train_sets = [
            {"tasks": minibatcher.create_minibatches(use_all=True, rg=np.random.RandomState(i))} for i in range(10)
        ]
        fold["minibatches_train_sets"] = minibatches_train_sets

        # validation
        fold["minibatches_validation"] = chd.loaders.minibatching.create_bins_ordered(
            fold["cells_validation"],
            np.arange(fragments.n_regions),
            n_cells_step=n_cells_step,
            n_regions_step=n_regions_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            permute_regions=False,
            rg=rg,
        )
        fold["minibatches_validation_trace"] = fold["minibatches_validation"][:8]

        # all
        minibatches_all = chd.loaders.minibatching.create_bins_ordered(
            np.arange(fragments.n_cells),
            np.arange(fragments.n_regions),
            fragments.n_genes,
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
            use_all=True,
            permute_regions=False,
            permute_cells=False,
        )
        fold["minibatches_all"] = minibatches_all
    return folds


def get_folds_inference(fragments, folds):
    for fold in folds[:1]:
        cells_train = list(fold["cells_train"])
        genes_train = list(fold["genes_train"])
        cells_validation = list(fold["cells_validation"])

        minibatches_all = chd.loaders.minibatching.create_bins_ordered(
            np.arange(fragments.n_cells),
            np.arange(fragments.n_regions),
            fragments.n_genes,
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
            use_all=True,
            permute_regions=False,
        )
        fold["minibatches"] = minibatches_all

        fold["phases"] = {
            "train": [cells_train, genes_train],
            "validation": [cells_validation, genes_train],
            "all": [cells_train + cells_validation, genes_train],
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
            n_regions_step=n_regions_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            permute_regions=False,
            rg=rg,
        )
        fold["minibatches"] = minibatches

        fold["phases"] = {"test": [cells_test, genes_test]}
    return folds
