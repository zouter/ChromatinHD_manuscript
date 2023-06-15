import chromatinhd.loaders.fragments
import chromatinhd.loaders.fragmentmotif
import chromatinhd.models
import chromatinhd.models.positional.counter
import chromatinhd.models.positional.v1
import chromatinhd.models.positional.v14
import chromatinhd.models.positional.v15
import chromatinhd.models.positional.v20
import chromatinhd.models.positional.v21

import pickle
import numpy as np

n_cells_step = 200
n_genes_step = 500


def get_design(transcriptome, fragments):
    transcriptome_X_dense = transcriptome.X.dense()
    general_model_parameters = {
        "mean_gene_expression": transcriptome_X_dense.mean(0),
        "n_genes": fragments.n_genes,
    }

    general_loader_parameters = {
        "fragments": fragments,
        "cellxgene_batch_size": n_cells_step * n_genes_step,
    }

    design = {}
    design["counter"] = {
        "model_cls": chromatinhd.models.positional.counter.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["counter_10k"] = {
        "model_cls": chromatinhd.models.positional.counter.Model,
        "model_parameters": {
            **general_model_parameters,
            "window": np.array([0, 10000]),
        },
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["counter_binary"] = {
        "model_cls": chromatinhd.models.positional.counter.Model,
        "model_parameters": {**general_model_parameters, "reduce": "mean"},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["v20"] = {
        "model_cls": chromatinhd.models.positional.v20.Model,
        "model_parameters": {
            **general_model_parameters,
        },
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["v20_initdefault"] = {
        "model_cls": chromatinhd.models.positional.v20.Model,
        "model_parameters": {
            **general_model_parameters,
            "embedding_to_expression_initialization": "default",
        },
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["v21"] = {
        "model_cls": chromatinhd.models.positional.v21.Model,
        "model_parameters": {
            **general_model_parameters,
        },
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
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
                    use_all=True, rg=np.random.RandomState(i)
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
            rg=rg,
        )
        fold["minibatches_validation_trace"] = fold["minibatches_validation"]
    return folds


def get_folds_inference(
    fragments,
    folds,
    n_cells_step=5000,
    n_genes_step=100,
    genes_oi=None,
):
    for fold in folds:
        cells = []
        fold["phases"] = {}

        if genes_oi is None:
            genes_all = np.arange(fragments.n_genes)
        else:
            genes_all = np.arange(fragments.n_genes)[genes_oi]
        fold["genes_all"] = genes_all
        fold["gene_ix_mapper"] = np.zeros(fragments.n_genes, dtype=int)
        fold["gene_ix_mapper"][genes_all] = np.arange(len(genes_all))

        if "cells_train" in fold:
            cells_train = list(fold["cells_train"])[:500]
            fold["phases"]["train"] = [cells_train, genes_all]
            cells.extend(cells_train)

        if "cells_validation" in fold:
            cells_validation = list(fold["cells_validation"])
            fold["phases"]["validation"] = [cells_validation, genes_all]
            cells.extend(cells_validation)

        if "cells_test" in fold:
            cells_test = list(fold["cells_test"])
            fold["phases"]["test"] = [cells_test, genes_all]
            cells.extend(cells_test)

        fold["cells_all"] = cells
        fold["cell_ix_mapper"] = np.zeros(fragments.n_cells, dtype=int)
        fold["cell_ix_mapper"][cells] = np.arange(len(cells))

        rg = np.random.RandomState(0)

        minibatches = chd.loaders.minibatching.create_bins_ordered(
            cells,
            genes_all,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all=True,
            rg=rg,
        )
        fold["minibatches"] = minibatches
    return folds, n_cells_step * n_genes_step


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
