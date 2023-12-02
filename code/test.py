import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

torch.use_deterministic_algorithms(True)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd

# dataset_name = "pbmc10k"
dataset_name = "pbmc10k/subsets/top5"
dataset_name = "pbmc10k/subsets/top1"
# dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "e18brain"
regions_name = "100k100k"
# regions_name = "10k10k"

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

torch.manual_seed(1)

import chromatinhd.models.pred.model.better

model_params2 = dict(
    n_embedding_dimensions=100,
    n_layers_fragment_embedder=5,
    residual_fragment_embedder=False,
    n_layers_embedding2expression=5,
    residual_embedding2expression=False,
    dropout_rate_fragment_embedder=0.0,
    dropout_rate_embedding2expression=0.0,
    encoder="spline_binary",
)
train_params2 = dict(
    weight_decay=1e-1,
    lr=1e-4,
)

gene_oi = fragments.var.index[0]

for i in range(10):
    model2 = chd.models.pred.model.better.Model(
        fragments=fragments, transcriptome=transcriptome, fold=fold, layer="magic", region_oi=gene_oi, **model_params2
    )

    model_params = dict(
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="multi_spline_binary",
        # distance_encoder="split",
        # library_size_encoder="linear",
    )
    train_params = dict(
        # optimizer = "sgd",
        optimizer="adam",
        weight_decay=1e-1,
        # lr = 1e-2,
        # lr = 1e-3,
        lr=1e-4,
    )

    import chromatinhd.models.pred.model.binary

    model = chd.models.pred.model.binary.Model(
        fragments=fragments, transcriptome=transcriptome, fold=fold, layer="magic", **model_params
    )

    model2.fragment_embedder.encoder.w.data = torch.rand_like(input=model2.fragment_embedder.encoder.w.data)
    model.fragment_embedder.encoder.w[0].data = model2.fragment_embedder.encoder.w.data

    model.fragment_embedder.nn0[0].weight[0].data = model2.fragment_embedder.nn0[0].weight.data.T
    model.fragment_embedder.nn1[0].weight[0].data = model2.fragment_embedder.nn1[0].weight.data.T
    model.fragment_embedder.nn2[0].weight[0].data = model2.fragment_embedder.nn2[0].weight.data.T
    model.fragment_embedder.nn3[0].weight[0].data = model2.fragment_embedder.nn3[0].weight.data.T
    model.fragment_embedder.nn4[0].weight[0].data = model2.fragment_embedder.nn4[0].weight.data.T

    model.fragment_embedder.nn0[0].bias[0].data = model2.fragment_embedder.nn0[0].bias.data
    model.fragment_embedder.nn1[0].bias[0].data = model2.fragment_embedder.nn1[0].bias.data
    model.fragment_embedder.nn2[0].bias[0].data = model2.fragment_embedder.nn2[0].bias.data
    model.fragment_embedder.nn3[0].bias[0].data = model2.fragment_embedder.nn3[0].bias.data
    model.fragment_embedder.nn4[0].bias[0].data = model2.fragment_embedder.nn4[0].bias.data

    model.embedding_to_expression.nn0[0].weight[0].data = model2.embedding_to_expression.nn0[0].weight.data.T
    model.embedding_to_expression.nn1[0].weight[0].data = model2.embedding_to_expression.nn1[0].weight.data.T
    model.embedding_to_expression.nn2[0].weight[0].data = model2.embedding_to_expression.nn2[0].weight.data.T
    model.embedding_to_expression.nn3[0].weight[0].data = model2.embedding_to_expression.nn3[0].weight.data.T
    model.embedding_to_expression.nn4[0].weight[0].data = model2.embedding_to_expression.nn4[0].weight.data.T
    model.embedding_to_expression.final.weight[0].data = model2.embedding_to_expression.final.weight.data.T

    model.embedding_to_expression.nn0[0].bias[0].data = model2.embedding_to_expression.nn0[0].bias.data
    model.embedding_to_expression.nn1[0].bias[0].data = model2.embedding_to_expression.nn1[0].bias.data
    model.embedding_to_expression.nn2[0].bias[0].data = model2.embedding_to_expression.nn2[0].bias.data
    model.embedding_to_expression.nn3[0].bias[0].data = model2.embedding_to_expression.nn3[0].bias.data
    model.embedding_to_expression.nn4[0].bias[0].data = model2.embedding_to_expression.nn4[0].bias.data

    # model.embedding_to_expression.final.weight[0].data[:] = 0.1
    # model2.embedding_to_expression.final.weight.data[:] = 0.1

    model.train_model(
        **train_params, warmup_epochs=0, n_epochs=500, early_stopping=True, n_regions_step=1, device="cuda"
    )
    model2.train_model(**train_params2, n_epochs=500, early_stopping=True, device="cuda")

    prediction1 = model.get_prediction(cell_ixs=fold["cells_test"]).sel(gene=gene_oi)
    prediction2 = model2.get_prediction(cell_ixs=fold["cells_test"]).sel(gene=gene_oi)
    print(
        "test ",
        np.corrcoef(prediction1["predicted"], prediction1["expected"])[0, 1],
        np.corrcoef(prediction2["predicted"], prediction2["expected"])[0, 1],
    )

    prediction1 = model.get_prediction().sel(gene=gene_oi)
    prediction2 = model2.get_prediction().sel(gene=gene_oi)
    print(
        "train",
        np.corrcoef(prediction1["predicted"], prediction1["expected"])[0, 1],
        np.corrcoef(prediction2["predicted"], prediction2["expected"])[0, 1],
    )
