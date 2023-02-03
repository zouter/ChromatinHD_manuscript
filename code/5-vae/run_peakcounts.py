import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data
import chromatinhd.loaders.fragmentmotif
import chromatinhd.loaders.minibatching

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"

import chromatinhd.peakcounts

import pickle
import numpy as np

n_cells_step = 200
n_genes_step = 5000


class Prediction(chd.flow.Flow):
    pass


for dataset_name in [
    "e18brain",
    "lymphoma",
    "pbmc10k",
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    transcriptome = chromatinhd.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )

    fragments = chromatinhd.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )
    fragments.window = window
    fragments.create_cut_data()

    for peaks_name in ["cellranger", "macs2", "stack"]:
        peakcounts = chd.peakcounts.FullPeak(
            folder=chd.get_output() / "peakcounts" / dataset_name / peaks_name
        )

        # create design to run
        from design import get_design_peakcount

        design = get_design_peakcount(fragments, peakcounts)
        design = {
            k: design[k]
            for k in [
                # "pca_50",
                # "pca_20",
                # "pca_200",
                "pca_5",
            ]
        }
        fold_slice = slice(0, 1)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))

        for prediction_name, design_row in design.items():
            print(f"{dataset_name=} {promoter_name=} {peaks_name=} {prediction_name=}")
            prediction = Prediction(
                chd.get_output()
                / "prediction_vae"
                / dataset_name
                / promoter_name
                / peaks_name
                / prediction_name
            )

            loader_train = design_row["loader_cls"](**design_row["loader_parameters"])

            models = []
            for fold_ix, fold in [
                (fold_ix, fold) for fold_ix, fold in enumerate(folds)
            ][fold_slice]:
                # model
                model = design_row["model_cls"](**design_row["model_parameters"])

                cells_oi = fold["cells_train"]
                genes_oi = np.arange(fragments.n_genes)
                cellxgene_oi = (
                    cells_oi[:, None] * fragments.n_genes + genes_oi
                ).flatten()

                minibatch = chd.loaders.minibatching.Minibatch(
                    cells_oi, genes_oi, cellxgene_oi
                )
                data = loader_train.load(minibatch)
                model.fit(data)

                pickle.dump(
                    model,
                    open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"),
                )
