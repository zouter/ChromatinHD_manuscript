import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"


class Prediction(chd.flow.Flow):
    pass


print(torch.cuda.memory_allocated(0))

from chromatinhd_manuscript.designs import (
    traindataset_testdataset_splitter_method_combinations as design,
)

#
# design = design.loc[design["testdataset"].isin(["pbmc10k"])]
# design = design.query("trainsplitter == 'random_5fold'")
# design = design.query("method == 'counter'")

design["force"] = False

for testdataset, subdesign in design.groupby("testdataset"):
    print(f"{testdataset=}")
    folder_data_preproc = folder_data / testdataset

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = chromatinhd.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )
    fragments.window = window

    transcriptome = chromatinhd.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    print(fragments.n_genes)

    # create design to run
    from design import get_design, get_folds_inference

    methods_info = get_design(transcriptome, fragments)

    # fold_slice = slice(0, 1)
    # fold_slice = slice(0, 5)
    fold_slice = slice(None, None)

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds" / "all.pkl").open("rb"))
    folds = get_folds_inference(fragments, folds)

    for (method_name, trainsplitter, traindataset), subdesign in subdesign.groupby(
        ["method", "trainsplitter", "traindataset"]
    ):
        method_info = methods_info[method_name]

        print(f"{traindataset=} {promoter_name=} {method_name=}")
        trainprediction = chd.flow.Flow(
            chd.get_output()
            / "prediction_positional"
            / traindataset
            / promoter_name
            / trainsplitter
            / method_name
        )

        testprediction = chd.flow.Flow(
            chd.get_output()
            / "prediction_positional"
            / testdataset
            / promoter_name
            / "all"
            / method_name
        )

        scores_dir = testprediction.path / "scoring" / "overall"
        scores_dir.mkdir(parents=True, exist_ok=True)

        # check if outputs are already there
        desired_outputs = [scores_dir / ("scores.pkl")]
        force = subdesign["force"].iloc[0]
        if not all([desired_output.exists() for desired_output in desired_outputs]):
            force = True

        if force:
            # loaders
            if "loaders" in globals():
                loaders.terminate()
                del loaders
                import gc

                gc.collect()

            loaders = chd.loaders.LoaderPool(
                method_info["loader_cls"],
                method_info["loader_parameters"],
                n_workers=20,
                shuffle_on_iter=False,
            )

            # load all models
            models = [
                pickle.load(
                    open(
                        trainprediction.path / ("model_0.pkl"),
                        "rb",
                    )
                )
            ]

            outcome = transcriptome.X.dense()
            scorer = chd.scoring.prediction.Scorer(
                models,
                folds,
                loaders,
                outcome,
                fragments.var.index,
                device=device,
            )

            (
                transcriptome_predicted_full,
                scores_overall,
                genescores_overall,
            ) = scorer.score(return_prediction=True)

            scores_overall.to_pickle(scores_dir / "scores.pkl")
            genescores_overall.to_pickle(scores_dir / "genescores.pkl")
            pickle.dump(
                transcriptome_predicted_full,
                (scores_dir / "transcriptome_predicted_full.pkl").open("wb"),
            )
