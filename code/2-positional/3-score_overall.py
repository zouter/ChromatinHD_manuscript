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


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:4096'

print(torch.cuda.memory_allocated(0))

from chromatinhd_manuscript.designs import (
    dataset_splitter_method_combinations as design,
)

#
# design = design.loc[~design["dataset"].isin(["alzheimer"])]
# design = design.query("splitter == 'random_5fold'")
# design = design.loc[design["splitter"].isin(["leiden_0.1", "celltype"])]
# design = design.query("method == 'counter'")


design = design.query("method == 'v20_initdefault'")
design = design.query("dataset == 'pbmc10k'")
design = design.query("splitter == 'random_5fold'")


design["force"] = False

for dataset_name, subdesign in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

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

    for splitter, subdesign in subdesign.groupby("splitter"):
        # create design to run
        from design import get_design, get_folds_inference

        methods_info = get_design(transcriptome, fragments)

        # fold_slice = slice(0, 1)
        # fold_slice = slice(0, 5)
        fold_slice = slice(None, None)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
        folds = get_folds_inference(fragments, folds)

        for method_name, subdesign in subdesign.groupby("method"):
            method_info = methods_info[method_name]
            prediction = Prediction(
                chd.get_output()
                / "prediction_positional"
                / dataset_name
                / promoter_name
                / splitter
                / method_name
            )

            scores_dir = prediction.path / "scoring" / "overall"
            scores_dir.mkdir(parents=True, exist_ok=True)

            # check if outputs are already there
            desired_outputs = [scores_dir / ("scores.pkl")]
            force = subdesign["force"].iloc[0]
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                print(subdesign)

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
                        open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")
                    )
                    for fold_ix, fold in enumerate(folds[fold_slice])
                ]

                outcome = transcriptome.X.dense()
                scorer = chd.scoring.prediction.Scorer(
                    models,
                    folds[: len(models)],
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
