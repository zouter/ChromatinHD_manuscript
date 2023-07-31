# Script to score overall performance of ChromatinHD models for prediction of gene expression using test cells

import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"

from chromatinhd_manuscript.designs import (
    dataset_splitter_method_combinations as design,
)

# design = design.query("method == 'v21'")
design = design.query("method == 'v20'")
# design = design.query("method == 'v22'")
# design = design.query("method == 'counter'")
# design = design.query("dataset == 'pbmc10k'")
# design = design.query("splitter == 'random_5fold'")
design = design.query("splitter == 'permutations_5fold5repeat'")
design = design.query("promoter == '10k10k'")
# design = design.query("promoter == '100k100k'")
# design = design.query("promoter == '20kpromoter'")
# outcome_source = "counts"
outcome_source = "magic"

design["force"] = True

for (dataset_name, promoter_name), subdesign in design.groupby(["dataset", "promoter"]):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

    # fragments
    if promoter_name == "10k10k":
        window = np.array([-100000, 100000])
    elif promoter_name == "100k100k":
        window = np.array([-1000000, 1000000])
    elif promoter_name == "20kpromoter":
        window = np.array([-20000, 0])
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

        fold_slice = slice(None, None)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
        folds, cellxgene_batch_size = get_folds_inference(fragments, folds)

        for method_name, subdesign in subdesign.groupby("method"):
            method_info = methods_info[method_name]
            prediction = chd.flow.Flow(
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
                    {
                        **method_info["loader_parameters"],
                        "cellxgene_batch_size": cellxgene_batch_size,
                    },
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

                if outcome_source == "counts":
                    outcome = transcriptome.X.dense()
                else:
                    outcome = torch.from_numpy(transcriptome.adata.layers["magic"])

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

                ####

                scores_dir_overall = prediction.path / "scoring" / "overall"
                transcriptome_predicted_full = pickle.load(
                    (scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb")
                )

                # %%
                scorer_folder = prediction.path / "scoring" / "nothing"
                scorer_folder.mkdir(exist_ok=True, parents=True)

                # %%
                nothing_filterer = chd.scoring.prediction.filterers.NothingFilterer()
                Scorer2 = chd.scoring.prediction.Scorer2
                nothing_scorer = Scorer2(
                    models,
                    folds[: len(models)],
                    loaders,
                    outcome,
                    fragments.var.index,
                    fragments.obs.index,
                    device=device,
                )

                # %%
                models = [model.to("cpu") for model in models]

                # %%
                nothing_scoring = nothing_scorer.score(
                    transcriptome_predicted_full=transcriptome_predicted_full,
                    filterer=nothing_filterer,
                    extract_total=True,
                )

                # %%
                scorer_folder = prediction.path / "scoring" / "nothing"
                scorer_folder.mkdir(exist_ok=True, parents=True)
                nothing_scoring.save(scorer_folder)
