import pandas as pd
import numpy as np

import chromatinhd as chd
import chromatinhd.scorer

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"

for dataset_name in [
    "pbmc10k",
    "lymphoma",
    "e18brain",
    # "pbmc10k_gran",
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
    fragments.window = window

    # create design to run
    from design import get_design, get_folds_inference

    class Prediction(chd.flow.Flow):
        pass

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_inference(fragments, folds)

    for latent_name in ["leiden_1"]:
        # create design to run
        from design import get_design, get_folds_inference

        design = get_design(dataset_name, latent_name, fragments)
        design = {
            k: design[k]
            for k in [
                "v2",
                "v2_baseline",
                "v2_64c",
                "v2_64c_baseline",
                # "v4",
                # "v4_baseline",
                # "v4_64",
                # "v4_64_baseline",
                # "v4_32",
                # "v4_32_baseline",
                # "v4_16",
                # "v4_16_baseline",
                # "v4_32-16",
                # "v4_32-16_baseline",
                # "v4_64_1l",
                # "v4_64_1l_baseline",
                # "v4_64_2l",
                # "v4_64_2l_baseline",
            ]
        }
        fold_slice = slice(0, 1)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
        folds = get_folds_inference(fragments, folds)

        for prediction_name, design_row in design.items():
            print(f"{dataset_name=} {promoter_name=} {prediction_name=}")
            prediction = chd.flow.Flow(
                chd.get_output()
                / "prediction_likelihood"
                / dataset_name
                / promoter_name
                / latent_name
                / prediction_name
            )

            # loaders
            if "loaders" in globals():
                loaders.terminate()
                del loaders
                import gc

                gc.collect()

            loaders = chd.loaders.LoaderPool(
                design_row["loader_cls"],
                design_row["loader_parameters"],
                n_workers=10,
                shuffle_on_iter=False,
            )

            # load all models
            models = [
                pickle.load(
                    open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")
                )
                for fold_ix, fold in enumerate(folds[fold_slice])
            ]

            # score
            scorer = chd.scoring.likelihood.Scorer(
                models,
                folds[: len(models)],
                loaders=loaders,
                device=device,
                gene_ids=fragments.var.index,
                cell_ids=fragments.obs.index,
            )
            scores, genescores = scorer.score()

            scores_dir = prediction.path / "scoring" / "overall"
            scores_dir.mkdir(parents=True, exist_ok=True)

            scores.to_pickle(scores_dir / "scores.pkl")
            genescores.to_pickle(scores_dir / "genescores.pkl")
