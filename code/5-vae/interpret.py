import pandas as pd
import numpy as np

import peakfreeatac as pfa
import peakfreeatac.scorer

import pickle

device = "cuda:0"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

for dataset_name in [
    "e18brain",
    "lymphoma",
    "pbmc10k",
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name
    transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = pfa.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )
    fragments.window = window

    # create design to run
    class Prediction(pfa.flow.Flow):
        pass

    # folds & minibatching
    from design import get_design, get_folds_inference
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_inference(fragments, folds)

    # create design to run
    design = get_design(fragments)
    design = {k:design[k] for k in [
        # "v4",
        # "v4_baseline",
        # "v4_decoder1",
        # "v4_decoder2",
        "v5",
        "v5_baseline",
        "v5_8",
        "v5_32",
        "v5_s0.8",
        # "v5_1decoder",
        # "v5_norescale",
        # "v5_encoder32",
        # "v5_regularizefragmentcounts",
        # "v5_regularizefragmentcounts_400epoch",
        # "v5_s0.5",
        # "v5_s0.3",
        # "v5_mixtureautoscale",
        # "v5_mixturescale0.1",
        # "v5_mixturelaplace",
        # "v6",
        # "v6",
    ]}
    fold_slice = slice(0, 1)

    for prediction_name, design_row in design.items():
        print(f"{dataset_name=} {promoter_name=} {prediction_name=}")
        prediction = Prediction(pfa.get_output() / "prediction_vae" / dataset_name / promoter_name / prediction_name)

        # loaders
        if "loaders" in globals():  
            loaders.terminate()
            del loaders
            import gc
            gc.collect()

        loaders = pfa.loaders.LoaderPool(
            design_row["loader_cls"],
            design_row["loader_parameters"],
            n_workers = 10,
            shuffle_on_iter = False
        )

        # load all models
        models = [pickle.load(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")) for fold_ix, fold in enumerate(folds[fold_slice])]

        # score
        scorer = pfa.scoring.vae.Scorer(models, transcriptome, folds[:len(models)], loaders = loaders, device = device, gene_ids = fragments.var.index, cell_ids = fragments.obs.index)
        scores, scores_cells = scorer.score()

        scores_dir = (prediction.path / "scoring" / "overall")
        scores_dir.mkdir(parents = True, exist_ok = True)

        scores.to_pickle(scores_dir / "scores.pkl")
        scores_cells.to_pickle(scores_dir / "scores_cells.pkl")

