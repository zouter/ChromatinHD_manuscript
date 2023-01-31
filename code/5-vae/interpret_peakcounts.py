import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.data
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.loaders.minibatching

import pickle

import matplotlib.pyplot as plt

device = "cuda:0"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

import peakfreeatac.peakcounts

import pickle
import copy
import numpy as np

def empty_cache():
    if "loaders" in globals():
        global loaders  
        loaders.terminate()
        del loaders
        import gc
        gc.collect()

class Prediction(pfa.flow.Flow):
    pass

for dataset_name in [
    "e18brain",
    "lymphoma",
    "pbmc10k",
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    transcriptome = peakfreeatac.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )

    fragments = peakfreeatac.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )
    fragments.window = window
    fragments.create_cut_data()

    for peaks_name in [
        "cellranger",
        "macs2",
        "stack"
    ]:
        peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)

        # create design to run
        from design import get_design_peakcount, get_folds_inference
        design = get_design_peakcount(fragments, peakcounts)
        design = {k:design[k] for k in [
            "pca_20",
            "pca_50",
            "pca_200",
            "pca_5",
        ]}
        fold_slice = slice(0, 1)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
        folds = get_folds_inference(fragments, folds)

        for prediction_name, design_row in design.items():
            print(f"{dataset_name=} {promoter_name=} {peaks_name=} {prediction_name=}")
            prediction = Prediction(pfa.get_output() / "prediction_vae" / dataset_name / promoter_name / peaks_name / prediction_name)

            # loaders
            empty_cache()

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