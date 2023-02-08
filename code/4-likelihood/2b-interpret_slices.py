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


class Prediction(chd.flow.Flow):
    pass


import itertools

design = pd.DataFrame.from_records(
    itertools.chain(
        # itertools.product(
        #     ["lymphoma"],
        #     ["celltype"],
        #     ["v9_128-64-32"],
        # ),
        itertools.product(
            [
                "pbmc10k",
                # "e18brain",
                # "brain",
                # "alzheimer",
            ],
            ["leiden_0.1"],
            # ["v4_128-64-32_30_rep"],
            # ["v5_128-64-32"],
            # ["v8_128-64-32"],
            ["v9_128-64-32"],
            [
                "significant_up",
                "cellranger",
                "macs2_improved",
                "macs2_leiden_0.1_merged",
                "genrich",
            ],
        ),
    ),
    columns=["dataset", "latent", "method", "peakcaller"],
)
print(design)
design["force"] = True

for dataset_name, design_dataset in design.groupby("dataset"):
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

    print(fragments.n_genes)

    for latent_name, design_latent in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"

        # create design to run
        from design import get_design, get_folds_training

        methods_info = get_design(dataset_name, latent_name, fragments)

        fold_slice = slice(0, 1)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))

        for method_name, design_method in design_latent.groupby("method"):
            method_info = methods_info[method_name]

            print(f"{dataset_name=} {promoter_name=} {method_name=}")
            prediction = Prediction(
                chd.get_output()
                / "prediction_likelihood"
                / dataset_name
                / promoter_name
                / latent_name
                / method_name
            )

            models = []
            for fold_ix, fold in [
                (fold_ix, fold) for fold_ix, fold in enumerate(folds)
            ][fold_slice]:
                force = True
                # if not (prediction.path / "probs.pkl").exists():
                #     force = True

                if force:
                    probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
                    design = pickle.load((prediction.path / "design.pkl").open("rb"))

                    probs_diff = probs - probs.mean(1, keepdims=True)
                    design["gene_ix"] = design["gene_ix"]
                    x = (
                        (design["coord"].values)
                        .astype(int)
                        .reshape(
                            (
                                len(design["gene_ix"].cat.categories),
                                len(design["active_latent"].cat.categories),
                                len(design["coord"].cat.categories),
                            )
                        )
                    )
                    desired_x = torch.arange(*window)
                    probs_interpolated = chd.utils.interpolate_1d(
                        desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
                    ).numpy()

                    prob_cutoff = np.log(1.0)

                    basepair_ranking = probs_interpolated - probs_interpolated.mean(
                        1, keepdims=True
                    )
                    basepair_ranking[probs_interpolated < prob_cutoff] = -np.inf

                    for peakcaller, design_peakcaller in design_method.groupby(
                        "peakcaller"
                    ):
                        scores_dir = prediction.path / "scoring" / peakcaller
                        scores_dir.mkdir(parents=True, exist_ok=True)

                        if peakcaller == "significant_up":
                            pureregionresult = chd.differential.DifferentialSlices.from_basepair_ranking(
                                basepair_ranking, window, cutoff=np.log(2.0)
                            )

                            pickle.dump()
