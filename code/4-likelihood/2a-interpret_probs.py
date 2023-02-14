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

from designs import dataset_latent_method_combinations as design

print(design)
design["force"] = False

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

        for method_name, method_row in design_latent.groupby("method"):
            design_row = method_row.iloc[0]

            force = design_row["force"]

            print(method_row)
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
                if not (prediction.path / "probs.pkl").exists():
                    force = True
                print(prediction.path)

                if force:
                    model = pickle.load(
                        (prediction.path / f"model_{fold_ix}.pkl").open("rb")
                    )

                    latent = pd.read_pickle(latent_folder / (latent_name + ".pkl"))

                    device = "cuda"
                    model = model.to(device).eval()

                    design_gene = pd.DataFrame(
                        {"gene_ix": np.arange(fragments.n_genes)}
                    )
                    design_latent = pd.DataFrame(
                        {"active_latent": np.arange(latent.shape[1])}
                    )
                    design_coord = pd.DataFrame(
                        {"coord": np.arange(window[0], window[1] + 1, step=25)}
                    )
                    design = chd.utils.crossing(
                        design_gene, design_latent, design_coord
                    )
                    batch_size = 100000
                    design["batch"] = np.floor(
                        np.arange(design.shape[0]) / batch_size
                    ).astype(int)

                    probs = []
                    for _, design_subset in tqdm.tqdm(design.groupby("batch")):
                        pseudocoordinates = torch.from_numpy(
                            design_subset["coord"].values
                        ).to(device)
                        pseudocoordinates = (pseudocoordinates - window[0]) / (
                            window[1] - window[0]
                        )
                        pseudolatent = torch.nn.functional.one_hot(
                            torch.from_numpy(design_subset["active_latent"].values).to(
                                device
                            ),
                            latent.shape[1],
                        ).to(torch.float)
                        gene_ix = torch.from_numpy(design_subset["gene_ix"].values).to(
                            device
                        )

                        prob = model.evaluate_pseudo(
                            pseudocoordinates.to(device),
                            latent=pseudolatent.to(device),
                            gene_ix=gene_ix,
                        )

                        probs.append(prob.numpy())
                    probs = np.hstack(probs)

                    probs = probs.reshape(
                        (
                            design_gene.shape[0],
                            design_latent.shape[0],
                            design_coord.shape[0],
                        )
                    )

                    pickle.dump(probs, (prediction.path / "probs.pkl").open("wb"))

                    design["gene_ix"] = design["gene_ix"].astype("category")
                    design["active_latent"] = design["active_latent"].astype("category")
                    design["batch"] = design["batch"].astype("category")
                    design["coord"] = design["coord"].astype("category")
                    pickle.dump(design, (prediction.path / "design.pkl").open("wb"))
