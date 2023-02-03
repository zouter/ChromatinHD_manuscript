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
        #     ["v4_128-64-32_30_rep"]
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
            ["v5_128-64-32"],
        ),
    ),
    columns=["dataset", "latent", "method"],
)
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
                if (prediction.path / "probs2.pkl").exists():
                    force = force or False

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
                    design["batch"] = np.floor(
                        np.arange(design.shape[0]) / 10000
                    ).astype(int)

                    mixtures = []
                    rho_deltas = []
                    rhos = []
                    probs = []
                    probs2 = []
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

                        mixture, rho_delta, rho, prob, prob2 = model.evaluate_pseudo2(
                            pseudocoordinates.to(device),
                            latent=pseudolatent.to(device),
                            gene_ix=gene_ix,
                        )

                        mixtures.append(mixture.numpy())
                        rho_deltas.append(rho_delta.numpy())
                        rhos.append(rho.numpy())
                        probs.append(prob.numpy())
                        probs2.append(prob2.numpy())
                    mixtures = np.hstack(mixtures)
                    rho_deltas = np.hstack(rho_deltas)
                    rhos = np.hstack(rhos)
                    probs = np.hstack(probs)
                    probs2 = np.hstack(probs2)

                    mixtures = mixtures.reshape(
                        (
                            design_gene.shape[0],
                            design_latent.shape[0],
                            design_coord.shape[0],
                        )
                    )
                    rho_deltas = rho_deltas.reshape(
                        (
                            design_gene.shape[0],
                            design_latent.shape[0],
                            design_coord.shape[0],
                        )
                    )
                    rhos = rhos.reshape(
                        (
                            design_gene.shape[0],
                            design_latent.shape[0],
                            design_coord.shape[0],
                        )
                    )
                    probs = probs.reshape(
                        (
                            design_gene.shape[0],
                            design_latent.shape[0],
                            design_coord.shape[0],
                        )
                    )
                    probs2 = probs2.reshape(
                        (
                            design_gene.shape[0],
                            design_latent.shape[0],
                            design_coord.shape[0],
                        )
                    )

                    mixture_diff = mixtures - mixtures.mean(-2, keepdims=True)
                    probs_diff = (
                        mixture_diff + rho_deltas
                    )  # - rho_deltas.mean(-2, keepdims = True)

                    prob_cutoff = np.log(1.0)
                    mask = probs2 > prob_cutoff

                    probs_diff_masked = probs_diff.copy()
                    probs_diff_masked[~mask] = np.nan
                    mixture_diff_masked = mixture_diff.copy()
                    mixture_diff_masked[~mask] = np.nan

                    X = mixture_diff_masked.copy()
                    X[np.isnan(X)] = 0.0

                    pickle.dump(probs2, (prediction.path / "probs2.pkl").open("wb"))
                    pickle.dump(probs, (prediction.path / "probs.pkl").open("wb"))
                    pickle.dump(mixtures, (prediction.path / "mixtures.pkl").open("wb"))
                    pickle.dump(rhos, (prediction.path / "rhos.pkl").open("wb"))
                    pickle.dump(
                        rho_deltas, (prediction.path / "rho_deltas.pkl").open("wb")
                    )

                    design["gene_ix"] = design["gene_ix"].astype("category")
                    design["active_latent"] = design["active_latent"].astype("category")
                    design["batch"] = design["batch"].astype("category")
                    design["coord"] = design["coord"].astype("category")
                    pickle.dump(design, (prediction.path / "design.pkl").open("wb"))

                ######
                # For ranking
                ######
                model = pickle.load(
                    (prediction.path / f"model_{fold_ix}.pkl").open("rb")
                )

                latent = pd.read_pickle(latent_folder / (latent_name + ".pkl"))

                device = "cuda"
                model = model.to(device).eval()

                n_latent = latent.shape[1]

                # create design for inference
                design_gene = pd.DataFrame({"gene_ix": np.arange(fragments.n_genes)})
                design_latent = pd.DataFrame({"active_latent": np.arange(n_latent)})
                design_coord = pd.DataFrame(
                    {"coord": np.arange(window[0], window[1] + 1, step=25)}
                )
                design = chd.utils.crossing(design_gene, design_latent, design_coord)
                design["batch"] = np.floor(np.arange(design.shape[0]) / 10000).astype(
                    int
                )

                # infer
                probs = []
                rho_deltas = []
                rhos = []
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
                        n_latent,
                    ).to(torch.float)
                    gene_ix = torch.from_numpy(design_subset["gene_ix"].values).to(
                        device
                    )

                    likelihood_mixture, rho_delta, rho = model.evaluate_pseudo(
                        pseudocoordinates.to(device),
                        latent=pseudolatent.to(device),
                        gene_ix=gene_ix,
                    )
                    prob_mixture = likelihood_mixture.detach().cpu().numpy()
                    rho_delta = rho_delta.detach().cpu().numpy()
                    rho = rho.detach().cpu().numpy()

                    probs.append(prob_mixture)
                    rho_deltas.append(rho_delta)
                    rhos.append(rho)
                probs = np.hstack(probs)
                rho_deltas = np.hstack(rho_deltas)
                rhos = np.hstack(rhos)

                probs = probs.reshape(
                    (
                        design_gene.shape[0],
                        design_latent.shape[0],
                        design_coord.shape[0],
                    )
                )
                rho_deltas = rho_deltas.reshape(
                    (
                        design_gene.shape[0],
                        design_latent.shape[0],
                        design_coord.shape[0],
                    )
                )
                rhos = rhos.reshape(
                    (
                        design_gene.shape[0],
                        design_latent.shape[0],
                        design_coord.shape[0],
                    )
                )

                pickle.dump(probs, (prediction.path / "ranking_probs.pkl").open("wb"))
                pickle.dump(
                    rho_deltas, (prediction.path / "ranking_rho_deltas.pkl").open("wb")
                )
                pickle.dump(rhos, (prediction.path / "ranking_rhos.pkl").open("wb"))

                design["gene_ix"] = design["gene_ix"].astype("category")
                design["active_latent"] = design["active_latent"].astype("category")
                design["batch"] = design["batch"].astype("category")
                design["coord"] = design["coord"].astype("category")
                pickle.dump(design, (prediction.path / "ranking_design.pkl").open("wb"))
