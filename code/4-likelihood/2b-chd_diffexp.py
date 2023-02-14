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


from designs import dataset_latent_combinations as design

design = chd.utils.crossing(design, pd.DataFrame({"method": ["v9_128-64-32"]}))
design["force"] = False

print(design)

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

    # fragments
    promoter_name, window = "10k10k", np.array([-10000, 10000])

    fragments = chromatinhd.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )
    fragments.window = window

    print(fragments.n_genes)

    for latent_name, subdesign in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"

        for method_name, subdesign in subdesign.groupby("method"):
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
            for fold_ix in [0]:
                scores_dir = prediction.path / "scoring" / "significant_up"
                scores_dir.mkdir(parents=True, exist_ok=True)

                # check if outputs exist
                desired_outputs = [(scores_dir / "slices.pkl")]
                force = subdesign["force"].iloc[0]
                if not all(
                    [desired_output.exists() for desired_output in desired_outputs]
                ):
                    force = True

                if force:
                    # create base pair ranking
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

                    region_result = (
                        chd.differential.DifferentialSlices.from_basepair_ranking(
                            basepair_ranking, window, cutoff=np.log(2.0)
                        )
                    )
                    pickle.dump(region_result, (scores_dir / "slices.pkl").open("wb"))
