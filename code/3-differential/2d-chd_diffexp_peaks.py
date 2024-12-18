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


from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_method_combinations as design,
)

# design = design.query("dataset == 'GSE198467_H3K27ac'")
design = design.query("dataset in ['pbmc10k']")
# design = design.query("dataset in ['lymphoma', 'e18brain']")
# design = design.query("peakcaller in ['stack', '1k1k']")
design = design.query("diffexp in ['scanpy']")
# design = design.query("peakcaller == 'encode_screen'")

design["force"] = True
print(design)

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

    for latent_name, subdesign in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"

        for method_name, subdesign in subdesign.groupby("method"):
            prediction = chd.flow.Flow(
                chd.get_output()
                / "prediction_likelihood"
                / dataset_name
                / promoter_name
                / latent_name
                / method_name
            )

            models = []
            for fold_ix in [0]:
                try:
                    probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
                    design = pickle.load((prediction.path / "design.pkl").open("rb"))
                except FileNotFoundError as e:
                    print(e)
                    continue

                basepair_ranking = None

                for (peakcaller, diffexp), design_peakcaller in subdesign.groupby(
                    ["peakcaller", "diffexp"]
                ):

                    print(
                        f"{dataset_name=} {promoter_name=} {method_name=} {peakcaller=} {diffexp=}"
                    )
                    # create scores dir
                    scores_dir = prediction.path / "scoring" / peakcaller / diffexp
                    scores_dir.mkdir(parents=True, exist_ok=True)

                    desired_outputs = [(scores_dir / "slices.pkl")]
                    force = subdesign["force"].iloc[0]
                    if not all(
                        [desired_output.exists() for desired_output in desired_outputs]
                    ):
                        force = True

                    if force:
                        # load peak diffexp
                        peak_scores_dir = (
                            chd.get_output()
                            / "prediction_differential"
                            / dataset_name
                            / promoter_name
                            / latent_name
                            / diffexp
                            / peakcaller
                        )

                        try:
                            peakresult = pickle.load(
                                (peak_scores_dir / "slices.pkl").open("rb")
                            )
                        except FileNotFoundError as e:
                            print(e)
                            continue

                        # create base pair ranking if not done yet
                        if basepair_ranking is None:
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

                            if np.prod(probs.shape) < 10**8:
                                basepair_step = 1
                                desired_x = torch.arange(*window, step=basepair_step)
                            else:
                                basepair_step = 10
                                desired_x = torch.arange(*window, step=basepair_step)

                            probs_interpolated = chd.utils.interpolate_1d(
                                desired_x,
                                torch.from_numpy(x)[0][0],
                                torch.from_numpy(probs),
                            ).numpy()

                            prob_cutoff = np.log(1.0)

                            basepair_ranking = (
                                probs_interpolated
                                - probs_interpolated.mean(1, keepdims=True)
                            )
                            basepair_ranking[probs_interpolated < prob_cutoff] = -99999

                            # filter_background = True
                            # background_window = 250
                            # if filter_background:

                            #     def moving_average(a, n=3):
                            #         dim = -1
                            #         a_padded = torch.nn.functional.pad(
                            #             a, (n // 2, n // 2), mode="replicate"
                            #         )
                            #         ret = torch.cumsum(a_padded, axis=dim)
                            #         ret[..., :-n] = ret[..., n:] - ret[..., :-n]

                            #         return ret[..., : a.shape[dim]] / n

                            #     n = (background_window // basepair_step) + 1
                            #     x = torch.from_numpy(np.exp(probs_interpolated))
                            #     probs_smoothened = moving_average(x, n).numpy()

                            #     mask = (
                            #         np.exp(probs_interpolated) / probs_smoothened
                            #     ) < 1.2
                            #     print(mask.mean())
                            #     basepair_ranking[mask] = -99999

                        # set cutoffs for each cluster to get the same # of nucleotides in each
                        cutoffs = []
                        for cluster_ix in range(peakresult.n_clusters):
                            slices_oi = peakresult.cluster_ixs == cluster_ix

                            # calculate percentage of base pairs in a peak
                            # the same percentage will be used later to select regions
                            n = (
                                peakresult.positions[slices_oi, 1]
                                - peakresult.positions[slices_oi, 0]
                            ).sum()
                            perc = n / (fragments.n_genes * (window[1] - window[0]))
                            cutoff = np.quantile(
                                basepair_ranking[:, cluster_ix], 1 - perc
                            )
                            print(perc)
                            assert not pd.isnull(cutoff), cluster_ix
                            cutoffs.append(cutoff)
                        cutoffs = np.array(cutoffs)

                        print(cutoffs)

                        # call differential regions
                        regionresult = (
                            chd.models.diff.DifferentialSlices.from_basepair_ranking(
                                basepair_ranking,
                                window,
                                cutoff=cutoffs[:, None],
                                resolution=basepair_step,
                            )
                        )

                        print(
                            peakresult.get_slicescores()
                            .groupby("cluster_ix")["length"]
                            .sum()
                            / regionresult.get_slicescores()
                            .groupby("cluster_ix")["length"]
                            .sum()
                        )

                        pickle.dump(
                            regionresult, (scores_dir / "slices.pkl").open("wb")
                        )
