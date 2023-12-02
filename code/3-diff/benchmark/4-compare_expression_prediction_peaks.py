import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data
import chromatinhd.loaders.fragmentmotif
import chromatinhd.loaders.minibatches

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"


class Prediction(chd.flow.Flow):
    pass


import itertools

from designs import dataset_latent_peakcaller_combinations as design

design = chd.utils.crossing(
    design,
    pd.DataFrame(
        {
            "predictor": [
                "svr",
                "xgboost_100",
            ]
        }
    ),
)
design = design.query("dataset == 'lymphoma'")
design["force"] = False
print(design)

from prediction_methods import predictor_funcs


def calculate_scores(
    peakcounts,
    y_clusters,
    peakcounts_clusters,
    train_clusters,
    validation_cluster,
    predictor_func,
):
    peak_to_peak_ix = peakcounts.var["ix"].to_dict()
    scores = []
    peaks = peakcounts.peaks
    peaks = peaks.loc[peaks["gene_ix"].isin(np.arange(y_clusters.shape[1]))]
    for gene_ix, peaks_gene in tqdm.tqdm(peaks.groupby("gene_ix")):
        peak_ixs = [peak_to_peak_ix[peak] for peak in peaks_gene["peak"]]
        y = y_clusters[train_clusters, gene_ix]
        y_validation = y_clusters[[validation_cluster], gene_ix]

        X = peakcounts_clusters[:, peak_ixs][train_clusters]
        X_validation = peakcounts_clusters[:, peak_ixs][[validation_cluster]]

        rmse = predictor_func(X, y, X_validation, y_validation)
        scores.append({"gene_ix": gene_ix, "rmse": rmse, "validation_cluster": validation_cluster})
    return pd.DataFrame(scores)


promoter_name = "10k10k"

for dataset_name, subdesign in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

    transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

    for latent_name, subdesign in subdesign.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

        n_latent_dimensions = latent.shape[-1]

        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

        for peakcaller, subdesign in subdesign.groupby("peakcaller"):
            for predictor, subdesign in subdesign.groupby("predictor"):
                predictor_func = predictor_funcs[predictor]

                scores_dir = (
                    chd.get_output()
                    / "prediction_expression_pseudobulk"
                    / dataset_name
                    / promoter_name
                    / latent_name
                    / peakcaller
                    / predictor
                )
                scores_dir.mkdir(exist_ok=True, parents=True)

                # check if outputs are already there
                desired_outputs = [(scores_dir / "scores.pkl")]
                force = subdesign["force"].iloc[0]
                if not all([desired_output.exists() for desired_output in desired_outputs]):
                    force = True

                if force:
                    # get peak counts
                    peakcounts = chd.peakcounts.FullPeak(
                        folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
                    )
                    peakcounts_clusters = np.vstack(
                        pd.from_dummies(latent)
                        .rename(columns=lambda x: "latent")
                        .groupby("latent")
                        .apply(lambda x: np.array(peakcounts.counts[x.index].mean(0))[0])
                    )

                    # get y cluster
                    y_cells = np.array(transcriptome.X.to_scipy_csr().todense())
                    y_clusters = (
                        pd.DataFrame(y_cells, index=pd.from_dummies(latent))
                        .groupby(level=0)
                        .mean()
                        .values
                        # .values[:, :500]
                    )
                    print(y_clusters)

                    # calculate scores

                    scores = []

                    for validation_cluster in np.arange(len(cluster_info)):
                        train_clusters = np.arange(len(cluster_info)) != validation_cluster

                        scores_validation_cluster = calculate_scores(
                            peakcounts,
                            y_clusters,
                            peakcounts_clusters,
                            train_clusters,
                            validation_cluster,
                            predictor_func,
                        )
                        scores_validation_cluster["validation_cluster"] = validation_cluster
                        scores.append(scores_validation_cluster)

                    scores = pd.concat(scores).set_index(["gene_ix", "validation_cluster"])

                    pickle.dump(scores, (scores_dir / "scores.pkl").open("wb"))
