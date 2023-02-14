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

design = chd.utils.crossing(
    design,
    pd.DataFrame(
        {
            "predictor": [
                "xgboost_100",
                "svr",
            ]
        }
    ),
)
# design = design.query("predictor == 'svr'")
design = design.query("dataset == 'lymphoma'")
print(design)

from prediction_methods import predictor_funcs


def calculate_scores(
    y_clusters, probs, train_clusters, validation_cluster, predictor_func
):
    scores = []
    for gene_ix in tqdm.tqdm(np.arange(y_clusters.shape[1])):
        y = y_clusters[train_clusters, gene_ix]
        y_validation = y_clusters[[validation_cluster], gene_ix]

        X = probs[gene_ix, train_clusters]
        X_validation = probs[gene_ix, [validation_cluster]]

        rmse = predictor_func(X, y, X_validation, y_validation)
        scores.append(
            {"gene_ix": gene_ix, "rmse": rmse, "validation_cluster": validation_cluster}
        )
    return pd.DataFrame(scores)


promoter_name = "10k10k"

design["force"] = False

for dataset_name, subdesign in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

    transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

    for latent_name, subdesign in subdesign.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

        n_latent_dimensions = latent.shape[-1]

        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

        for method_name, subdesign in subdesign.groupby("method"):
            for predictor, subdesign in subdesign.groupby("predictor"):
                predictor_func = predictor_funcs[predictor]

                scores_dir = (
                    chd.get_output()
                    / "prediction_expression_pseudobulk"
                    / dataset_name
                    / promoter_name
                    / latent_name
                    / method_name
                    / predictor
                )
                scores_dir.mkdir(exist_ok=True, parents=True)

                desired_outputs = [(scores_dir / "scores.pkl")]
                force = subdesign["force"].iloc[0]
                if not all(
                    [desired_output.exists() for desired_output in desired_outputs]
                ):
                    force = True

                if force:

                    class Prediction(chd.flow.Flow):
                        pass

                    prediction = Prediction(
                        chd.get_output()
                        / "prediction_likelihood"
                        / dataset_name
                        / promoter_name
                        / latent_name
                        / method_name
                    )
                    probs = pickle.load((prediction.path / "probs.pkl").open("rb"))

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
                        train_clusters = (
                            np.arange(len(cluster_info)) != validation_cluster
                        )

                        scores_validation_cluster = calculate_scores(
                            y_clusters,
                            np.exp(probs),
                            train_clusters,
                            validation_cluster,
                            predictor_func,
                        )
                        scores_validation_cluster[
                            "validation_cluster"
                        ] = validation_cluster
                        scores.append(scores_validation_cluster)

                    scores = pd.concat(scores).set_index(
                        ["gene_ix", "validation_cluster"]
                    )

                    pickle.dump(scores, (scores_dir / "scores.pkl").open("wb"))
