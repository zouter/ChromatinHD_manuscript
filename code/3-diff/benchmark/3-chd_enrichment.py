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


from designs import dataset_latent_combinations as design

design = chd.utils.crossing(
    design,
    pd.DataFrame({"method": ["v9_128-64-32"]}),
    pd.DataFrame({"motifscan": ["cutoff_0001"], "enricher": ["cluster_vs_clusters"]}),
)
design["force"] = False

# design = design.query("dataset == 'pbmc10k'")
design = design.query("dataset == 'brain'")

print(design)

for dataset_name, subdesign in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

    # fragments
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    fragments = chromatinhd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
    fragments.window = window

    print(fragments.n_genes)

    for latent_name, subdesign in subdesign.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

        for method_name, subdesign in subdesign.groupby("method"):
            print(f"{dataset_name=} {promoter_name=} {method_name=}")
            prediction = chd.flow.Flow(
                chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / method_name
            )

            for (motifscan_name, enricher), subdesign in subdesign.groupby(["motifscan", "enricher"]):
                # load motifscan
                motifscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
                motifscan = chd.data.Motifscan(motifscan_folder)

                # load regions
                regions_dir = prediction.path / "scoring" / "significant_up"
                regionresult = pickle.load((regions_dir / "slices.pkl").open("rb"))

                scores_dir = prediction.path / "scoring" / "significant_up" / "enrichment" / motifscan_name / enricher
                scores_dir.mkdir(parents=True, exist_ok=True)

                # check if outputs exist
                desired_outputs = [(scores_dir / "slices.pkl")]
                force = subdesign["force"].iloc[0]
                if not all([desired_output.exists() for desired_output in desired_outputs]):
                    force = True

                if force:
                    scores_dir = prediction.path / "scoring" / "significant_up" / motifscan_name
                    scores_dir.mkdir(exist_ok=True, parents=True)

                    regions = regionresult.get_slicescores()
                    regions["cluster"] = pd.Categorical(
                        cluster_info.reset_index().set_index("dimension")["cluster"][regions["cluster_ix"]]
                    )

                    assert enricher == "cluster_vs_clusters"

                    enrichmentscores = chd.models.diff.enrichment.enrich_cluster_vs_clusters(
                        motifscan, window, regions, "cluster", fragments.n_genes
                    )
                    pickle.dump(enrichmentscores, (scores_dir / "scores.pkl").open("wb"))
