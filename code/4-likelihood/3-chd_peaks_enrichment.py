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


## MOTIF
from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations as design,
)

# design = design.query("dataset == 'pbmc10k_gran'")

# design = design.query("dataset != 'alzheimer'")
# design = design.query("dataset == 'morf_20'")
# design = design.query("peakcaller in ['1k1k', 'stack']")
# design = design.query("diffexp in ['signac']")
design = design.query("dataset == 'GSE198467_H3K27ac'")


# design = design.query("dataset in ['lymphoma', 'e18brain']")
# design = design.query("dataset == 'brain'")
design = design.query("enricher == 'cluster_vs_clusters'")

## QTL
# from chromatinhd_manuscript.designs import (
#     dataset_latent_peakcaller_diffexp_method_qtl_enricher_combinations as design,
# )

# design = design.query("dataset == 'pbmc10k_gran'")
# design = design.query("peakcaller == 'cellranger'")
# design = design.query("motifscan == 'gwas_immune'")
# design = design.query("motifscan == 'onek1k_0.2'")
##

design["force"] = False
print(design)

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

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

    # onehot_promoters is going to be loaded if necessary
    onehot_promoters = None

    for latent_name, subdesign in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

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
                probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
                design = pickle.load((prediction.path / "design.pkl").open("rb"))

                basepair_ranking = None

                for (peakcaller, diffexp), subdesign in subdesign.groupby(
                    ["peakcaller", "diffexp"]
                ):
                    for ((motifscan_name, enricher), subdesign) in subdesign.groupby(
                        ["motifscan", "enricher"]
                    ):
                        # create scores dir
                        scores_dir = (
                            prediction.path
                            / "scoring"
                            / peakcaller
                            / diffexp
                            / motifscan_name
                            / enricher
                        )
                        scores_dir.mkdir(parents=True, exist_ok=True)

                        desired_outputs = [
                            (scores_dir / "scores_peaks.pkl"),
                            (scores_dir / "scores_regions.pkl"),
                        ]
                        force = subdesign["force"].iloc[0]
                        if not all(
                            [
                                desired_output.exists()
                                for desired_output in desired_outputs
                            ]
                        ):
                            force = True

                        if force:
                            print(
                                f"{dataset_name=} {promoter_name=} {method_name=} {peakcaller=} {diffexp=} {motifscan_name=} {enricher=}"
                            )
                            # load motifscan
                            motifscan_folder = (
                                chd.get_output()
                                / "motifscans"
                                / dataset_name
                                / promoter_name
                                / motifscan_name
                            )
                            motifscan = chd.data.Motifscan(motifscan_folder)

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

                            region_scores_dir = (
                                prediction.path / "scoring" / peakcaller / diffexp
                            )
                            regionresult = pickle.load(
                                (region_scores_dir / "slices.pkl").open("rb")
                            )

                            # enrichment of peak result
                            regions = peakresult.get_slicescores()
                            regions["cluster"] = pd.Categorical(
                                cluster_info.reset_index().set_index("dimension")[
                                    "cluster"
                                ][regions["cluster_ix"]]
                            )

                            if enricher == "cluster_vs_clusters":
                                enrichmentscores = chd.differential.enrichment.enrich_cluster_vs_clusters(
                                    motifscan,
                                    window,
                                    regions,
                                    "cluster",
                                    fragments.n_genes,
                                )
                            elif enricher in [
                                "cluster_vs_background",
                                "cluster_vs_background_gc",
                            ]:
                                if onehot_promoters is None:
                                    onehot_promoters = pickle.load(
                                        (
                                            folder_data_preproc
                                            / (
                                                "onehot_promoters_"
                                                + promoter_name
                                                + ".pkl"
                                            )
                                        ).open("rb")
                                    ).flatten(0, 1)
                                enrichmentscores = chd.differential.enrichment.enrich_cluster_vs_background(
                                    motifscan,
                                    window,
                                    regions,
                                    "cluster",
                                    fragments.n_genes,
                                    onehot_promoters=onehot_promoters,
                                )
                            elif enricher in ["cluster_vs_all"]:
                                enrichmentscores = (
                                    chd.differential.enrichment.enrich_cluster_vs_all(
                                        motifscan,
                                        window,
                                        regions,
                                        "cluster",
                                        fragments.n_genes,
                                        gene_ids=fragments.var.index,
                                    )
                                )

                            pickle.dump(
                                enrichmentscores,
                                (scores_dir / "scores_peaks.pkl").open("wb"),
                            )

                            # enrichment of region result
                            regions = regionresult.get_slicescores()
                            regions["cluster"] = pd.Categorical(
                                cluster_info.reset_index().set_index("dimension")[
                                    "cluster"
                                ][regions["cluster_ix"]]
                            )

                            if enricher == "cluster_vs_clusters":
                                enrichmentscores = chd.differential.enrichment.enrich_cluster_vs_clusters(
                                    motifscan,
                                    window,
                                    regions,
                                    "cluster",
                                    fragments.n_genes,
                                )
                            elif enricher in [
                                "cluster_vs_background",
                                "cluster_vs_background_gc",
                            ]:
                                enrichmentscores = chd.differential.enrichment.enrich_cluster_vs_background(
                                    motifscan,
                                    window,
                                    regions,
                                    "cluster",
                                    fragments.n_genes,
                                    onehot_promoters=onehot_promoters,
                                )
                            elif enricher in ["cluster_vs_all"]:
                                enrichmentscores = (
                                    chd.differential.enrichment.enrich_cluster_vs_all(
                                        motifscan,
                                        window,
                                        regions,
                                        "cluster",
                                        fragments.n_genes,
                                        gene_ids=fragments.var.index,
                                    )
                                )
                            pickle.dump(
                                enrichmentscores,
                                (scores_dir / "scores_regions.pkl").open("wb"),
                            )
