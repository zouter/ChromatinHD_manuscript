import pandas as pd
import itertools
import chromatinhd as chd

from .designs_peaks import dataset_peakcaller_combinations

dataset_latent_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            [
                "lymphoma",
                "pbmc10k",
                "e18brain",
                # "brain",
                "alzheimer",
                "pbmc10k_gran",
                "liver",
                "hspc",
                # "hepatocytes",
                "pbmc20k",
            ],
            ["leiden_0.1"],
        ),
    ),
    columns=["dataset", "latent"],
)


dataset_latent_peakcaller_combinations = pd.merge(dataset_peakcaller_combinations, dataset_latent_combinations)


dataset_latent_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_latent_combinations.loc[
                dataset_latent_combinations["dataset"].isin(
                    [
                        "lymphoma",
                        "pbmc10k",
                        "e18brain",
                        # "brain",
                        "alzheimer",
                        "pbmc10k_gran",
                        "liver",
                        "hspc",
                        # "hepatocytes",
                        "pbmc20k",
                    ]
                )
            ],
            pd.DataFrame({"method": ["v30", "v31"]}),
            pd.DataFrame({"splitter": ["5x1"]}),
            pd.DataFrame({"regions": ["10k10k", "100k100k"]}),
        ),
    ]
)


dataset_latent_peakcaller_diffexp_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_latent_peakcaller_combinations,
            pd.DataFrame({"diffexp": ["t-test", "t-test-foldchange", "wilcoxon"]}),
        ),
        chd.utils.crossing(
            dataset_latent_peakcaller_combinations.query(
                "peakcaller in ['cellranger', 'macs2_improved', 'macs2_leiden_0.1_merged']"
            ),
            pd.DataFrame(
                {
                    "diffexp": [
                        "signac",
                        "snap",
                    ]
                }
            ),
        ),
        chd.utils.crossing(
            dataset_latent_peakcaller_combinations.query(
                "peakcaller in ['cellranger', 'macs2_improved', 'macs2_leiden_0.1_merged', 'rolling_100', 'macs2_summits']"
            ),
            pd.DataFrame(
                {
                    "diffexp": [
                        "logreg",
                    ]
                }
            ),
        ),
    ]
)


dataset_latent_peakcaller_diffexp_method_combinations = pd.merge(
    dataset_latent_method_combinations, dataset_latent_peakcaller_diffexp_combinations
)


dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations = chd.utils.crossing(
    dataset_latent_peakcaller_diffexp_method_combinations,
    pd.DataFrame(
        {
            "motifscan": ["cutoff_0001"],
        }
    ),
    pd.DataFrame(
        {
            "enricher": [
                "cluster_vs_clusters",
                "cluster_vs_background",
            ],
        }
    ),
)


## QTL

dataset_qtl_combinations = pd.DataFrame(
    [
        ["pbmc10k", "gwas_immune2"],
        ["pbmc10k", "onek1k_0.2"],
        ["pbmc10k", "gtex_immune"],
        # ["brain", "gwas_cns"],
        # ["brain", "gtex_cerebellum"],
        # ["lymphoma", "gwas_lymphoma"],
    ],
    columns=["dataset", "motifscan"],
)

dataset_latent_peakcaller_diffexp_method_qtl_enricher_combinations = pd.merge(
    pd.concat(
        [
            chd.utils.crossing(
                dataset_latent_peakcaller_combinations,
                pd.DataFrame({"method": ["v9_128-64-32"]}),
                pd.DataFrame({"diffexp": ["scanpy"]}),
                pd.DataFrame({"enricher": ["cluster_vs_all"]}),
            ),
            chd.utils.crossing(
                dataset_latent_peakcaller_combinations.query(
                    "peakcaller in ['cellranger', 'macs2_improved', 'macs2_leiden_0.1_merged']"
                ),
                pd.DataFrame({"method": ["v9_128-64-32"]}),
                pd.DataFrame({"diffexp": ["signac", "scanpy_wilcoxon"]}),
                pd.DataFrame({"enricher": ["cluster_vs_all"]}),
            ),
        ]
    ),
    dataset_qtl_combinations,
)
