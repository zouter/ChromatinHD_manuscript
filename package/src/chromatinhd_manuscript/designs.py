import pandas as pd
import itertools
import chromatinhd as chd

## Regular datasets

dataset_peakcaller_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "e18brain",
                "brain",
                "alzheimer",
                "pbmc10k_gran",
            ],
            [
                "rolling_100",
                "rolling_500",
                "rolling_50",
                "macs2_improved",
                "macs2_leiden_0.1_merged",
                "macs2_leiden_0.1",
                "encode_screen",
                "1k1k",
                "gene_body",
                "stack",
            ],
        ),
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "e18brain",
                "brain",
                # "alzheimer", # No bam file available, so no genrich
                "pbmc10k_gran",
                # "morf_20",
            ],
            ["genrich"],
        ),
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "e18brain",
                "brain",
                "alzheimer",
                "pbmc10k_gran",
                # "morf_20", # was not processed using cellranger
            ],
            ["cellranger"],
        ),
        # itertools.product(
        #     ["morf_20"],
        #     [
        #         "rolling_100",
        #         "rolling_500",
        #         "rolling_50",
        #         "macs2_improved",
        #         "encode_screen",
        #         "1k1k",
        #         "gene_body",
        #         "stack",
        #     ],
        # ),
    ),
    columns=["dataset", "peakcaller"],
)
dataset_peakcaller_combinations["promoter"] = "10k10k"

dataset_splitter_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["lymphoma"],
            ["random_5fold", "celltype"],
        ),
        itertools.product(
            # ["morf_20"],
            ["random_5fold", "overexpression"],
        ),
        itertools.product(
            [
                "pbmc10k",
                "pbmc10k_gran",
                "e18brain",
                "brain",
                "alzheimer",
            ],
            ["random_5fold", "leiden_0.1"],
        ),
    ),
    columns=["dataset", "splitter"],
)
dataset_splitter_combinations["promoter"] = "10k10k"

dataset_splitter_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_splitter_combinations.loc[
                dataset_splitter_combinations["dataset"].isin(
                    [
                        "lymphoma",
                        "pbmc10k",
                        "pbmc10k_gran",
                        "e18brain",
                        "brain",
                        "alzheimer",
                        # "morf_20",
                    ]
                )
            ],
            pd.DataFrame(
                {
                    "method": [
                        "v20",
                        "v20_initdefault",
                        "counter",
                    ]
                }
            ),
        ),
    ]
)


dataset_splitter_peakcaller_combinations = pd.merge(
    dataset_peakcaller_combinations,
    dataset_splitter_combinations,
    on=["dataset", "promoter"],
)


dataset_splitter_peakcaller_predictor_combinations = chd.utils.crossing(
    dataset_splitter_peakcaller_combinations,
    pd.DataFrame(
        {
            "predictor": [
                # "xgboost",
                "linear",
            ]
        }
    ),
)

## Test datasets
traindataset_testdataset_combinations = pd.DataFrame(
    [
        ["pbmc10k", "pbmc3k-pbmc10k"],
        ["pbmc10k", "pbmc10k_gran-pbmc10k"],
        ["pbmc10k", "lymphoma-pbmc10k"],
    ],
    columns=["traindataset", "testdataset"],
)
traindataset_testdataset_combinations["promoter"] = "10k10k"

traindataset_testdataset_splitter_combinations = chd.utils.crossing(
    traindataset_testdataset_combinations, pd.DataFrame({"splitter": ["random_5fold"]})
)

traindataset_testdataset_splitter_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            traindataset_testdataset_splitter_combinations,
            pd.DataFrame({"method": ["v20", "counter"]}),
        ),
    ]
)


traindataset_testdataset_peakcaller_combinations = pd.merge(
    dataset_peakcaller_combinations.rename(columns={"dataset": "traindataset"}),
    traindataset_testdataset_combinations,
)


traindataset_testdataset_splitter_peakcaller_combinations = pd.merge(
    traindataset_testdataset_peakcaller_combinations,
    traindataset_testdataset_splitter_combinations,
)


traindataset_testdataset_splitter_peakcaller_predictor_combinations = (
    chd.utils.crossing(
        traindataset_testdataset_splitter_peakcaller_combinations,
        pd.DataFrame({"predictor": ["linear"]}),
    )
)


## LIKELIHOOD
dataset_latent_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["lymphoma"],
            ["celltype"],
        ),
        itertools.product(
            [
                "pbmc10k",
                "e18brain",
                "brain",
                "alzheimer",
                "pbmc10k_gran",
            ],
            ["leiden_0.1"],
        ),
        itertools.product(
            # ["morf_20"],
            ["overexpression"],
        ),
    ),
    columns=["dataset", "latent"],
)


dataset_latent_peakcaller_combinations = pd.merge(
    dataset_peakcaller_combinations, dataset_latent_combinations
)


dataset_latent_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_latent_combinations.loc[
                dataset_latent_combinations["dataset"].isin(
                    [
                        "lymphoma",
                        "pbmc10k",
                        "e18brain",
                        "brain",
                        "alzheimer",
                        "pbmc10k_gran",
                        # "morf_20",
                    ]
                )
            ],
            pd.DataFrame({"method": ["v9_128-64-32"]}),
        ),
    ]
)
dataset_latent_method_combinations["promoter"] = "10k10k"


dataset_latent_peakcaller_diffexp_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_latent_peakcaller_combinations,
            pd.DataFrame({"diffexp": ["scanpy"]}),
        ),
        chd.utils.crossing(
            dataset_latent_peakcaller_combinations.query(
                "peakcaller in ['cellranger', 'macs2_improved', 'macs2_leiden_0.1_merged']"
            ),
            pd.DataFrame({"diffexp": ["signac"]}),
        ),
    ]
)

dataset_latent_peakcaller_diffexp_method_combinations = chd.utils.crossing(
    dataset_latent_peakcaller_diffexp_combinations,
    pd.DataFrame({"method": ["v9_128-64-32"]}),
)


dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations = (
    chd.utils.crossing(
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
)


## QTL

dataset_qtl_combinations = pd.DataFrame(
    [
        ["pbmc10k", "gwas_immune"],
        ["pbmc10k", "onek1k_0.2"],
        ["pbmc10k", "gtex_immune"],
        # ["brain", "gwas_cns"],
        # ["brain", "gtex_cerebellum"],
        # ["lymphoma", "gwas_lymphoma"],
    ],
    columns=["dataset", "motifscan"],
)


dataset_latent_peakcaller_diffexp_method_qtl_enricher_combinations = pd.merge(
    chd.utils.crossing(
        dataset_latent_peakcaller_combinations,
        pd.DataFrame({"method": ["v9_128-64-32"]}),
        pd.DataFrame({"diffexp": ["scanpy"]}),
        pd.DataFrame({"enricher": ["cluster_vs_all"]}),
    ),
    dataset_qtl_combinations,
)
