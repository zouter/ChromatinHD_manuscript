import pandas as pd
import itertools
import chromatinhd as chd

## Regulare datasets

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
                "morf_20",
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
                "morf_20",
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
            ["morf_20"],
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
                        "morf_20",
                    ]
                )
            ],
            pd.DataFrame({"method": ["v20", "counter"]}),
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
