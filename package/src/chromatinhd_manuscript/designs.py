import pandas as pd
import itertools
import chromatinhd as chd

## Regular datasets


dataset_splitter_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["pbmc10k", "lymphoma"],
            ["100k100k", "10k10k"],
            [
                "5x1",
            ],
            ["magic"],
        ),
    ),
    columns=["dataset", "regions", "splitter", "layer"],
)

dataset_splitter_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_splitter_combinations.loc[
                dataset_splitter_combinations["dataset"].isin(
                    [
                        "lymphoma",
                        "pbmc10k",
                        "pbmc3k",
                        "pbmc10k_gran",
                        "e18brain",
                        "brain",
                    ]
                )
            ],
            pd.DataFrame(
                {
                    "method": [
                        "v30",
                        "v31",
                    ]
                }
            ),
        ),
    ]
)


dataset_splitter_peakcaller_combinations = pd.merge(
    dataset_peakcaller_combinations,
    dataset_splitter_combinations,
    on=["dataset", "regions", "layer"],
)


dataset_splitter_peakcaller_predictor_combinations = chd.utils.crossing(
    dataset_splitter_peakcaller_combinations,
    pd.DataFrame(
        {
            "predictor": [
                "linear",
                "lasso",
                "xgboost",
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
traindataset_testdataset_combinations["regions"] = "10k10k"

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


traindataset_testdataset_splitter_peakcaller_predictor_combinations = chd.utils.crossing(
    traindataset_testdataset_splitter_peakcaller_combinations,
    pd.DataFrame({"predictor": ["linear", "lasso", "xgboost"]}),
)
