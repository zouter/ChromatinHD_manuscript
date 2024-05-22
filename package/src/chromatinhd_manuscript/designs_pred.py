import pandas as pd
import itertools
import chromatinhd as chd

## Regular datasets
dataset_baseline_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "e18brain",
                # "brain",
                "pbmc10k_gran",
                "hspc",
                "pbmc20k",
                "alzheimer",
                "liver",
                "hspc",
            ],
            ["10k10k", "100k100k"],
            ["5x1"],
            ["magic"],
            ["baseline_v42", "baseline_v21"],
        ),
    ),
    columns=["dataset", "regions", "splitter", "layer", "method"],
)
# dataset_peakcaller_combinations = dataset_peakcaller_combinations.loc[
#     ~(dataset_peakcaller_combinations["regions"].isin(["100k100k"]) & (dataset_peakcaller_combinations["peakcaller"].isin(["rolling_50", "rolling_100"])))
# ]

from .designs_peaks import dataset_peakcaller_combinations

dataset_splitter_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "hspc",
                "pbmc10k_gran",
                "e18brain",
                # "brain",
                "alzheimer",
                "pbmc20k",
                "liver",
            ],
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
            dataset_splitter_combinations,
            pd.DataFrame(
                {
                    "method": [
                        "v31",
                        "v32",
                        "v33",
                        "v35",
                    ]
                }
            ),
        ),
        chd.utils.crossing(
            dataset_splitter_combinations.query("dataset == 'pbmc10k' & regions == '100k100k'"),
            pd.DataFrame(
                {
                    "method": [
                        "v33_additive",
                    ]
                }
            ),
        ),
    ]
)


dataset_splitter_peakcaller_combinations = pd.merge(
    dataset_peakcaller_combinations,
    dataset_splitter_combinations,
    on=["dataset", "regions"],
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
dataset_splitter_peakcaller_predictor_combinations = dataset_splitter_peakcaller_predictor_combinations.loc[
    ~(
        # dataset_splitter_peakcaller_predictor_combinations["regions"].isin(["100k100k"])
        (
            dataset_splitter_peakcaller_predictor_combinations["peakcaller"].isin(
                ["rolling_50", "rolling_100", "rolling_500"]
            )
        )
        & (dataset_splitter_peakcaller_predictor_combinations["predictor"] == "linear")
    )
]

## Test datasets
traindataset_testdataset_combinations = chd.utils.crossing(
    pd.DataFrame(
        [
            ["pbmc10k", "pbmc3k-pbmc10k"],
            ["pbmc10k", "pbmc10k_gran-pbmc10k"],
            ["pbmc10k", "lymphoma-pbmc10k"],
            ["pbmc10k", "pbmc10kx-pbmc10k"],
        ],
        columns=["traindataset", "testdataset"],
    ),
    pd.DataFrame({"regions": ["10k10k", "100k100k"]}),
)

traindataset_testdataset_splitter_combinations = chd.utils.crossing(
    traindataset_testdataset_combinations, pd.DataFrame({"splitter": ["5x1"]})
)

traindataset_testdataset_splitter_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            traindataset_testdataset_splitter_combinations,
            pd.DataFrame({"method": ["v32", "v33"]}),
            pd.DataFrame(
                {
                    "layer": [
                        # "normalized",
                        "magic",
                    ],
                }
            ),
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
    pd.DataFrame(
        {
            "layer": [
                # "normalized",
                "magic",
            ],
        }
    ),
)

traindataset_testdataset_splitter_peakcaller_predictor_combinations = traindataset_testdataset_splitter_peakcaller_predictor_combinations.loc[
    ~(
        # traindataset_testdataset_splitter_peakcaller_predictor_combinations["regions"].isin(["100k100k"])
        (
            traindataset_testdataset_splitter_peakcaller_predictor_combinations["peakcaller"].isin(
                ["rolling_50", "rolling_100", "rolling_500"]
            )
        )
        & (traindataset_testdataset_splitter_peakcaller_predictor_combinations["predictor"] == "linear")
    )
]

## Baseline
dataset_splitter_baselinemethods_combinations = design = chd.utils.crossing(
    pd.DataFrame({"dataset": ["pbmc10k", "hspc"]}),
    pd.DataFrame({"splitter": ["5x1"]}),
    pd.DataFrame(
        {
            "layer": [
                "normalized",
                "magic",
            ]
        }
    ),
    pd.DataFrame({"regions": ["10k10k", "100k100k"]}),
    pd.DataFrame(
        {
            "method": ["baseline_v21", "baseline_v42"],
        }
    ),
)

## Simulation
dataset_splitter_simulation_combinations = chd.utils.crossing(
    pd.DataFrame({"dataset": ["pbmc10k"], "layer": ["normalized"]}),
    pd.DataFrame({"splitter": ["5x1"]}),
    pd.DataFrame({"regions": ["10k10k", "100k100k"]}),
    pd.DataFrame(
        {
            "method": [
                "simulated_both",
                "simulated_atac",
                "simulated_expression",
                "simulated_both_50",
                "simulated_both_10",
            ],
            "params": [
                dict(noisy_atac=True, noisy_expression=True, atac_modifier=1),
                dict(noisy_atac=True, noisy_expression=False, atac_modifier=1),
                dict(noisy_atac=False, noisy_expression=True, atac_modifier=1),
                dict(noisy_atac=True, noisy_expression=True, atac_modifier=1 / 3),
                dict(noisy_atac=True, noisy_expression=True, atac_modifier=1 / 10),
            ],
        }
    ),
)
