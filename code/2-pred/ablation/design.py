import pandas as pd
import itertools
import chromatinhd as chd

from params import main_param_ids

dataset_peakcaller_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            [
                "pbmc10k/subsets/top250",
            ],
            ["10k10k", "100k100k"],
            [
                "rolling_100",
                "rolling_500",
                "rolling_50",
                "macs2_leiden_0.1_merged",
                "encode_screen",
            ],
        ),
    ),
    columns=["dataset", "regions", "peakcaller"],
)

dataset_splitter_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["pbmc10k/subsets/top250"],
            ["500k500k", "100k100k", "10k10k"],
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
                        # "radial_binary_1000-31frequencies_residualfull_lne2e_linearlib",
                        # "radial_binary_1000-31frequencies_residualfull_lnfull",
                        # "radial_binary_1000-31frequencies_residualfull_bne2e",
                        # "radial_binary_1000-31frequencies_residualfull_lne2e",
                        # "radial_binary_1000-31frequencies_adamw",
                        # "radial_binary_1000-31frequencies_splitdistance_residualfull",
                        # "radial_binary_1000-31frequencies_splitdistance_residualfull_adamw",
                        # "radial_binary_31frequencies",
                        # "radial_binary_1000frequencies",
                        # "radial_binary_1000-31frequencies",
                        # "radial_binary_1000-31frequencies_10embedding",
                        # "radial_binary_1000-31frequencies_splitdistance",
                        # *lr_design.index,
                        # *nlayers_design.index,
                        # *wd_design.index,
                        # *nfrequencies_design.index,
                        # *nfrequencies_tophat_design.index,
                        # *nhidden_design.index,
                        # "sine_50frequencies_residualfull_lne2e",
                        # "v33",
                        # "v33_nodistance",
                        # "v33_relu",
                        # "v33_gelu",
                        # "v33_tanh",
                        # "v33_sigmoid",
                        # "v33_nolib",
                        # "counter",
                        *main_param_ids
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
                "xgboost_gpu",
            ]
        }
    ),
)


region_progression = ["1k1k", "2k2k", "5k5k", "10k10k", "20k20k", "50k50k", "100k100k", "200k200k", "500k500k", "1m1m"]

dataset_splitter_combinations2 = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["pbmc10k/subsets/top250"],
            region_progression,
            [
                "5x1",
            ],
            ["magic"],
        ),
    ),
    columns=["dataset", "regions", "splitter", "layer"],
)

dataset_splitter_method_combinations2 = pd.concat(
    [
        chd.utils.crossing(
            dataset_splitter_combinations2,
            pd.DataFrame(
                {
                    "method": [
                        "spline_binary_residualfull_lne2e_1layerfe",
                    ]
                }
            ),
        ),
    ]
)

dataset_splitter_method_combinations = pd.concat(
    [
        dataset_splitter_method_combinations,
        dataset_splitter_method_combinations2,
    ]
).drop_duplicates()
