import pandas as pd
import itertools
import chromatinhd as chd

from params import binwidth_titration_ids, w_delta_p_scale_titration_ids, lr_titration_ids

dataset_splitter_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            ["pbmc10k/subsets/top250"],
            ["100k100k", "10k10k"],
            [
                "5x1",
            ],
            ["leiden_0.1"],
        ),
    ),
    columns=["dataset", "regions", "splitter", "clustering"],
)

dataset_splitter_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_splitter_combinations,
            pd.DataFrame(
                {
                    "method": [
                        "binary_1000-50bw",
                        "binary_50bw",
                        "binary_1000-50bw_nodiff",
                        "binary_1000-50bw_wdeltareg-no",
                        "binary_1000-50bw_earlystop-no",
                        "binary_1000-25bw",
                        "binary_25bw",
                        *binwidth_titration_ids,
                        *w_delta_p_scale_titration_ids,
                        *lr_titration_ids,
                        "binary_shared_lowrank_[5k,1k,500,100,50,25]bw",
                        "binary_shared_[5k,1k,500,100,50,25]bw",
                        "binary_split_[5k,1k,500,100,50,25]bw",
                        "binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep",
                        "binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop",
                        "binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop_100epochs",
                        "binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop_500epochs",
                    ]
                }
            ),
        ),
    ]
)
