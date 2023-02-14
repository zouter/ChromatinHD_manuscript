import pandas as pd
import itertools
import chromatinhd as chd

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
            ],
            ["leiden_0.1"],
        ),
    ),
    columns=["dataset", "latent"],
)


dataset_latent_peakcaller_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_latent_combinations.loc[
                dataset_latent_combinations["dataset"].isin(
                    ["lymphoma", "pbmc10k", "e18brain", "brain", "alzheimer"]
                )
            ],
            pd.DataFrame(
                {
                    "peakcaller": [
                        "rolling_100",
                        "rolling_500",
                        "rolling_50",
                        "cellranger",
                        "macs2_improved",
                        "macs2_leiden_0.1_merged",
                        "macs2_leiden_0.1",
                    ]
                }
            ),
        ),
        chd.utils.crossing(
            dataset_latent_combinations.loc[
                dataset_latent_combinations["dataset"].isin(
                    ["lymphoma", "pbmc10k", "e18brain", "brain"]
                )
            ],
            pd.DataFrame(
                {
                    "peakcaller": [
                        "genrich",
                    ]
                }
            ),
        ),
    ]
)


dataset_latent_method_combinations = pd.concat(
    [
        chd.utils.crossing(
            dataset_latent_combinations.loc[
                dataset_latent_combinations["dataset"].isin(
                    ["lymphoma", "pbmc10k", "e18brain", "brain", "alzheimer"]
                )
            ],
            pd.DataFrame({"method": ["v9_128-64-32"]}),
        ),
    ]
)
