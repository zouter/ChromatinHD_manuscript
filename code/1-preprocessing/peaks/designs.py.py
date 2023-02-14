import pandas as pd
import chromatinhd as chd

dataset_peakcaller_combinations = pd.concat(
    [
        chd.utils.crossing(
            pd.DataFrame(
                {
                    "dataset": [
                        "lymphoma",
                        "pbmc10k",
                        "pbmc10k_gran",
                        "e18brain",
                        "brain",
                    ]
                }
            ),
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
                        "encode_screen",
                    ]
                }
            ),
        ),
    ]
)
