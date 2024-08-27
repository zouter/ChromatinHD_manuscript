import itertools
import pandas as pd


dataset_peakcaller_combinations = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(["pbmc10k"], ["100k100k"], ["macs2_leiden_0.1"]),
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "e18brain",
                # "brain",
                "pbmc10k_gran",
                "pbmc3k",
                "pbmc10kx",
                "hspc",
                "liver",
                # "hepatocytes",
                "pbmc20k",
                "alzheimer",
            ],
            ["10k10k", "100k100k"],
            [
                "rolling_100",
                "rolling_500",
                "rolling_50",
                "macs2_improved",
                "macs2_leiden_0.1_merged",
                # "macs2_leiden_0.1",
                "macs2_summit",
                "macs2_summits",
                "encode_screen",
                # "1k1k",
                # "stack",
                "gene_body",
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
                "pbmc3k",
                "pbmc10kx",
                "pbmc20k",
            ],
            ["10k10k", "100k100k"],
            ["genrich"],
        ),
        itertools.product(
            [
                "pbmc10k",
                "lymphoma",
                "e18brain",
                "brain",
                "pbmc10k_gran",
                "pbmc3k",
                "pbmc10kx",
                "hspc",
                "pbmc20k",
                "alzheimer",
            ],
            ["10k10k", "100k100k"],
            ["cellranger"],
        ),
    ),
    columns=["dataset", "regions", "peakcaller"],
)
dataset_peakcaller_combinations = dataset_peakcaller_combinations.loc[
    ~(
        dataset_peakcaller_combinations["regions"].isin(["100k100k"])
        & (dataset_peakcaller_combinations["peakcaller"].isin(["rolling_50", "rolling_100"]))
    )
]
