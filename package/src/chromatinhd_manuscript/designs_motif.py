import pandas as pd
import chromatinhd as chd


datasets = chd.utils.crossing(
    pd.DataFrame(
        {
            "dataset": [
                "pbmc10k",
                "hspc",
                "lymphoma",
                "pbmc10kx",
                "pbmc10k_gran",
                "e18brain",
                "liver",
                # "hepatocytes",
                "pbmc20k",
                "alzheimer",
            ],
            "organism": ["hs", "hs", "hs", "hs", "hs", "mm", "mm", "hs", "mm"],
        }
    ),
    pd.DataFrame({"regions": ["100k100k", "10k10k"]}),
)
motifs = pd.DataFrame(
    [
        ["hocomocov12_1e-4"],
    ],
    columns=["motifscan"],
)


design = pd.concat(
    [
        chd.utils.crossing(datasets, motifs),
    ],
    ignore_index=True,
)
