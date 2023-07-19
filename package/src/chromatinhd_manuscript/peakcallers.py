import pandas as pd

peakcallers = pd.DataFrame(
    [
        {"peakcaller": "macs2_improved", "label": "MACS2 all cells"},
        {"peakcaller": "macs2_leiden_0.1", "label": "MACS2 per celltype"},
        {"peakcaller": "genrich", "label": "Genrich"},
        {"peakcaller": "macs2_leiden_0.1_merged", "label": "MACS2 per celltype merged"},
        {"peakcaller": "1k1k", "label": "-1kb ← TSS → +1kb"},
        {"peakcaller": "encode_screen", "label": "ENCODE screen"},
        {"peakcaller": "stack", "label": "-10kb ← TSS → +10kb"},
        {"peakcaller": "rolling_100", "label": "Window 100bp"},
        {"peakcaller": "rolling_50", "label": "Window 50bp"},
        {"peakcaller": "rolling_500", "label": "Window 500bp"},
        {"peakcaller": "cellranger", "label": "Cellranger"},
        {"peakcaller": "gene_body", "label": "TSS → +10kb"},
    ]
).set_index("peakcaller")
peakcallers.loc[
    peakcallers.index.isin(
        [
            "cellranger",
            "macs2_improved",
            "macs2_leiden_0.1_merged",
            "macs2_leiden_0.1",
            "genrich",
        ]
    ),
    "type",
] = "peak"
peakcallers.loc[
    peakcallers.index.isin(
        [
            "stack",
            "gene_body",
            "1k1k",
            "encode_screen",
            "gene_body",
        ]
    ),
    "type",
] = "predefined"
peakcallers.loc[
    pd.isnull(peakcallers["type"]),
    "type",
] = "rolling"

peakcallers["color"] = pd.Series(
    {"peak": "#FF4136", "predefined": "#2ECC40", "rolling": "#FF851B"}
)[peakcallers["type"]].values


diffexps = pd.DataFrame(
    [
        ["scanpy", "T-test (scanpy default)", "T"],
        ["scanpy_wilcoxon", "Wilcoxon rank-sum test", "W"],
        ["signac", "Logistic regression (signac default)", "L"],
    ],
    columns=["diffexp", "label", "label_short"],
).set_index("diffexp")
