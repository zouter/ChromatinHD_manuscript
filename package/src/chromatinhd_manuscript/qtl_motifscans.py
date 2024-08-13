import pandas as pd

motifscans = pd.DataFrame(
    [
        ["gwas_immune_main", "GWAS immune diseases"],
        ["gtex_caviar_immune", "GTEx blood eQTLs"],
    ],
    columns=["motifscan", "label"],
).set_index("motifscan")
