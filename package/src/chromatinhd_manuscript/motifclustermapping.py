import pandas as pd


def get_motifclustermapping(dataset_name, motifscan, clustering):
    def select_motif(str):
        return motifscan.motifs.index[motifscan.motifs.index.str.contains(str)][0]

    if dataset_name == "pbmc10k":
        # motifclustermapping = (
        #     enrichment.sort_values("q_value")
        #     .query("odds > 2.0")
        #     .groupby("cluster")
        #     .head(10)
        #     .sort_values("odds", ascending=False)
        #     .reset_index()
        # )
        # motifclustermapping = (
        #     enrichment.sort_values("odds", ascending=False)
        #     .query("q_value < 0.05")
        #     .groupby("cluster")
        #     .head(20)
        #     .sort_values("odds", ascending=False)
        #     .reset_index()
        # )
        motifclustermapping = pd.DataFrame(
            [
                [select_motif("TCF7"), ["CD4 memory T", "CD4 naive T"]],
                [select_motif("IRF8"), ["cDCs"]],
                [select_motif("IRF4"), ["cDCs"]],
                [
                    select_motif("CEBPB"),
                    ["CD14+ Monocytes", "FCGR3A+ Monocytes", "cDCs"],
                ],
                [
                    select_motif("ITF2"),
                    ["pDCs", "memory B", "naive B"],
                ],
                [
                    select_motif("ATF4"),
                    ["CD14+ Monocytes", "FCGR3A+ Monocytes", "cDCs"],
                ],
                [select_motif("TAL1"), ["Plasma"]],
                [select_motif("GATA4"), ["CD4 memory T", "CD4 naive T"]],
                # [select_motif("RARA"), ["MAIT"]],
                [select_motif("RUNX2"), ["NK"]],
                [select_motif("RUNX1"), ["CD8 activated T", "CD8 naive T"]],
                [select_motif("RUNX3"), ["CD8 activated T", "CD8 naive T"]],
                [select_motif("LEF1"), ["CD4 naive T", "CD8 naive T", "CD4 memory T"]],
                [
                    select_motif("TBX21"),
                    ["NK"],
                ],
                [
                    select_motif("SPI1"),
                    ["FCGR3A+ Monocytes", "CD14+ Monocytes", "cDCs"],
                ],
                [
                    select_motif("ZEB1"),
                    ["memory B", "naive B", "CD4 memory T", "CD4 naive T", "CD8 naive T", "MAIT", "pDCs"],
                ],
                [select_motif("PAX5"), ["memory B", "naive B"]],
                [select_motif("PO2F2"), ["memory B", "naive B"]],
                [select_motif("NFKB2"), ["memory B", "naive B"]],
                [select_motif("TFE2"), ["memory B", "naive B", "pDCs"]],  # TCF3
                [select_motif("BHA15"), ["pDCs"]],
                [select_motif("HTF4"), ["pDCs"]],
                [select_motif("FOS"), ["cDCs"]],
                [select_motif("IRF1"), ["cDCs"]],
            ],
            columns=["motif", "clusters"],
        ).set_index("motif")
        motifclustermapping = (
            motifclustermapping.explode("clusters")
            .rename(columns={"clusters": "cluster"})
            .reset_index()[["cluster", "motif"]]
        )
    elif dataset_name == "lymphoma":
        motifclustermapping = pd.DataFrame(
            [
                [
                    select_motif("CEBPB"),
                    ["Monocytes", "cDCs"],
                ],
                [select_motif("PO2F2"), ["Lymphoma", "Lymphoma cycling"]],
                [select_motif("PAX5"), ["Lymphoma", "Lymphoma cycling"]],
                [
                    select_motif("SPI1"),
                    ["Monocytes", "B", "Lymphoma", "Lymphoma cycling", "pDCs"],
                    # [*clustering.var.index],
                ],
                [select_motif("BHA15"), ["pDCs"]],
                [select_motif("FOS"), ["cDCs"]],
                [select_motif("IRF1"), ["cDCs"]],
            ],
            columns=["motif", "clusters"],
        ).set_index("motif")
        motifclustermapping = (
            motifclustermapping.explode("clusters")
            .rename(columns={"clusters": "cluster"})
            .reset_index()[["cluster", "motif"]]
        )
    elif dataset_name == "liver":
        motifclustermapping = pd.DataFrame(
            [
                [select_motif("ZEB1"), ["Cholangiocyte"]],
                [select_motif("TEAD1"), ["Cholangiocyte"]],
                [select_motif("ETV2"), ["LSEC"]],
                [select_motif("ERG"), ["LSEC"]],
                [select_motif("SPI1"), ["KC", "Immune"]],
                [select_motif("STAT1"), ["KC", "Immune"]],
                [select_motif("IRF1"), ["KC", "Immune"]],
                [select_motif("LEF1"), ["Central Hepatocyte"]],
                [select_motif("TF7L1"), ["Central Hepatocyte"]],
                [select_motif("CUX1"), ["Portal Hepatocyte", "Mid Hepatocyte", "Central Hepatocyte"]],
                [select_motif("HNF4"), ["Portal Hepatocyte", "Mid Hepatocyte"]],
                [select_motif("SUH"), ["Portal Hepatocyte", "Mid Hepatocyte"]],
            ],
            columns=["motif", "clusters"],
        ).set_index("motif")
        motifclustermapping = (
            motifclustermapping.explode("clusters")
            .rename(columns={"clusters": "cluster"})
            .reset_index()[["cluster", "motif"]]
        )
    elif dataset_name == "hspc":
        motifclustermapping = pd.DataFrame(
            [
                [select_motif("KLF1"), ["Erythroblast", "Erythrocyte precursors", "MEP"]],
                [select_motif("GATA1"), ["Erythroblast", "Erythrocyte precursors", "MEP"]],
                [select_motif("TAL1.H12CORE.1.P.B"), ["Erythroblast", "Erythrocyte precursors", "MEP"]],
                [select_motif("SPI1"), ["Myeloid", "Myeloblast", "MPP", "GMP"]],
                # [select_motif("IRF8"), ["Myeloid"]],
                # [select_motif("STAT1"), ["Myeloid"]],
                # [select_motif("KLF1"), ["Erythroblast", "Erythrocyte precursors"]],
                # [select_motif("SPI1"), ["B-cell precursors", "Myeloid"]],
            ],
            columns=["motif", "clusters"],
        ).set_index("motif")
        motifclustermapping = (
            motifclustermapping.explode("clusters")
            .rename(columns={"clusters": "cluster"})
            .reset_index()[["cluster", "motif"]]
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    motifclustermapping["motif_ix"] = motifscan.motifs.index.get_indexer(motifclustermapping["motif"])
    motifclustermapping["cluster_ix"] = clustering.var.index.get_indexer(motifclustermapping["cluster"])

    return motifclustermapping
