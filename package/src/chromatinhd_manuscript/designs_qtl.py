import pandas as pd
import chromatinhd as chd

immune_qtls = pd.DataFrame(
    [
        ["hs/gwas", "gwas_immune", "gwas_immune"],
        ["hs/gwas", "gwas_immune", "gwas_immune_main"],
        ["hs/causaldb", "causaldb_immune", "causaldb_immune"],
        ["hs/gtex", "gtex_immune", "gtex_immune"],
        ["hs/gtex_caviar", "gtex_immune", "gtex_caviar_immune"],
        ["hs/gtex_caviar", "gtex_immune", "gtex_caviar_immune_differential"],
        ["hs/gtex_caveman", "gtex_immune", "gtex_caveman_immune"],
        ["hs/gtex_caveman", "gtex_immune", "gtex_caveman_immune_differential"],
        # ["hs/onek1k", "gwas_specific", "onek1k_gwas_specific"],
    ],
    columns=["folder_qtl", "qtl_name", "motifscan"],
)
immune_datasets = chd.utils.crossing(
    pd.DataFrame(
        {
            "dataset": ["pbmc10k", "hspc", "lymphoma", "pbmc10kx", "pbmc10k_gran", "pbmc20k"],
            "genome": ["GRch38", "GRCh38", "GRCh38", "GRCh38", "GRCh38", "GRCh38"],
        }
    ),
    pd.DataFrame({"regions": ["100k100k", "10k10k"]}),
)


liver_qtls = pd.DataFrame(
    [
        ["hs/gwas", "gwas_liver", "gwas_liver"],
        ["hs/gwas", "gwas_liver", "gwas_liver_main"],
        ["hs/causaldb", "causaldb_liver", "causaldb_liver"],
        ["hs/gtex", "gtex_liver", "gtex_liver"],
        ["hs/gtex_caviar", "gtex_liver", "gtex_caviar_liver"],
        ["hs/gtex_caviar", "gtex_liver", "gtex_caviar_liver_differential"],
        ["hs/gtex_caveman", "gtex_liver", "gtex_caveman_liver"],
        ["hs/gtex_caveman", "gtex_liver", "gtex_caveman_liver_differential"],
    ],
    columns=["folder_qtl", "qtl_name", "motifscan"],
)
liver_datasets = chd.utils.crossing(
    pd.DataFrame({"dataset": ["liver", "hepatocytes"], "genome": ["mm10", "mm10"]}),
    pd.DataFrame({"regions": ["100k100k", "10k10k"]}),
)

lymphoma_qtls = pd.DataFrame(
    [
        ["hs/gwas", "gwas_lymphoma", "gwas_lymphoma"],
        ["hs/gwas", "gwas_lymphoma", "gwas_lymphoma_main"],
        # ["hs/causaldb", "causaldb_lymphoma", "causaldb_lymphoma"],
    ],
    columns=["folder_qtl", "qtl_name", "motifscan"],
)
lymphoma_datasets = chd.utils.crossing(
    pd.DataFrame({"dataset": ["lymphoma"], "genome": ["GRCh38"]}),
    pd.DataFrame({"regions": ["100k100k", "10k10k"]}),
)


hema_qtls = pd.DataFrame(
    [
        ["hs/gwas", "gwas_hema", "gwas_hema"],
        ["hs/gwas", "gwas_hema", "gwas_hema_main"],
        # ["hs/causaldb", "causaldb_hema", "causaldb_hema"],
    ],
    columns=["folder_qtl", "qtl_name", "motifscan"],
)
hema_datasets = chd.utils.crossing(
    pd.DataFrame({"dataset": ["hspc"], "genome": ["GRCh38"]}),
    pd.DataFrame({"regions": ["100k100k", "10k10k"]}),
)


cns_qtls = pd.DataFrame(
    [
        ["hs/gwas", "gwas_cns", "gwas_cns"],
        ["hs/gwas", "gwas_cns", "gwas_cns_main"],
        ["hs/gtex", "gtex_cerebellum", "gtex_cerebellum"],
        ["hs/gtex_caviar", "gtex_cerebellum", "gtex_caviar_cerebellum"],
        ["hs/gtex_caviar", "gtex_cerebellum", "gtex_caviar_cerebellum_differential"],
        ["hs/gtex_caveman", "gtex_cerebellum", "gtex_caveman_cerebellum"],
        ["hs/gtex_caveman", "gtex_cerebellum", "gtex_caveman_cerebellum_differential"],
        # ["hs/causaldb", "causaldb_cns", "causaldb_cns"],
    ],
    columns=["folder_qtl", "qtl_name", "motifscan"],
)
cns_datasets = chd.utils.crossing(
    pd.DataFrame({"dataset": ["e18brain", "alzheimer"], "genome": ["mm10", "mm10"]}),
    pd.DataFrame({"regions": ["100k100k", "10k10k"]}),
)

design = pd.concat(
    [
        chd.utils.crossing(immune_qtls, immune_datasets),
        chd.utils.crossing(liver_qtls, liver_datasets),
        chd.utils.crossing(lymphoma_qtls, lymphoma_datasets),
        chd.utils.crossing(hema_qtls, hema_datasets),
        chd.utils.crossing(cns_qtls, cns_datasets),
    ],
    ignore_index=True,
)
