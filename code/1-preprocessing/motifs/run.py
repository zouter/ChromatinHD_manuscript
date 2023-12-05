import pandas as pd
import numpy as np
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data.peakcounts

import pathlib

from chromatinhd_manuscript.designs_pred import dataset_peakcaller_combinations as design_1
from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_peakcaller_combinations as design_2,
)

device = "cuda:1"

design = chd.utils.crossing(
    pd.DataFrame({"dataset": ["pbmc10k", "hspc"], "organism": ["hs", "hs"]}),
    pd.DataFrame({"motifs": ["hocomocov12"]}),
    pd.DataFrame({"regions": ["10k10k", "100k100k"]}),
    pd.DataFrame(
        {"cutoff": ["cutoff_0.001", "cutoff_0.0005", "cutoff_0.0001"], "cutoff_label": ["1e-3", "5e-4", "1e-4"]}
    ),
)
# design["force"] = False
design["force"] = True

for _, row in design.iterrows():
    print(row)
    dataset_name = row["dataset"]
    regions_name = row["regions"]
    organism = row["organism"]
    motifs_name = row["motifs"]

    regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / regions_name)

    motifs_folder = chd.get_output() / "data" / "motifs" / organism / motifs_name
    pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism=organism)

    motifscan_name = motifs_name + "_" + row["cutoff_label"]

    if organism == "hs":
        fasta_file = "/data/genome/GRCh38/GRCh38.fa"
    else:
        raise ValueError(f"Unknown organism {organism}")

    motifscan = chd.data.Motifscan(
        path=chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
    )
    if motifscan.scanned and not row["force"]:
        continue
    motifscan = chd.data.Motifscan.from_pwms(
        pwms,
        regions,
        motifs=motifs,
        cutoff_col=row["cutoff"],
        fasta_file=fasta_file,
        path=chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
        overwrite=True,
    )

    motifscan.create_region_indptr()
    motifscan.create_indptr()
