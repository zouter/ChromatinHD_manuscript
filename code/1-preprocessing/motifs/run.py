"""
Create all motifscan views for the different datasets
Make sure to first run scanp.ipynb to create the motifscans for the whole genome
"""

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
    pd.DataFrame.from_records(
        [
            ["pbmc10k", "hs"],
            ["pbmc10kx", "hs"],
            ["pbmc3k", "hs"],
            ["pbmc10k_gran", "hs"],
            ["e18brain", "mm"],
            ["hspc", "hs"],
            ["lymphoma", "hs"],
            ["liver", "mm"],
        ],
        columns=["dataset", "organism"],
    ),
    pd.DataFrame({"motifs": ["hocomocov12"]}),
    pd.DataFrame({"regions": ["10k10k", "100k100k"]}),
    pd.DataFrame({"cutoff": ["cutoff_0.0001"], "cutoff_label": ["1e-4"]}),
)
design = design.query("dataset == 'liver'")
# design["force"] = False
# design["force"] = True

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

    if row["organism"] == "hs":
        genome_folder = chd.get_output() / "genomes" / "GRCh38"
    elif row["organism"] == "mm":
        genome_folder = chd.get_output() / "genomes" / "mm10"
    parent = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)

    motifscan = chd.data.motifscan.MotifscanView(
        path=chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
    )
    if motifscan.scanned and not row["force"]:
        continue
    motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
        parent,
        regions,
        path=chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
    )
