import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import tqdm.auto as tqdm

import pickle

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_method_combinations as design,
)
from chromatinhd_manuscript.diff_params import params

design = design.query("splitter == '5x1'")
design = design.query("method == 'v31'")
# design = design.query("dataset == 'alzheimer'")

design = chd.utils.crossing(
    design,
    pd.DataFrame(
        {
            "motifscan": ["hocomocov12_1e-4"],
        }
    ),
)

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

for (dataset_name, regions_name, splitter, latent, method_name), subdesign in design.groupby(
    ["dataset", "regions", "splitter", "latent", "method"]
):
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")
    clustering = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    assert method_name == "v31"
    assert splitter == "5x1"

    regionpositional = chd.models.diff.interpret.RegionPositional(
        chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional"
    )
    regionpositional.regions = fragments.regions

    # slices_folder = regionpositional.path / "differential" / "0-1.5"
    slices_folder = regionpositional.path / "differential" / "-1-3"

    for motifscan_name, subdesign in subdesign.groupby("motifscan"):
        enrichment_folder = slices_folder / "enrichment" / motifscan_name
        enrichment_folder.mkdir(exist_ok=True, parents=True)

        print(enrichment_folder / "enrichment.pkl")

        force = subdesign["force"].iloc[0]
        if not (enrichment_folder / "enrichment.pkl").exists():
            force = True

        if force:
            if not (slices_folder / "differential_slices.pkl").exists():
                print("!! no differential slices found", dataset_name, regions_name, method_name)
                continue

            differential_slices = pickle.load(open(slices_folder / "differential_slices.pkl", "rb"))

            slicescores = differential_slices.get_slice_scores(regions=fragments.regions, clustering=clustering)

            slicescores["slice"] = pd.Categorical(
                slicescores["region_ix"].astype(str)
                + ":"
                + slicescores["start"].astype(str)
                + "-"
                + slicescores["end"].astype(str)
            )
            slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

            motifscan = chd.data.motifscan.MotifscanView(
                chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name
            )

            # count motifs in slices
            slicecounts = motifscan.count_slices(slices)
            enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores, slicecounts)
            enrichment["log_odds"] = np.log(enrichment["odds"])

            print(enrichment_folder / "enrichment.pkl")

            # store enrichment
            pickle.dump(enrichment, open(enrichment_folder / "enrichment.pkl", "wb"))
