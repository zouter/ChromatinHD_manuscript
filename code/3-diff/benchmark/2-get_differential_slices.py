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
# design = design.query("dataset == 'pbmc10k_gran'")
# design = design.query("dataset == 'lymphoma'")
# design = design.query("dataset == 'liver'")
# design = design.query("dataset == 'hepatocytes'")

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

print(design)

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

    if not regionpositional.scored:
        print("!! not scored", dataset_name, regions_name, method_name)
        continue

    # scoring_folder, prob_cutoff, fc_cutoff = regionpositional.path / "differential" / "-1-2", -1.0, 2.0
    scoring_folder, prob_cutoff, fc_cutoff = regionpositional.path / "differential" / "-1-3", -1.0, 3
    # scoring_folder, prob_cutoff, fc_cutoff = regionpositional.path / "differential" / "-2-3", -2.0, 3
    # scoring_folder, prob_cutoff, fc_cutoff = regionpositional.path / "differential" / "0-4", 0, 4
    scoring_folder.mkdir(exist_ok=True, parents=True)

    force = subdesign["force"].iloc[0]
    if not (scoring_folder / "differential_slices.pkl").exists():
        force = True

    if force:
        print(subdesign)
        slices = regionpositional.calculate_slices(prob_cutoff, step=5)
        differential_slices = regionpositional.calculate_differential_slices(slices, fc_cutoff=fc_cutoff)

        slicescores = differential_slices.get_slice_scores(regions=fragments.regions, clustering=clustering)

        slicescores["slice"] = pd.Categorical(
            slicescores["region_ix"].astype(str)
            + ":"
            + slicescores["start"].astype(str)
            + "-"
            + slicescores["end"].astype(str)
        )
        slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

        pickle.dump(
            differential_slices,
            open(scoring_folder / "differential_slices.pkl", "wb"),
        )
