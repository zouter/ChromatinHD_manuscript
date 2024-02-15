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

    try:
        clustering = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / latent)
    except FileNotFoundError:
        print("latent not found", dataset_name, latent)
        continue

    assert method_name == "v31"
    assert splitter == "5x1"

    regionpositional = chd.models.diff.interpret.RegionPositional(
        chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional"
    )
    regionpositional.regions = fragments.regions

    if not regionpositional.scored:
        print("not scored", dataset_name, regions_name, method_name)
        continue

    scoring_folder = regionpositional.path / "top" / "-1-1.5"
    scoring_folder.mkdir(exist_ok=True, parents=True)

    force = subdesign["force"].iloc[0]

    desired_outputs = [scoring_folder / "top_slices.pkl"]
    if not all([desired_output.exists() for desired_output in desired_outputs]):
        force = True

    if force:
        print(subdesign)
        slices = regionpositional.calculate_slices(-1.0, step=5)
        top_slices = regionpositional.calculate_top_slices(slices, fc_cutoff=1.5)

        slicescores = top_slices.get_slice_scores(regions=fragments.regions)

        slicescores["slice"] = pd.Categorical(
            slicescores["region_ix"].astype(str)
            + ":"
            + slicescores["start"].astype(str)
            + "-"
            + slicescores["end"].astype(str)
        )
        slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

        pickle.dump(
            top_slices,
            open(scoring_folder / "top_slices.pkl", "wb"),
        )
