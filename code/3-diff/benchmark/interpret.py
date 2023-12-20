import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import tqdm.auto as tqdm

import pickle

device = "cuda:0"
# device = "cuda:1"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_method_combinations as design,
)

design = design.query("splitter == '5x1'")
design = design.query("method == 'v31'")
design = design.query("dataset == 'liver'")
# design = design.query("dataset in ['pbmc10k', 'pbmc10kx']")

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

from chromatinhd_manuscript.diff_params import params

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    print(f"{dataset_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    for (splitter, latent), subdesign in subdesign.groupby(
        [
            "splitter",
            "latent",
        ]
    ):
        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / splitter)

        for method_name, subdesign in subdesign.groupby("method"):
            prediction = chd.flow.Flow(chd.get_output() / "diff" / dataset_name / regions_name / splitter / method_name)

            force = subdesign["force"].iloc[0]

            models = chd.models.diff.model.binary.Models(
                chd.get_output() / "diff" / dataset_name / regions_name / splitter / method_name
            )
            fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
            clustering = chromatinhd.data.Clustering(dataset_folder / "latent" / latent)
            models.fragments = fragments
            models.clustering = clustering

            regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

            if models.trained:
                print(models.trained)
                print(subdesign)
                if not len(regionpositional.probs) == fragments.n_regions:
                    regionpositional.score(
                        models=models,
                        device="cpu",
                        step=25,
                    )
