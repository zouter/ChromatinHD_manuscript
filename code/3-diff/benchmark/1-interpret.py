from chromatinhd_manuscript.designs_diff import (
    dataset_latent_method_combinations as design,
)
from chromatinhd_manuscript.diff_params import params

import pickle

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

device = "cuda:0"
# device = "cuda:1"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"


design = design.query("splitter == '5x1'")
design = design.query("method == 'v31'")
design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'alzheimer'")
# design = design.query("dataset in ['pbmc10k', 'pbmc10kx']")

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True


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
            regionpositional.regions = fragments.regions

            if models.trained:
                print(models.trained)
                print(subdesign)
                if not len(regionpositional.probs) == fragments.n_regions:
                    regionpositional.score(
                        models=models,
                        device="cpu",
                        step=25,
                        # regions = ["ENSG00000125347"]
                    )
