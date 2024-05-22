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
design = design.query("regions == '100k100k'")
design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'hepatocytes'")
# design = design.query("dataset in ['pbmc10k', 'pbmc10kx']")

design = design.copy()
dry_run = False
# design["force"] = False
design["force"] = True
# dry_run = True


for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    print(f"{dataset_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

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

            try:
                clustering = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / latent)
            except FileNotFoundError:
                print("latent not found", dataset_name, latent)
                continue

            method_info = params[method_name]
            models = chd.models.diff.model.binary.Models.create(
                fragments=fragments,
                clustering=clustering,
                folds=folds,
                model_params={**method_info["model_params"]},
                train_params=method_info["train_params"],
                path=prediction.path,
                overwrite=force,
            )

            print(subdesign.iloc[0])
            performance = chd.models.diff.interpret.Performance.create(
                path=prediction.path / "scoring" / "performance",
                folds=folds,
                fragments=fragments,
                overwrite=force,
            )

            if not performance.scores["scored"][:].all():
                models.train_models()
