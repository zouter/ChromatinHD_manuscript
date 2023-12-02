import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import pickle
import copy
import tqdm.auto as tqdm
import time

from design import (
    dataset_splitter_method_combinations as design,
)
from params import params

device = "cuda:0"

design = design.query("dataset == 'pbmc10k/subsets/top250'")
design = design.query("layer == 'magic'")
# design = design.query("regions == '50k50k'")
# design = design.query("regions in ['5k5k']")
# design = design.query("regions in ['1m1m', '500k500k', '200k200k', '50k50k', '20k20k', '10k10k', '100k100k']")
# design = design.query("method == 'spline_binary_residualfull_lne2e_1layerfe'")
# design = design.query("method == 'radial_binary_1000-31frequencies_residualfull_lne2e_1layerfe_3layere2e'")
# design = design.query("method == 'radial_binary_1000-31frequencies_adamw'")
# design = design.query("method == 'radial_binary_1000-31frequencies_residualfull_bne2e_lr1e-5'")
# design = design.query("method == 'radial_binary_1000-31frequencies_residualfull_bne2e'")
# design = design.query("method == 'sine_50frequencies_residualfull_lne2e'")

design = design.copy()
design["force"] = False
dry_run = False
# design["force"] = True
# dry_run = True

print(design)

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    print(f"{dataset_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    # fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    fragments = chd.flow.Flow.from_path(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    for (splitter, layer), subdesign in subdesign.groupby(["splitter", "layer"]):
        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / splitter)

        for method_name, subdesign in subdesign.groupby("method"):
            method_info = params[method_name]

            print(subdesign.iloc[0])

            prediction = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / method_name, reset=False
            )
            performance = chd.models.pred.interpret.Performance.create(
                path=prediction.path / "scoring" / "performance",
                folds=folds,
                transcriptome=transcriptome,
                fragments=fragments,
                overwrite=subdesign["force"].iloc[0],
            )

            for region in tqdm.tqdm(transcriptome.var.index):
                if dry_run or performance.scores["scored"][region].all():
                    continue

                if "time" not in performance.scores:
                    performance.scores.create_variable(
                        "time", dtype=np.float32, sparse=False, dimensions=("region",), fill_value=np.nan
                    )

                start = time.time()

                models = chd.models.pred.model.better.Models.create(
                    fragments=fragments,
                    transcriptome=transcriptome,
                    folds=folds,
                    # reset = True,
                    model_params={**method_info["model_params"], "layer": layer},
                    train_params=method_info["train_params"],
                    regions_oi=[region],
                )
                models.train_models(pbar=True)
                end = time.time()

                performance.score(models)

                performance.scores["time"][region] = end - start
