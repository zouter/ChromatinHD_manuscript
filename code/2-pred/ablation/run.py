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
from params import (
    params,
    nfrequencies_design,
    nfrequencies_tophat_design,
    nhidden_design,
    layernorm_design,
    ncells_design,
    residual_design,
    dropout_design,
    encoder_design,
)

print(encoder_design)

device = "cuda:0"

design = design.query("dataset == 'pbmc10k/subsets/top250'")
design = design.query("layer == 'magic'")
design = design.query("regions == '100k100k'")
# design = design.query("regions in ['5k5k']")
# design = design.query("regions in ['1m1m', '500k500k', '200k200k', '50k50k', '20k20k', '10k10k', '100k100k']")
# design = design.query("method == 'v33_windows500'")
design = design.loc[design["method"].isin(["v33_silu"])]
# design = design.loc[design["method"].isin(["v33_windows100", "v33_windows500"])]
# design = design.loc[design["method"].isin(dropout_design.index)]
# design = design.loc[design["method"].isin(nhidden_design.index)]
# design = design.loc[design["method"].isin(layernorm_design.index)]
# design = design.loc[design["method"].isin(encoder_design.index)]
# design = design.iloc[[-4]]

design = design.copy()
design["force"] = False
dry_run = False
design["force"] = True
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

        for method_name, subdesign in subdesign.groupby("method", sort=False):
            method_info = params[method_name]
            interpret = method_info["interpret"] if "interpret" in method_info else False

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

            if interpret:
                if "interpretation_params" not in method_info:
                    interpretation_params = {}
                else:
                    interpretation_params = method_info["interpretation_params"]
                interpretation_device = (
                    interpretation_params["device"] if "device" in interpretation_params else "cuda:0"
                )
                window_sizes = (
                    interpretation_params["window_sizes"] if "window_sizes" in interpretation_params else (50,)
                )

                censorer = chd.models.pred.interpret.censorers.MultiWindowCensorer(
                    fragments.regions.window, window_sizes=window_sizes
                )
                regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
                    path=prediction.path / "scoring" / "regionmultiwindow",
                    folds=folds,
                    transcriptome=transcriptome,
                    fragments=fragments,
                    censorer=censorer,
                    overwrite=True,
                )
                if "time_interpretation" not in performance.scores:
                    performance.scores.create_variable(
                        "time_interpretation",
                        dtype=np.float32,
                        sparse=False,
                        dimensions=("region",),
                        fill_value=np.nan,
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

                if interpret:
                    start = time.time()

                    regionmultiwindow.score(models, min_fragments=3, device=interpretation_device)
                    end = time.time()
                    performance.scores["time_interpretation"][region] = end - start
