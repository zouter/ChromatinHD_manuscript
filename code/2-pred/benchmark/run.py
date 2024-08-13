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

from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_method_combinations as design,
)

design = design.query("splitter == '5x1'")
# design = design.query("splitter == '5x1'")
# design = design.query("method == 'counter'")
# design = design.query("method == 'v30'")
# design = design.query("method == 'v31'")
# design = design.query("method == 'v33'")
design = design.query("method == 'v33_additive'")
# design = design.query("method == 'v22'")
# design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'hspc'")
# design = design.query("dataset == 'pbmc10k_gran'")
# design = design.query("dataset == 'e18brain'")
# design = design.query("dataset == 'lymphoma'")
# design = design.query("dataset == 'liver'")
# design = design.query("regions == '10k10k'")
# design = design.query("regions == '20kpromoter'")
# design = design.query("regions == '100k100k'")
# design = design.query("regions == '100k100k'")

design = design.loc[(design["layer"] == "magic")]
# design = design.loc[(design["layer"] == "normalized")]

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

from chromatinhd_manuscript.pred_params import params

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    print(f"{dataset_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    for (splitter, layer), subdesign in subdesign.groupby(["splitter", "layer"]):
        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / splitter)

        for method_name, subdesign in subdesign.groupby("method"):
            prediction = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / method_name
            )

            force = subdesign["force"].iloc[0]

            performance = chd.models.pred.interpret.Performance.create(
                path=prediction.path / "scoring" / "performance",
                folds=folds,
                transcriptome=transcriptome,
                fragments=fragments,
                overwrite=force,
            )
            print(performance.scores.coords_pointed["region"])

            # censorer = chd.models.pred.interpret.censorers.MultiWindowCensorer(fragments.regions.window)
            # regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
            #     path=prediction.path / "scoring" / "regionmultiwindow",
            #     folds=folds,
            #     transcriptome=transcriptome,
            #     fragments=fragments,
            #     censorer=censorer,
            #     overwrite=force,
            # )

            method_info = params[method_name]

            print(subdesign.iloc[0])

            for region in tqdm.tqdm(transcriptome.var.sort_values("dispersions_norm", ascending=False).index):
                if performance.scores["scored"][region].all():
                    continue
                    # if (performance.scores["cor"][region, :, "test"].mean() < 0.2) or (
                    #     regionmultiwindow.scores["scored"][region].all()
                    # ):
                    #     continue

                print(transcriptome.symbol(region))

                models = chd.models.pred.model.better.Models.create(
                    fragments=fragments,
                    transcriptome=transcriptome,
                    folds=folds,
                    model_params={**method_info["model_params"], "layer": layer},
                    train_params=method_info["train_params"],
                    regions_oi=[region],
                )

                if dry_run:
                    continue
                models.train_models(device=device)

                performance.score(models)

                if performance.scores["cor"][region, :, "test"].mean() < 0.2:
                    continue
                # print("scoring region multi window")
                # regionmultiwindow.score(models, device="cpu")
