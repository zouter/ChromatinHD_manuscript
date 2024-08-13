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
    traindataset_testdataset_splitter_method_combinations as design,
)

design = design.query("splitter == '5x1'")
# design = design.query("method == 'v32'")
design = design.query("method == 'v33'")
# design = design.query("testdataset == 'lymphoma-pbmc10k'")
# design = design.query("testdataset == 'pbmc3k-pbmc10k'")
# design = design.query("testdataset == 'pbmc10k_gran-pbmc10k'")
# design = design.query("testdataset == 'pbmc10kx-pbmc10k'")
design = design.loc[(design["layer"] == "magic")]

print(design)

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

from chromatinhd_manuscript.pred_params import params

for (dataset_name, traindataset_name, regions_name), subdesign in design.groupby(
    ["testdataset", "traindataset", "regions"]
):
    print(f"{traindataset_name=} {dataset_name=} {regions_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name
    traindataset_folder = chd.get_output() / "datasets" / traindataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    train_fragments = chromatinhd.data.Fragments(traindataset_folder / "fragments" / regions_name)
    train_transcriptome = chromatinhd.data.Transcriptome(traindataset_folder / "transcriptome")

    for (splitter, layer), subdesign in subdesign.groupby(["splitter", "layer"]):
        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / splitter)

        for method_name, subdesign in subdesign.groupby("method"):
            prediction = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / method_name
            )

            force = subdesign["force"].iloc[0]

            print(folds[0]["cells_train"])

            folds = [
                {
                    "cells_train": np.concatenate([fold["cells_train"], fold["cells_test"]]),
                    "cells_validation": fold["cells_validation"],
                }
                for fold in folds
            ]

            folds_test = [
                {
                    "cells_test": np.arange(len(fragments.obs)),
                    "cells_train": np.arange(len(fragments.obs)),
                    "cells_validation": np.arange(len(fragments.obs)),
                }
                for i in range(len(folds))
            ]

            performance = chd.models.pred.interpret.Performance.create(
                path=prediction.path / "scoring" / "performance",
                folds=folds_test,
                transcriptome=transcriptome,
                fragments=fragments,
                overwrite=force,
            )
            print(performance.scores.coords_pointed["region"])

            method_info = params[method_name]

            print(subdesign.iloc[0])

            for region in tqdm.tqdm(transcriptome.var.sort_values("dispersions_norm", ascending=False).index):
                if performance.scores["scored"][region].all():
                    continue

                print(transcriptome.symbol(region))

                models = chd.models.pred.model.better.Models.create(
                    fragments=train_fragments,
                    transcriptome=train_transcriptome,
                    folds=folds,
                    model_params={**method_info["model_params"], "layer": layer},
                    train_params=method_info["train_params"],
                    regions_oi=[region],
                )

                if dry_run:
                    continue
                models.train_models(device=device)

                performance.score(models)
