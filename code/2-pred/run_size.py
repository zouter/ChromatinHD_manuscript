import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import tqdm.auto as tqdm

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"

design = pd.DataFrame(
    {
        "dataset": ["pbmc10k"],
        "splitter": ["5x5"],
        "method": ["v33"],
        "regions": ["100k100k"],
        "layer": ["magic"],
        "force": [False],
    }
)

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
            prediction_reference = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / "5x1" / layer / method_name
            )

            force = subdesign["force"].iloc[0]

            performance = chd.models.pred.interpret.Performance(prediction_reference / "scoring" / "performance")

            censorer = chd.models.pred.interpret.SizeCensorer(20)
            size = chd.models.pred.interpret.Size.create(
                folds,
                transcriptome,
                fragments,
                censorer,
                path=prediction.path / "scoring" / "size",
                overwrite=False,
            )

            method_info = params[method_name]

            models = chd.models.pred.model.better.Models.create(
                fragments=fragments,
                transcriptome=transcriptome,
                folds=folds,
                model_params={**method_info["model_params"], "layer": layer},
                train_params=method_info["train_params"],
                path=prediction.path,
            )

            for region in tqdm.tqdm(transcriptome.var.sort_values("dispersions_norm", ascending=False).index[:1500]):
                if models.trained(region):
                    continue

                if (region in size.scores) or (performance.scores["cor"][region, :, "test"].mean() < 0.2):
                    continue

                if dry_run:
                    continue
                models.train_models(device=device, regions_oi=[region])

                size.score(models, regions=[region])
