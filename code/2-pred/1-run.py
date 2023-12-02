import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import pickle

device = "cuda:0"

from chromatinhd_manuscript.designs_pred import (
    dataset_folds_method_combinations as design,
)

design = design.query("folds == '5x1'")
# design = design.query("folds == 'random_5fold'")
# design = design.query("splitter == 'permutations_5fold5repeat'")
# design = design.loc[design["method"].isin(["v20", "counter"])]
design = design.query("method == 'v30'")
# design = design.query("method == 'counter'")
# design = design.query("dataset == 'pbmc10k'")
design = design.query("dataset == 'lymphoma'")
# design = design.query("dataset == 'hspc'")
# design = design.query("layer == 'normalized'")
design = design.query("layer == 'magic'")
design = design.query("regions == '100k100k'")
# design = design.query("regions == '20kpromoter'")
# design = design.query("regions == '100k100k'")

design = design.copy()
design["force"] = False

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    print(f"{dataset_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    for (folds_name, layer), subdesign in subdesign.groupby(["folds", "layer"]):
        # create design to run
        from chromatinhd.models.pred.model.design import get_design

        methods_info = get_design(transcriptome, fragments)

        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / folds_name)

        for method_name, subdesign in subdesign.groupby("method"):
            method_info = methods_info[method_name]

            print(subdesign)

            models = chd.models.pred.model.better.Models.create(
                fragments=fragments,
                transcriptome=transcriptome,
                folds=folds,
                path=chd.get_output() / "pred" / dataset_name / regions_name / folds_name / layer / method_name,
                # reset = True,
                model_params={**method_info["model_params"], "layer": layer},
                train_params=method_info["train_params"],
            )
            models.train_models()
