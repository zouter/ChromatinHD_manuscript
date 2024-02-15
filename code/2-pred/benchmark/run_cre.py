# Script to predict and score performance of CRE-based models for prediction of gene expression using test cells

import chromatinhd as chd
import pickle
import pandas as pd
import chromatinhd.models.pred.model.peakcounts
import chromatinhd.data.peakcounts

from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_peakcaller_predictor_combinations as design,
)

# design = design.loc[(design["dataset"].isin(["hspc"]))]
# design = design.loc[~(design["dataset"].isin(["pbmc20k"]))]
# design = design.loc[(design["dataset"].isin(["alzheimer"]))]
# design = design.loc[(design["regions"] == "20kpromoter")]
# design = design.loc[(design["regions"] == "10k10k")]
# design = design.loc[(design["regions"] == "100k100k")]
design = design.loc[(design["layer"] == "magic")]
design = design.loc[(design["splitter"] == "5x1")]
# design = design.loc[(design["peakcaller"] == "macs2_leiden_0.1_merged")]
# design = design.loc[(design["peakcaller"] == "encode_screen")]
# design = design.loc[(design["peakcaller"] == "rolling_500")]
design = design.loc[(design["predictor"] == "lasso")]

dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

print(design)

for _, design_row in design.iterrows():
    dataset_name = design_row["dataset"]
    regions_name = design_row["regions"]
    peakcaller = design_row["peakcaller"]
    splitter = design_row["splitter"]
    layer = design_row["layer"]
    predictor = design_row["predictor"]
    prediction_path = (
        chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / peakcaller / predictor
    )

    force = design_row["force"]

    prediction = chd.models.pred.model.peakcounts.Prediction(prediction_path, reset=force)

    if not dry_run:
        try:
            peakcounts = chd.flow.Flow.from_path(
                chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name
            )
        except FileNotFoundError:
            print("Not found: ", dataset_name, regions_name, peakcaller)
            continue
        if not peakcounts.counted:
            print("Not counted: ", peakcounts)
            continue

        print(design_row)
        transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
        folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)
        prediction.initialize(peakcounts, transcriptome, folds)
        prediction.score(layer=layer, predictor=predictor)
