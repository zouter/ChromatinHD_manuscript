# Script to predict and score performance of CRE-based models for prediction of gene expression using test cells

import chromatinhd as chd
import pickle
import pandas as pd
import chromatinhd.models.pred.model.peakcounts
import chromatinhd.data.peakcounts

from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_splitter_peakcaller_predictor_combinations as design,
)

design["dataset"] = design["testdataset"]

# design = design.loc[(design["dataset"].isin(["pbmc10k_gran-pbmc10k"]))]

# design = design.loc[(design["dataset"].isin(["pbmc10kx-pbmc10k"]))]
# design = design.loc[(design["dataset"].isin(["lymphoma-pbmc10k", "pbmc3k-pbmc10k"]))]
design = design.loc[(design["splitter"] == "5x1")]
design = design.loc[(design["layer"] == "magic")]
design = design.loc[(design["predictor"] == "xgboost")]
# design = design.loc[(design["peakcaller"] == "encode_screen")]
# design = design.loc[(design["peakcaller"] == "macs2_improved")]
# design = design.loc[(design["peakcaller"] == "macs2_leiden_0.1_merged")]
# design = design.loc[(design["peakcaller"].str.startswith("rolling"))]

dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

print(design)

for _, design_row in design.iterrows():
    print(design_row)
    dataset_name = design_row["dataset"]
    traindataset_name = design_row["traindataset"]
    regions_name = design_row["regions"]
    peakcaller = design_row["peakcaller"]
    layer = design_row["layer"]
    predictor = design_row["predictor"]
    splitter = design_row["splitter"]
    prediction_path = (
        chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / peakcaller / predictor
    )

    force = design_row["force"]

    prediction = chd.models.pred.model.peakcounts.PredictionTest(prediction_path, reset=force)

    if not dry_run:
        try:
            peakcounts = chd.flow.Flow.from_path(
                chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name
            )
            train_peakcounts = chd.flow.Flow.from_path(
                chd.get_output() / "datasets" / traindataset_name / "peakcounts" / peakcaller / regions_name
            )
            print(train_peakcounts, peakcounts)
        except FileNotFoundError:
            print("Not found: ", dataset_name, regions_name, peakcaller)
            continue
        if not peakcounts.counted:
            print("Not counted: ", peakcounts)
            continue
        train_transcriptome = chd.data.Transcriptome(
            chd.get_output() / "datasets" / traindataset_name / "transcriptome"
        )
        transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
        folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)
        prediction.initialize(train_peakcounts, train_transcriptome, peakcounts, transcriptome, folds)
        prediction.score(layer=layer, predictor=predictor)
