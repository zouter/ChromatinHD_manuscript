# Script to predict and score performance of CRE-based models for prediction of gene expression using test cells

import chromatinhd as chd
import pickle
import pandas as pd
import chromatinhd.models.pred.model.peakcounts
import chromatinhd.data.peakcounts

from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_peakcaller_predictor_combinations as design,
)

# design = design.loc[(design["peakcaller"].str.startswith("stack"))]
# design = design.loc[~(design["peakcaller"].str.startswith("rolling_"))]
# design = design.loc[(design["peakcaller"] == "macs2_leiden_0.1_merged")]
# design = design.loc[(design["peakcaller"].isin(["rolling_100"]))]
# design = design.loc[(design["peakcaller"].isin(["cellranger", "rolling_500"]))]
# design = design.loc[(design["predictor"] == "linear")]
# design = design.loc[(design["predictor"].isin(["linear", "lasso"]))]
# design = design.loc[(design["predictor"] == "linear_magic")]
# design = design.loc[(design["predictor"] == "lasso_magic")]
# design = design.loc[(design["peakcaller"].isin(["gene_body"]))]
# design = design.loc[(design["predictor"] == "lasso")]
# design = design.loc[(design["predictor"] == "xgboost")]
# design = design.loc[(design["dataset"] != "alzheimer")]
design = design.loc[
    (
        design["dataset"].isin(
            ["hspc", "pbmc10k"]
            # ["e18brain"]
            # ["lymphoma"]
            # ["pbmc10k", "brain", "e18brain", "pbmc10k_gran", "lymphoma"]
        )
    )
]
# design = design.loc[(design["dataset"].isin(["pbmc10k"]))]
# design = design.loc[(design["regions"] == "20kpromoter")]
design = design.loc[(design["regions"] == "10k10k")]
# design = design.loc[(design["regions"] == "100k100k")]
design = design.loc[(design["splitter"] == "5x1")]

dry_run = False
design["force"] = False
print(design)

# design["force"] = True
# dry_run = True

for _, design_row in design.iterrows():
    print(design_row)
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
        transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
        folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)
        prediction.initialize(peakcounts, transcriptome, folds)
        prediction.score(layer=layer, predictor=predictor)
