import pandas as pd
import numpy as np

from chromatinhd_manuscript.peakcallers import peakcallers

## Peakcaller + diffexp combinations
from chromatinhd_manuscript.designs_diff import (
    dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations as design,
)

peakcaller_diffexp_combinations = design.groupby(["peakcaller", "diffexp"]).first().index.to_frame(index=False)

peakcaller_diffexp_combinations["type"] = peakcallers.reindex(peakcaller_diffexp_combinations["peakcaller"])[
    "type"
].values
peakcaller_diffexp_combinations["color"] = peakcallers.reindex(peakcaller_diffexp_combinations["peakcaller"])[
    "color"
].values

peakcaller_diffexp_combinations = peakcaller_diffexp_combinations.set_index(["peakcaller", "diffexp"])
peakcaller_diffexp_combinations["label"] = (
    peakcaller_diffexp_combinations.index.get_level_values("diffexp")
    + "_"
    + peakcaller_diffexp_combinations.index.get_level_values("peakcaller")
)
peakcaller_diffexp_combinations["label"] = peakcallers.reindex(
    peakcaller_diffexp_combinations.index.get_level_values("peakcaller")
)["label"].values

peakcaller_diffexp_combinations = peakcaller_diffexp_combinations.sort_values(["diffexp", "type", "label"])

peakcaller_diffexp_combinations["ix"] = -np.arange(peakcaller_diffexp_combinations.shape[0])

## Peakcaller + predictor combinations
from chromatinhd_manuscript.designs_pred import (
    dataset_splitter_peakcaller_predictor_combinations as design,
)

peakcaller_predictor_combinations = design.groupby(["peakcaller", "predictor"]).first().index.to_frame(index=False)

peakcaller_predictor_combinations["type"] = peakcallers.reindex(peakcaller_predictor_combinations["peakcaller"])[
    "type"
].values
peakcaller_predictor_combinations["color"] = peakcallers.reindex(peakcaller_predictor_combinations["peakcaller"])[
    "color"
].values

peakcaller_predictor_combinations = peakcaller_predictor_combinations.set_index(["peakcaller", "predictor"])
peakcaller_predictor_combinations["label"] = (
    peakcaller_predictor_combinations.index.get_level_values("peakcaller")
    + "/"
    + peakcaller_predictor_combinations.index.get_level_values("predictor")
)
peakcaller_predictor_combinations["label"] = peakcallers.reindex(
    peakcaller_predictor_combinations.index.get_level_values("peakcaller")
)["label"].values

## Simulation methods

from chromatinhd_manuscript.designs_pred import dataset_splitter_simulation_combinations

simulation_predictor_methods = pd.DataFrame(
    {"method": dataset_splitter_simulation_combinations["method"].unique()}
).set_index("method")
simulation_predictor_methods["color"] = "#AAAAAA"

## Baseline methods


from chromatinhd_manuscript.designs_pred import dataset_splitter_baselinemethods_combinations

baseline_predictor_methods = pd.DataFrame(
    {"method": dataset_splitter_baselinemethods_combinations["method"].unique()}
).set_index("method")
baseline_predictor_methods["color"] = "#333333"
baseline_predictor_methods["type"] = "baseline"
baseline_predictor_methods.loc["baseline_v42", "label"] = "ArchR 42"
baseline_predictor_methods.loc["baseline_v21", "label"] = "ArchR 21"

## Prediction Methods

peakcaller_diffexp_methods = peakcaller_diffexp_combinations.copy().reset_index()
peakcaller_diffexp_methods.index = pd.Series(
    [
        (peakcaller + "/" + diffexp)
        for peakcaller, diffexp in zip(
            peakcaller_diffexp_methods["peakcaller"],
            peakcaller_diffexp_methods["diffexp"],
        )
    ],
    name="method",
)


peakcaller_predictor_methods = peakcaller_predictor_combinations.copy().reset_index()
peakcaller_predictor_methods.index = pd.Series(
    [
        (peakcaller + "/" + predictor)
        for peakcaller, predictor in zip(
            peakcaller_predictor_methods["peakcaller"],
            peakcaller_predictor_methods["predictor"],
        )
    ],
    name="method",
)

chromatinhd_color = "#0074D9"

differential_methods = peakcaller_diffexp_methods
differential_methods.loc["chd", "label"] = "ChromatinHD"
differential_methods.loc["chd", "color"] = chromatinhd_color


## Prediction methods
prediction_methods = pd.concat([peakcaller_predictor_methods, simulation_predictor_methods, baseline_predictor_methods])

prediction_methods.loc["v31", "label"] = "ChromatinHD"
prediction_methods.loc["v31", "color"] = chromatinhd_color
prediction_methods.loc["v31", "type"] = "ours"

prediction_methods.loc["v32", "label"] = "ChromatinHD"
prediction_methods.loc["v32", "color"] = chromatinhd_color
prediction_methods.loc["v32", "type"] = "ours"

prediction_methods.loc["v33", "label"] = "ChromatinHD"
prediction_methods.loc["v33", "color"] = chromatinhd_color
prediction_methods.loc["v33", "type"] = "ours"
