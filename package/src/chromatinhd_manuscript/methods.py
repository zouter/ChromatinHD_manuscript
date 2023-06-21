import pandas as pd
import numpy as np

from chromatinhd_manuscript.peakcallers import peakcallers

## Peakcaller + diffexp combinations
from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations as design,
)

peakcaller_diffexp_combinations = (
    design.groupby(["peakcaller", "diffexp"]).first().index.to_frame(index=False)
)

peakcaller_diffexp_combinations["type"] = peakcallers.reindex(
    peakcaller_diffexp_combinations["peakcaller"]
)["type"].values
peakcaller_diffexp_combinations["color"] = peakcallers.reindex(
    peakcaller_diffexp_combinations["peakcaller"]
)["color"].values

# peakcaller_diffexp_combinations.loc[
#     peakcaller_diffexp_combinations["type"] == "baseline", "color"
# ] = "#888888"
# peakcaller_diffexp_combinations.loc[
#     peakcaller_diffexp_combinations["type"] == "ours", "color"
# ] = "#0074D9"
# peakcaller_diffexp_combinations.loc[
#     peakcaller_diffexp_combinations["type"] == "rolling", "color"
# ] = "#FF851B"
# peakcaller_diffexp_combinations.loc[
#     peakcaller_diffexp_combinations["type"] == "peak", "color"
# ] = "#FF4136"
# peakcaller_diffexp_combinations.loc[
#     peakcaller_diffexp_combinations["type"] == "predefined", "color"
# ] = "#2ECC40"
# peakcaller_diffexp_combinations.loc[
#     pd.isnull(peakcaller_diffexp_combinations["color"]), "color"
# ] = "#DDDDDD"

peakcaller_diffexp_combinations = peakcaller_diffexp_combinations.set_index(
    ["peakcaller", "diffexp"]
)
peakcaller_diffexp_combinations["label"] = (
    peakcaller_diffexp_combinations.index.get_level_values("diffexp")
    + "_"
    + peakcaller_diffexp_combinations.index.get_level_values("peakcaller")
)
peakcaller_diffexp_combinations["label"] = peakcallers.reindex(
    peakcaller_diffexp_combinations.index.get_level_values("peakcaller")
)["label"].values

peakcaller_diffexp_combinations = peakcaller_diffexp_combinations.sort_values(
    ["diffexp", "type", "label"]
)

peakcaller_diffexp_combinations["ix"] = -np.arange(
    peakcaller_diffexp_combinations.shape[0]
)

## Peakcaller + predictor combinations
from chromatinhd_manuscript.designs import (
    dataset_splitter_peakcaller_predictor_combinations as design,
)

peakcaller_predictor_combinations = (
    design.groupby(["peakcaller", "predictor"]).first().index.to_frame(index=False)
)

peakcaller_predictor_combinations["type"] = peakcallers.reindex(
    peakcaller_predictor_combinations["peakcaller"]
)["type"].values
peakcaller_predictor_combinations["color"] = peakcallers.reindex(
    peakcaller_predictor_combinations["peakcaller"]
)["color"].values

# peakcaller_predictor_combinations.loc[
#     peakcaller_predictor_combinations["type"] == "baseline", "color"
# ] = "#888888"
# peakcaller_predictor_combinations.loc[
#     peakcaller_predictor_combinations["type"] == "ours", "color"
# ] = "#0074D9"
# peakcaller_predictor_combinations.loc[
#     peakcaller_predictor_combinations["type"] == "rolling", "color"
# ] = "#FF851B"
# peakcaller_predictor_combinations.loc[
#     peakcaller_predictor_combinations["type"] == "peak", "color"
# ] = "#FF4136"
# peakcaller_predictor_combinations.loc[
#     peakcaller_predictor_combinations["type"] == "predefined", "color"
# ] = "#2ECC40"
# peakcaller_predictor_combinations.loc[
#     pd.isnull(peakcaller_predictor_combinations["color"]), "color"
# ] = "#DDDDDD"


peakcaller_predictor_combinations = peakcaller_predictor_combinations.set_index(
    ["peakcaller", "predictor"]
)
peakcaller_predictor_combinations["label"] = (
    peakcaller_predictor_combinations.index.get_level_values("peakcaller")
    + "/"
    + peakcaller_predictor_combinations.index.get_level_values("predictor")
)
peakcaller_predictor_combinations["label"] = peakcallers.reindex(
    peakcaller_predictor_combinations.index.get_level_values("peakcaller")
)["label"].values

## Methods

peakcaller_diffexp_methods = peakcaller_diffexp_combinations.copy().reset_index()
peakcaller_diffexp_methods.index = pd.Series(
    [
        (peakcaller + "_" + diffexp)
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


methods = pd.concat(
    [peakcaller_diffexp_methods, peakcaller_predictor_methods], names=["method"]
)
methods.loc["ChromatinHD", "label"] = "ChromatinHD"
methods.loc["ChromatinHD", "color"] = "#0074D9"
methods.loc["ChromatinHD", "label"] = "ChromatinHD"
methods.loc["v20", "color"] = "#0074D9"


## Prediction methods
prediction_methods = pd.concat([peakcaller_predictor_methods])
prediction_methods.loc["v20", "label"] = "ChromatinHD"
prediction_methods.loc["v20", "color"] = "#0074D9"
prediction_methods.loc["v20", "type"] = "ours"
