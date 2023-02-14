import chromatinhd as chd
import pickle
import pandas as pd
import chromatinhd.models.positional.peak.prediction

from chromatinhd_manuscript.designs import (
    dataset_splitter_peakcaller_predictor_combinations as design,
)

design["force"] = False

predictors = {
    "xgboost": chromatinhd.models.positional.peak.prediction.PeaksGene,
    "linear": chromatinhd.models.positional.peak.prediction.PeaksGeneLinear,
    "polynomial": chromatinhd.models.positional.peak.prediction.PeaksGenePolynomial,
    "lasso": chromatinhd.models.positional.peak.prediction.PeaksGeneLasso,
}

design = design.loc[(design["peakcaller"].str.startswith("stack"))]
# design = design.loc[~(design["peakcaller"].str.startswith("rolling_"))]
# design = design.loc[(design["predictor"] == "linear")]
# design = design.loc[(design["dataset"] == "pbmc10k_gran")]
design = design.loc[(design["splitter"] == "random_5fold")]


for _, design_row in design.iterrows():
    print(design_row)
    dataset_name = design_row["dataset"]
    promoter_name = design_row["promoter"]
    peakcaller = design_row["peakcaller"]
    predictor = design_row["predictor"]
    splitter = design_row["splitter"]
    prediction_path = (
        chd.get_output()
        / "prediction_positional"
        / dataset_name
        / promoter_name
        / splitter
        / peakcaller
        / predictor
    )

    desired_outputs = [prediction_path / "scoring" / "overall" / "scores.pkl"]
    force = design_row["force"]
    if not all([desired_output.exists() for desired_output in desired_outputs]):
        force = True

    if force:
        transcriptome = chd.data.Transcriptome(
            chd.get_output() / "data" / dataset_name / "transcriptome"
        )
        peakcounts = chd.peakcounts.FullPeak(
            folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
        )

        try:
            peaks = peakcounts.peaks
        except FileNotFoundError as e:
            print(e)
            continue

        gene_peak_links = peaks.reset_index()
        gene_peak_links["gene"] = pd.Categorical(
            gene_peak_links["gene"], categories=transcriptome.adata.var.index
        )

        fragments = chromatinhd.data.Fragments(
            chd.get_output() / "data" / dataset_name / "fragments" / promoter_name
        )
        folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

        method_class = predictors[predictor]
        prediction = method_class(
            prediction_path,
            transcriptome,
            peakcounts,
        )

        prediction.score(
            gene_peak_links,
            folds,
        )

        prediction.scores = prediction.scores
        # prediction.models = prediction.models
