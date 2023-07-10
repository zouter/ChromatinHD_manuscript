import chromatinhd as chd
import pickle
import pandas as pd
import chromatinhd.models.positional.peak.prediction

from chromatinhd_manuscript.designs import (
    dataset_splitter_peakcaller_predictor_combinations as design,
)


predictors = {
    "xgboost": chromatinhd.models.positional.peak.prediction.PeaksGeneXGBoost,
    "linear": chromatinhd.models.positional.peak.prediction.PeaksGeneLinear,
    "polynomial": chromatinhd.models.positional.peak.prediction.PeaksGenePolynomial,
    "lasso": chromatinhd.models.positional.peak.prediction.PeaksGeneLasso,
}

# design = design.loc[(design["peakcaller"].str.startswith("stack"))]
# design = design.loc[~(design["peakcaller"].str.startswith("rolling_"))]
# design = design.loc[(design["peakcaller"] == "cellranger")]
# design = design.loc[~(design["peakcaller"].isin(["cellranger", "rolling_50"]))]
# design = design.loc[(design["predictor"] == "xgboost")]
# design = design.loc[(design["predictor"] == "linear")]
# design = design.loc[(design["peakcaller"].isin(["rolling_100", "rolling_50"]))]
# design = design.loc[(design["predictor"] == "lasso")]
# design = design.loc[(design["dataset"] != "alzheimer")]
design = design.loc[
    (
        design["dataset"].isin(
            ["pbmc10k", "brain", "e18brain", "pbmc10k_gran", "lymphoma"]
        )
    )
]
# design = design.loc[(design["dataset"].isin(["pbmc10k"]))]
design = design.loc[(design["promoter"] == "20kpromoter")]
# design = design.loc[(design["promoter"] == "10k10k")]
# design = design.loc[(design["promoter"] == "100k100k")]
design = design.loc[(design["splitter"] == "random_5fold")]

design["force"] = False
print(design)


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
        if promoter_name == "10k10k":
            peakcounts = chd.peakcounts.FullPeak(
                folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
            )
        else:
            peakcounts = chd.peakcounts.FullPeak(
                folder=chd.get_output()
                / "peakcounts"
                / dataset_name
                / peakcaller
                / promoter_name
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

        try:
            peakcounts.counts
        except FileNotFoundError as e:
            print(e)
            continue

        prediction.score(
            gene_peak_links,
            folds,
        )

        prediction.scores = prediction.scores
        # prediction.models = prediction.models
