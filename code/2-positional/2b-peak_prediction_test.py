import chromatinhd as chd
import pickle
import pandas as pd
import chromatinhd.models.positional.peak.prediction_test

from chromatinhd_manuscript.designs import (
    traindataset_testdataset_splitter_peakcaller_predictor_combinations as design,
)

design["force"] = True

predictors = {
    "linear": chromatinhd.models.positional.peak.prediction_test.PeaksGeneLinear,
}

# design = design.loc[~(design["peakcaller"].str.startswith("rolling"))]
# design = design.loc[(design["peakcaller"].str.startswith("macs2_leiden_0.1"))]
design = design.loc[(design["predictor"] == "linear")]
# design = design.loc[(design["testdataset"] != "pbmc3k-pbmc10k")]
design = design.loc[(design["splitter"] == "random_5fold")]


for _, design_row in design.iterrows():
    print(design_row)
    traindataset_name = design_row["traindataset"]
    testdataset_name = design_row["testdataset"]
    promoter_name = design_row["promoter"]
    peakcaller = design_row["peakcaller"]
    predictor = design_row["predictor"]
    trainsplitter = design_row["splitter"]
    prediction_path = (
        chd.get_output()
        / "prediction_positional"
        / testdataset_name
        / promoter_name
        / "all"
        / peakcaller
        / predictor
    )

    desired_outputs = [prediction_path / "scoring" / "overall" / "scores.pkl"]
    force = design_row["force"]
    if not all([desired_output.exists() for desired_output in desired_outputs]):
        force = True

    if force:
        traintranscriptome = chd.data.Transcriptome(
            chd.get_output() / "data" / traindataset_name / "transcriptome"
        )
        trainpeakcounts = chd.peakcounts.FullPeak(
            folder=chd.get_output() / "peakcounts" / traindataset_name / peakcaller
        )

        testtranscriptome = chd.data.Transcriptome(
            chd.get_output() / "data" / testdataset_name / "transcriptome"
        )
        testpeakcounts = chd.peakcounts.FullPeak(
            folder=chd.get_output() / "peakcounts" / testdataset_name / peakcaller
        )

        try:
            testpeakcounts.peaks
        except FileNotFoundError as e:
            print(e)
            continue

        peaks = trainpeakcounts.peaks
        gene_peak_links = peaks.reset_index()
        gene_peak_links["gene"] = pd.Categorical(
            gene_peak_links["gene"], categories=traintranscriptome.adata.var.index
        )

        fragments = chromatinhd.data.Fragments(
            chd.get_output() / "data" / traindataset_name / "fragments" / promoter_name
        )
        folds = pickle.load(
            (fragments.path / "folds" / (trainsplitter + ".pkl")).open("rb")
        )[:1]

        method_class = predictors[predictor]
        prediction = method_class(
            prediction_path,
            traintranscriptome,
            trainpeakcounts,
            testtranscriptome,
            testpeakcounts,
        )

        prediction.score(
            gene_peak_links,
            folds,
        )

        prediction.scores = prediction.scores
