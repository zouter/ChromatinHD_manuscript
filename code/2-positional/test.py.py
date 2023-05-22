# %%
import IPython

if IPython.get_ipython():
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
    IPython.get_ipython().run_line_magic("matplotlib", "inline")
    IPython.get_ipython().run_line_magic(
        "config", "InlineBackend.figure_format = 'retina'"
    )

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
design = design.loc[(design["predictor"] == "xgboost")]
# design = design.loc[(design["predictor"] == "lasso")]
design = design.loc[(design["dataset"] != "alzheimer")]
design = design.loc[(design["promoter"] == "10k10k")]
# design = design.loc[(design["promoter"] == "100k100k")]
# design = design.loc[(design["splitter"] == "random_5fold")]

design["force"] = False
print(design)

# %%
design_row = design.query(
    "dataset == 'brain' and peakcaller == 'cellranger' and predictor == 'xgboost' and splitter == 'random_5fold'"
).iloc[0]

# %%
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
# %%
transcriptome = chd.data.Transcriptome(
    chd.get_output() / "data" / dataset_name / "transcriptome"
)
peakcounts = chd.peakcounts.FullPeak(
    folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
)

peaks = peakcounts.peaks

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

# %%
prediction.score(
    gene_peak_links,
    folds[:1],
)
# %%
prediction.scores
# %%
import scipy
import numpy as np
import tqdm

peak_gene_links = gene_peak_links

self = prediction

if scipy.sparse.issparse(self.transcriptome.adata.X):
    X_transcriptome = self.transcriptome.adata.X.tocsc()
else:
    X_transcriptome = scipy.sparse.csc_matrix(self.transcriptome.adata.X)
# X_transcriptome = scipy.sparse.csc_matrix(
#     self.transcriptome.adata.layers["magic"]
# )  #! Use magic

X_peaks = self.peaks.counts.tocsc()

var_transcriptome = self.transcriptome.var
var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

var_peaks = self.peaks.var
var_peaks["ix"] = np.arange(var_peaks.shape[0])


def extract_data(gene_oi, peaks_oi):
    x = np.array(X_peaks[:, peaks_oi["ix"]].todense())
    y = np.array(X_transcriptome[:, gene_oi["ix"]].todense())[:, 0]
    return x, y


obs = self.transcriptome.obs
obs["ix"] = np.arange(obs.shape[0])

scores = []

for fold_ix, fold in enumerate(folds):
    train_ix, validation_ix, test_ix = (
        fold["cells_train"],
        fold["cells_validation"],
        fold["cells_test"],
    )

    for gene, peak_gene_links_oi in tqdm.tqdm(peak_gene_links.groupby("gene")):
        peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
        gene_oi = var_transcriptome.loc[gene]

        x, y = extract_data(gene_oi, peaks_oi)

        self.regressor = self._create_regressor(len(obs), train_ix, validation_ix)

        x = self._preprocess_features(x)
        x[np.isnan(x)] = 0.0

        break
    break

# %%
import xgboost

regressor = xgboost.XGBRegressor(n_estimators=100, early_stopping_rounds=50)
# %%
regressor.fit(
    x[train_ix], y[train_ix],
    
)
# %%
eval_set = [(x[validation_ix], y[validation_ix])]
regressor.fit(
    x[train_ix], y[train_ix], eval_set=eval_set, verbose=False
)
predicted = regressor.predict(x)
# %%
