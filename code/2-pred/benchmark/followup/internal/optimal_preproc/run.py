# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %%
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
dataset_name = "pbmc10k"
transcriptome_original = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "10k10k")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")
fold = folds[1]

# %%
transcriptomes = [chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome") for dataset_name in ["pbmc10k", "pbmc10k_int"]]

# %%
all_genes = set(transcriptome_original.var.index)
for transcriptome in transcriptomes:
    all_genes = all_genes & set(transcriptome.var.index)

# %%
genes_oi1 = transcriptomes[1].var.loc[list(all_genes)].query("n_cells > 500").sort_values("dispersions_norm", ascending = False).index[:20]
genes_oi2 = transcriptomes[0].var.loc[list(all_genes)].query("n_cells > 500").sort_values("dispersions_norm", ascending = False).index[:20]
genes_oi = set(transcriptome_original.gene_id(["CCL4", "IL1B", "CD79A", "QKI", "CD79B"])) | set(genes_oi2) | set(genes_oi1)
print(len(genes_oi))

# %%
gene_oi = transcriptome_original.gene_id("IL1B")

# %%
# datasets = {}
# for regions_name in ["100k100k", "10k10k"]:
#     fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
#     for gene_oi in genes_oi:
#         if not (chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / regions_name / "fragments").exists():
#             fragments = fragments_original.filter_regions(
#                 fragments_original.regions.filter(
#                     [gene_oi], path=chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / "regions" / regions_name
#                 ),
#                 path=chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / regions_name / "fragments",
#             )
#             fragments.create_regionxcell_indptr()

# for gene_oi in genes_oi:
#     if not (chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / "transcriptome").exists():
#         transcriptome = transcriptome_original.filter_genes(
#             [gene_oi], path=chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / "transcriptome"
#         )

# %% [markdown]
# ## Model per gene

# %%
from params import params

# %%
import pickle
import logging
chd.models.pred.trainer.trainer.logger.handlers = []
chd.models.pred.trainer.trainer.logger.propagate = False

# %%
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/*/*/7bd6b1354af76bfa853313c822cb26cb
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/10k10k/*/e4ba9692843da9372d6a79c5570fa0c3
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name}/*/7bd6b1354af76bfa853313c822cb26cb
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/10k10k/ENSG00000158813/7bd6b1354af76bfa853313c822cb26cb

# %%
layer = "magic"

# %%
for param_id, param in params.items():
    param = param.copy()
    cls = param.pop("cls")

    train_params = {}
    if "n_cells_step" in param:
        train_params["n_cells_step"] = param.pop("n_cells_step")
    if "lr" in param:
        train_params["lr"] = param.pop("lr")

    if "label" in param:
        param.pop("label")
    for dataset_name in ["pbmc10k", "pbmc10k_int"]:
        for regions_name in ["10k10k", "100k100k"]:
            for gene_oi in genes_oi:
                model_folder = chd.get_output() / "models" / "mini" / dataset_name / regions_name / gene_oi / param_id

                if (model_folder / "performance").exists():
                    continue

                print(param_id, dataset_name, regions_name, gene_oi)
                transcriptome = chd.data.Transcriptome(path=chd.get_output() / "datasets" / dataset_name / "transcriptome")
                fragments = chd.data.Fragments(path=chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

                model = cls(
                    fragments = fragments,
                    transcriptome=transcriptome,
                    fold = fold,
                    layer = layer,
                    regions_oi = [gene_oi],
                    **param,
                )
                model.train_model(**train_params, pbar = True)
                performance = chd.models.pred.interpret.Performance(path = model_folder / "performance")
                performance.score(fragments, transcriptome, [model], [fold], pbar = False)

                trace = {
                    "n_train_checkpoints":len(model.trace.train_steps)
                }
                pickle.dump(trace, open(model_folder / "trace.pkl", "wb"))

# %% [markdown]
# ## Peakcounts

# %%
from params_peakcounts import params_peakcounts
import pickle
def r2(y, y_predicted, y_train):
    return 1 - ((y_predicted - y) ** 2).sum() / ((y - y_train.mean()) ** 2).sum()


# %%
import chromatinhd.data.peakcounts

for dataset_name in dataset_names:
    for param_id, param in params_peakcounts.items():
        param = param.copy()

        for regions_name in ["10k10k", "100k100k"]:
            peakcounts = chd.data.peakcounts.PeakCounts(
                path=chd.get_output() / "datasets" / dataset_name / "peakcounts" / param["peakcaller"] / regions_name
            )

            for gene_oi in genes_oi:
                model_folder = chd.get_output() / "models" / "mini" / dataset_name / regions_name / gene_oi / param_id

                if (model_folder / "performance" / "scores.pkl").exists():
                    continue

                import chromatinhd.data.peakcounts
                import sklearn.linear_model

                peak_ids = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi]["peak"]
                print(len(peak_ids), gene_oi)
                # peak_ids = pd.Series([peak_ids[3]])
                peak_ixs = peakcounts.var.loc[peak_ids, "ix"]

                if len(peak_ixs) > 0:
                    x = np.array(peakcounts.counts[:, peak_ixs].todense())
                    y = transcriptome_original.layers[layer][:, transcriptome_original.var.index == gene_oi][:, 0]

                    x_train = x[fold["cells_train"]]
                    x_validation = x[fold["cells_validation"]]
                    x_test = x[fold["cells_test"]]

                    y_train = y[fold["cells_train"]]
                    y_validation = y[fold["cells_validation"]]
                    y_test = y[fold["cells_test"]]

                    if param["predictor"] == "linear":
                        lm = sklearn.linear_model.LinearRegression()
                    elif param["predictor"] == "lasso":
                        lm = sklearn.linear_model.LassoCV(n_alphas = 10)
                    # lm = sklearn.linear_model.RidgeCV(alphas = 10)
                    lm.fit(x_train, y_train)

                    # raise ValueError()

                    cors = []
                    r2s = []

                    y_predicted = lm.predict(x_train)
                    cors.append(np.corrcoef(y_train, y_predicted)[0, 1])
                    r2s.append(r2(y_train, y_predicted, y_train))

                    y_predicted = lm.predict(x_validation)
                    cors.append(np.corrcoef(y_validation, y_predicted)[0, 1])
                    r2s.append(r2(y_validation, y_predicted, y_train))

                    y_predicted = lm.predict(x_test)
                    cors.append(np.corrcoef(y_test, y_predicted)[0, 1])
                    r2s.append(r2(y_test, y_predicted, y_train))
                else:
                    cors = [0, 0, 0]
                    r2s = [0, 0, 0]

                score = xr.Dataset({
                    "cor":xr.DataArray(np.array(cors)[None, :], coords = {"model":pd.Index([0], name = "model"), "phase":pd.Index(["train", "validation", "test"], name = "phase")}),
                    "r2":xr.DataArray(np.array(r2s)[None, :], coords = {"model":pd.Index([0], name = "model"), "phase":pd.Index(["train", "validation", "test"], name = "phase")})
                })

                performance_folder = model_folder / "performance"
                performance_folder.mkdir(parents = True, exist_ok = True)
                pickle.dump(score, open(performance_folder / "scores.pkl", "wb"))

# %% [markdown]
# ## Compare

# %%
from params import params
from params_peakcounts import params_peakcounts
import pickle

# %%
dataset_names = ["pbmc10k", "pbmc10k_int"]

# %%
scores = []
for dataset_name in dataset_names:
    for param_id, param in {**params, **params_peakcounts}.items():
        for gene_oi in genes_oi:
            for regions_name in ["10k10k", "100k100k"]:
                model_folder = chd.get_output() / "models" / "mini" / dataset_name / regions_name / gene_oi / param_id
                performance_folder = model_folder / "performance"
                if (performance_folder / "scores.pkl").exists():
                    score = pickle.load(open(performance_folder / "scores.pkl", "rb"))
                    score = score.mean("model").to_dataframe().reset_index()
                    score["param"] = param_id
                    score["gene"] = gene_oi
                    score["regions"] = regions_name
                    score["dataset"] = dataset_name
                    scores.append(score)
scores = pd.concat(scores).set_index(["dataset", "gene", "regions", "param", "phase"])
scores = xr.Dataset.from_dataframe(scores)
scores["r2"] = scores["cor"]**2

# %%
param_summary = pd.DataFrame({**params, **params_peakcounts}).T
param_summary["label"] = [param["label"] if str(param["label"]) != "nan" else param_id for param_id, param in param_summary.iterrows()]

scores.coords["param"] = pd.Index(param_summary["label"].values, name = "param")

# %%
diff = scores.sel(param = "radial_binary_1000-31frequencies_directdistance") - scores.sel(param = "peaks_main")

# %%
diff.sel(dataset = "pbmc10k").sel(phase = "test")["r2"].mean()

# %%
diff.sel(dataset = "pbmc10k_int").sel(phase = "test")["r2"].mean()
