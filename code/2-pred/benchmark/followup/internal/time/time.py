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
import chromatinhd.data.gradient

# %%
dataset_name = "hspc_gmp"
transcriptome = transcriptome_original = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "10k10k")
gradient = chd.data.gradient.Gradient(chd.get_output() / "datasets" / dataset_name / "gradient")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")
fold = folds[1]

# %%
import chromatinhd.loaders.transcriptome_fragments_time

# %%
import torch

# %%
adata = transcriptome.adata.copy()
import scanpy as sc
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# %%
genes_oi2 = transcriptome.var.query("means > 0.7").index
genes_oi3 = transcriptome.var.sort_values("dispersions", ascending = False).index[:50]
genes_oi1 = transcriptome.gene_id(["MPO", "CD74"]).tolist()
genes_oi = list(set(genes_oi1) | set(genes_oi2) | set(genes_oi3))
print(len(genes_oi))

# %%
sc.pl.umap(transcriptome.adata, color = genes_oi3[:10], use_raw = False)

# %%
# delta_times = [-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3]
# fig, ax = plt.subplots()
# cors_all = []
# for gene_oi in genes_oi:
#     cors = []
#     for delta_time in delta_times:
#         loader = chromatinhd.loaders.transcriptome_fragments_time.TranscriptomeFragmentsTime(
#             fragments = fragments,
#             transcriptome=transcriptome,
#             gradient=gradient,
#             cellxregion_batch_size=10000,
#             delta_time = delta_time,
#             n_bins = 10,
#         )
#         minibatch = chromatinhd.loaders.minibatches.Minibatch(np.arange(len(transcriptome.obs)), np.array([transcriptome.var.index.get_loc(gene_oi)]))

#         result = loader.load(minibatch)
#         n_fragments = torch.bincount(result.fragments.local_cell_ix, minlength=result.minibatch.n_cells)
#         # n_fragments = transcriptome.layers["normalized"][result.minibatch.cells_oi, :][:, result.minibatch.genes_oi][:, 0]
#         cor = np.corrcoef(result.transcriptome.value[:, 0], n_fragments)[0, 1]
#         cors.append(cor)
#     ax.plot(delta_times, cors)
#     cors_all.append(cors)

# # gene_ix = transcriptome.gene_ix("MPO")
# # sns.regplot(x = gradient.values[:, 0], y = transcriptome.layers["normalized"][:, gene_ix])
# # sns.regplot(x = gradient.values[:, 0], y = fragments.counts[:, gene_ix])

# fig, ax = plt.subplots()
# plt.plot(delta_times, np.stack(cors_all).mean(0))
# ax.axvline(0)

# %% [markdown]
# ## Model

# %%
from params import params

# %%
import logging
chd.models.pred.trainer.trainer.logger.handlers = []
chd.models.pred.trainer.trainer.logger.propagate = False

# %%
import pickle

# %%
dataset_name = "hspc_gmp"

# %%
layer = "magic"

# %%
pbar = tqdm.tqdm()
for param_id, param in params.items():
    param = param.copy()
    cls = param.pop("cls")

    train_params = {}
    if "n_cells_step" in param:
        train_params["n_cells_step"] = param.pop("n_cells_step")
    if "lr" in param:
        train_params["lr"] = param.pop("lr")

    if "layer" in param:
        layer = param.pop("layer")
    else:
        layer = "normalized"

    if "label" in param:
        param.pop("label")
    for regions_name in ["10k10k", "100k100k"]:
        for gene_oi in genes_oi:
            model_folder = chd.get_output() / "models" / "mini" / dataset_name / regions_name / gene_oi / param_id

            if (model_folder / "performance").exists():
                continue

            pbar.update(1)
            
            fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

            if "pret" in str(cls):
                model = cls(
                    fragments = fragments,
                    transcriptome=transcriptome,
                    gradient = gradient,
                    fold = fold,
                    layer = layer,
                    regions_oi = [gene_oi],
                    **param,
                )
            else:
                model = cls(
                    fragments = fragments,
                    transcriptome=transcriptome,
                    fold = fold,
                    layer = "magic",
                    regions_oi = [gene_oi],
                    **param,
                )
            model.train_model(**train_params, pbar = True)

            # break
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
from methods_peakcounts import *
import pickle

# %%
dataset_name = "hspc_gmp"

# %%
layer = "magic"

import chromatinhd.data.peakcounts

for param_id, param in params_peakcounts.items():
    param = param.copy()

    dt = None
    if "dt" in param:
        dt = param.pop("dt")

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
            import sklearn.ensemble

            peak_ids = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi]["peak"]
            print(len(peak_ids), gene_oi)
            # peak_ids = pd.Series([peak_ids[3]])
            peak_ixs = peakcounts.var.loc[peak_ids, "ix"]

            if len(peak_ixs) > 0:
                x = np.array(peakcounts.counts[:, peak_ixs].todense())

                if dt is None:
                    y = transcriptome_original.layers[layer][:, transcriptome_original.var.index == gene_oi][:, 0]
                else:
                    minibatch = chromatinhd.loaders.minibatches.Minibatch(
                        np.arange(len(transcriptome.obs)), np.array([transcriptome.var.index.get_loc(gene_oi)])
                    )
                    loader = chromatinhd.loaders.transcriptome_fragments_time.TranscriptomeTime(
                        transcriptome, gradient, delta_time=dt, layer="normalized"
                    )
                    result = loader.load(minibatch)
                    y = result.value.numpy()[:, 0]

                cells_train = np.hstack([fold["cells_train"]])

                x_train = x[cells_train]
                x_validation = x[fold["cells_validation"]]
                x_test = x[fold["cells_test"]]

                y_train = y[cells_train]
                y_validation = y[fold["cells_validation"]]
                y_test = y[fold["cells_test"]]

                if param["predictor"] == "linear":
                    lm = sklearn.linear_model.LinearRegression()
                    lm.fit(x_train, y_train)
                else:  # CV
                    if param["predictor"] == "lasso":
                        lm = lasso_cv(x_train, y_train, x_validation, y_validation)
                    elif param["predictor"] == "rf":
                        lm = rf_cv(x_train, y_train, x_validation, y_validation)
                    elif param["predictor"] == "ridge":
                        lm = sklearn.linear_model.RidgeCV(alphas=10)

                cors = []
                r2s = []

                y_predicted = lm.predict(x_train)
                cors.append(np.corrcoef(y_train, y_predicted)[0, 1])
                r2s.append(calculate_r2(y_train, y_predicted, y_train))

                y_predicted = lm.predict(x_validation)
                cors.append(np.corrcoef(y_validation, y_predicted)[0, 1])
                r2s.append(calculate_r2(y_validation, y_predicted, y_train))

                y_predicted = lm.predict(x_test)
                cors.append(np.corrcoef(y_test, y_predicted)[0, 1])
                r2s.append(calculate_r2(y_test, y_predicted, y_train))
            else:
                cors = [0, 0, 0]
                r2s = [0, 0, 0]

            score = xr.Dataset(
                {
                    "cor": xr.DataArray(
                        np.array(cors)[None, :],
                        coords={
                            "model": pd.Index([0], name="model"),
                            "phase": pd.Index(["train", "validation", "test"], name="phase"),
                        },
                    ),
                    "r2": xr.DataArray(
                        np.array(r2s)[None, :],
                        coords={
                            "model": pd.Index([0], name="model"),
                            "phase": pd.Index(["train", "validation", "test"], name="phase"),
                        },
                    ),
                }
            )

            performance_folder = model_folder / "performance"
            performance_folder.mkdir(parents=True, exist_ok=True)
            pickle.dump(score, open(performance_folder / "scores.pkl", "wb"))

# %% [markdown]
# ## Compare

# %%
from params import params
from params_peakcounts import params_peakcounts

# %%
dataset_name = "hspc_gmp"

# %%
scores = []
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
                scores.append(score)
scores = pd.concat(scores).set_index(["gene", "regions", "param", "phase"])

# %%
param_summary = pd.DataFrame({**params, **params_peakcounts}).T
# param_summary["label"] = [print(param["label"]) for param_id, param in param_summary.iterrows()]
param_summary["label"] = [param["label"] if str(param["label"]) != "nan" else param_id for param_id, param in param_summary.iterrows()]
# param_summary

# %%
cors = xr.DataArray.from_series(scores["cor"]).reindex(param = param_summary.index)
# cors = cors.fillna(0)
assert not param_summary["label"].duplicated().any()
cors.coords["param"] = param_summary["label"].values
r2 = cors**2
ran = ~np.isnan(cors)
ranks = cors.rank("param")
pranks = cors.rank("param", pct = True)
ran.sel(phase = "test").mean("gene").to_pandas().T.style.bar()

# %%
cors.sel(gene = genes_oi2).mean("gene").sel(phase = "test").to_pandas().T.sort_values("10k10k").style.bar()

# %%
r2.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
(45-38)/38

# %%
ranks.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
r2.sel(gene = [transcriptome_original.gene_id("MPO")]).mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
# a = "peaks_main"
# b = "pred/radial_binary_1000-31frequencies_splitdistance"
a = "dt0.05/peaks_main"
b = "dt0.05/radial_binary_1000-31frequencies_splitdistance"
c = []
# c = ["dt0.05/radial_binary_1000-31frequencies", "dt0.05/exponential", "dt0.05/peaks_main"]
# b = "sine"
plotdata = (
    (r2)
        .sel(param = [a, b, *c])
        # .sel(gene = genes_oi2)
        .sel(phase = "test")
        .sel(regions = ["100k100k"])
        .stack({"region_gene":["regions", "gene"]})
        .to_pandas().T
)
gene_order = plotdata.mean(1).sort_values().index
# gene_order = plotdata.index
plotdata = plotdata.loc[gene_order]
fig, ax = plt.subplots(figsize = np.array(plotdata.shape)*np.array([0.1, 0.2]))
ax.matshow(plotdata.T, aspect = "auto")
ax.set_yticks(np.arange(len(plotdata.columns)))
ax.set_yticklabels(plotdata.columns)
ax.set_xticks(np.arange(len(plotdata.index)))
ax.set_xticklabels(plotdata.index.get_level_values("regions") + " " + transcriptome_original.symbol(plotdata.index.get_level_values("gene")), rotation = 90, fontsize = 6)
""

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id("TAFA2"))

# %%
fig, ax = plt.subplots()
plotdata["diff"] = plotdata[b] - plotdata[a]
plt.scatter(np.arange(len(plotdata)), plotdata["diff"])
ax.axhline(0, color = "black", linestyle = "--")

# loess
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], np.arange(len(plotdata)), frac = 0.5)
plt.plot(z[:, 0], z[:, 1], color = "red")

#
fig, ax = plt.subplots()
plt.scatter(transcriptome.var["dispersions"].loc[plotdata.index.get_level_values("gene")], plotdata[b] - plotdata[a])
ax.axhline(0, color = "black", linestyle = "--")

# %%
# plotdata = pranks.sel(phase = "test").sel(regions = "10k10k").to_pandas()
plotdata = pranks.sel(phase = "test").sel(regions = "100k100k").to_pandas()
plotdata = plotdata.loc[:, plotdata.mean(0).sort_values().index]
sns.boxplot(plotdata, orient = "h")

# %%
from scipy.stats import ttest_rel, wilcoxon
import itertools

pairs = []
param_summary_narrow = param_summary.loc[:, [col for col in param_summary.columns if not col in []]]
for (param1, row1), (param2, row2) in itertools.product(param_summary_narrow.iterrows(), param_summary_narrow.iterrows()):
    if param1 == param2:
        continue
    row1_oi = row1[[idx for idx in row1.index if not idx in ["distance_encoder", "label"]]]
    row2_oi = row2[[idx for idx in row2.index if not idx in ["distance_encoder", "label"]]]

    if row1_oi.equals(row2_oi):
        if pd.isnull(row1["distance_encoder"]) and row2["distance_encoder"] == "split":
            pairs.append((row1["label"], row2["label"]))

tests = []
for pair in pairs:
    for regions_name in ["10k10k", "100k100k"]:
        cors1 = cors.sel(param = pair[0]).sel(gene = genes_oi2).sel(phase = "test").sel(regions = regions_name)
        cors2 = cors.sel(param = pair[1]).sel(gene = genes_oi2).sel(phase = "test").sel(regions = regions_name)
        # test = ttest_rel(
        test = wilcoxon(
            cors2, cors1
        )
        tests.append({
            "param_1":pair[0],
            "param_2":pair[1],
            "regions_name":regions_name,
            "pval":test.pvalue,
            "statistic":test.statistic,
            "delta":(cors2 - cors1).mean().values,
        })
tests = pd.DataFrame(tests)
tests

# %%
pranks.sel(phase = "test").mean("gene").to_pandas().T.sort_values("100k100k").style.bar()

# %%
np.log(cors / cors.sel(param = "exponential")).sel(phase = "test").sel(regions = "10k10k").sel(gene = genes_oi2).mean("gene").to_pandas()

# %%
plotdata = r2.sel(gene = genes_oi2).sel(phase = "test").sel(regions = "100k100k").to_pandas()
plotdata = plotdata.loc[plotdata.mean(1).sort_values().index, plotdata.mean(0).sort_values().index]
fig, ax = plt.subplots(figsize = np.array(plotdata.shape)*np.array([0.1, 0.2]))
ax.matshow(plotdata.T, aspect = "auto")
ax.set_yticks(np.arange(len(plotdata.columns)))
ax.set_yticklabels(plotdata.columns)
ax.set_xticks(np.arange(len(plotdata.index)))
ax.set_xticklabels(transcriptome_original.symbol(plotdata.index), rotation = 90, fontsize = 6)
""

# %%
plotdata.mean(0).plot.bar()
