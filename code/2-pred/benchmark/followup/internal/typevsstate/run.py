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
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

import pickle

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
dataset_name = "pbmc10k/subsets/mono_t_ab"

# %%
transcriptome_original = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "10k10k")

# %% [markdown]
# ## Diffexp

# %%
dataset_names = ["pbmc10k/subsets/mono_t_a", "pbmc10k/subsets/mono_t_b", "pbmc10k/subsets/mono_t_ab"]
datasets = {}
for dataset_name in dataset_names:
    dataset_folder = chd.get_output() / "datasets" / dataset_name
    transcriptome = chd.data.Transcriptome(path=dataset_folder / "transcriptome")
    datasets[dataset_name] = {"transcriptome":transcriptome}

    for regions_name in ["10k10k", "100k100k"]:
        fragments = chd.data.Fragments(path=dataset_folder / "fragments" / regions_name)
        fragments.create_regionxcell_indptr(overwrite = True)

# %%
adata = datasets["pbmc10k/subsets/mono_t_a"]["transcriptome"].adata

# %%
import scanpy as sc
sc.pl.umap(adata, color = ["ENSG00000148737"])

# %%
# ENSG00000145819
# ENSG00000148737 

# %%
import scanpy as sc

# %%
diffexps = []
for dataset_name in dataset_names:
    transcriptome = datasets[dataset_name]["transcriptome"]
    sc.tl.rank_genes_groups(transcriptome.adata, groupby = "celltype", use_raw=False, method = "t-test")
    diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, group = None).set_index(["group", "names"])
    diffexp.index.names = ["celltype", "gene"]
    diffexp["significant"] = (diffexp["pvals_adj"] < 0.05) & (diffexp["logfoldchanges"] > 0.5)
    diffexps.append(pd.DataFrame({
        "lfc":diffexp.groupby("gene")["logfoldchanges"].max(),
        "score":diffexp.groupby("gene")["scores"].max(),
        "diffexp":diffexp.groupby("gene")["significant"].any(),
        "dataset":dataset_name
    }).reset_index())
diffexps = pd.concat(diffexps).set_index(["gene", "dataset"])

# %%
diffexp_significant = diffexps["diffexp"].unstack()

# %% [markdown]
# ## Select genes

# %%
genes_oi_a = diffexps.xs(dataset_names[0], level = "dataset").query("diffexp").sort_values("score", ascending = False).index[:20].tolist()
genes_oi_b = diffexps.xs(dataset_names[1], level = "dataset").query("diffexp").sort_values("score", ascending = False).index[:20].tolist()
genes_oi = list(set(genes_oi_a + genes_oi_b))
print(len(genes_oi))

# %%
genes_a = pd.Series(genes_oi)[diffexp_significant.loc[genes_oi][dataset_names[0]].values]
genes_nona = pd.Series(genes_oi)[~diffexp_significant.loc[genes_oi][dataset_names[0]].values]
genes_b = pd.Series(genes_oi)[diffexp_significant.loc[genes_oi][dataset_names[1]].values]

# %%
adata = datasets["pbmc10k/subsets/mono_t_b"]["transcriptome"].adata
sc.pl.umap(adata, color = genes_b.values[:5], use_raw = False)

# %% [markdown]
# ## Model per gene

# %%
from params import params

# %%
import logging
chd.models.pred.trainer.trainer.logger.handlers = []
chd.models.pred.trainer.trainer.logger.propagate = False

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

    for dataset_name in dataset_names:
        for regions_name in ["10k10k", "100k100k"]:
            for gene_oi in genes_oi:
                model_folder = chd.get_output() / "models" / "mini" / dataset_name / regions_name / gene_oi / param_id

                if (model_folder / "performance").exists():
                    continue

                dataset_folder = chd.get_output() / "datasets" / dataset_name

                print(param_id, gene_oi, dataset_name, regions_name)
                transcriptome = chd.data.Transcriptome(path=dataset_folder / "transcriptome")
                fragments = chd.data.Fragments(path=dataset_folder / "fragments" / regions_name)
                folds = chd.data.folds.Folds(path = dataset_folder / "folds" / "5x5")
                fold = folds[0]

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
# params_peakcounts = {}

# %%
def r2(y, y_predicted, y_train):
    return 1 - ((y_predicted - y) ** 2).sum() / ((y - y_train.mean()) ** 2).sum()


# %%
import chromatinhd.data.peakcounts
dataset_name = "pbmc10k"

for param_id, param in params_peakcounts.items():
    param = param.copy()

    peakcaller = param.pop("peakcaller")
    label = param.pop("label")
    
    for regions_name in ["10k10k", "100k100k"]:
        peakcounts = chd.data.peakcounts.PeakCounts(
            path=chd.get_output() / "datasets" / "pbmc10k" / "peakcounts" / peakcaller / regions_name
        )

        for dataset_name in dataset_names:
            dataset_folder = chd.get_output() / "datasets" / dataset_name
            transcriptome = chd.data.Transcriptome(path=dataset_folder / "transcriptome")
            for gene_oi in genes_oi:
                model_folder = chd.get_output() / "models" / "mini" / dataset_name / regions_name / gene_oi / param_id

                if (model_folder / "performance" / "scores.pkl").exists():
                    continue

                import chromatinhd.models.pred.model.peakcounts2
                chromatinhd.models.pred.model.peakcounts2.Prediction(
                    path = model_folder
                ).score(peakcounts, transcriptome, gene_oi, fold, **param, layer = "magic", subset_cells = True)

# %% [markdown]
# ## Compare

# %%
from params import params
from params_peakcounts import params_peakcounts
import pickle

# %%
param_summary = pd.DataFrame({**params, **params_peakcounts}).T
param_summary["label"] = [param["label"] if str(param["label"]) != "nan" else param_id for param_id, param in param_summary.iterrows()]

# %%
scores = []
for param_id, param in {**params, **params_peakcounts}.items():
    for dataset_name in dataset_names:
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
scores = pd.concat(scores).set_index(["gene", "dataset", "regions", "param", "phase"])
scores = xr.Dataset.from_dataframe(scores).sel(param = param_summary.index)
scores["r2"] = scores["cor"]**2
scores.coords["param"] = pd.Index(param_summary["label"], name = "param")

# %%
scores.sel(gene = genes_b.values, phase = "test", regions = "100k100k", dataset = dataset_names[1]).to_dataframe()["r2"].unstack().mean()

# %%
scores.sel(gene = genes_b.values, phase = "test", regions = "100k100k", dataset = dataset_names[0]).to_dataframe()["r2"].unstack().mean()

# %%
adata = datasets["pbmc10k/subsets/mono_t_a"]["transcriptome"].adata
sc.pl.umap(adata, color = genes_a.values[:5], use_raw = False)
genes_a

# %%
sns.heatmap(
    scores.sel(gene = genes_b.values, phase = "test", regions = "100k100k", dataset = dataset_names[1]).to_dataframe()["r2"].unstack()
)

# %%
sns.heatmap(
    scores.sel(gene = genes_a.values, phase = "test", regions = "100k100k", dataset = dataset_names[0]).to_dataframe()["r2"].unstack()
)

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
generanking = r2.sel(param = "exponential").sel(regions = "10k10k").sel(phase = "test").to_pandas()
genes_non = generanking.index[generanking > 0.1]
print(len(genes_non))
r2.sel(gene = genes_non).mean("gene").sel(phase = "validation").to_pandas().T.sort_values("10k10k").style.bar()

# %%
r2.sel(gene = genes_oi2).mean("gene").sel(phase = "test").to_pandas().T.sort_values("10k10k").style.bar()

# %%
r2.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()
# ranks.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
ranks.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
r2.sel(gene = [transcriptome_original.gene_id("QKI")]).mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
a = "peaks_main"
# a = "rolling_500_lasso"
b = "radial_binary_1000-31frequencies_splitdistance"
c = []
# c = ["rolling_500_lasso", "radial_splitdistance"]
# b = "sine"
plotdata = (
    (r2)
        .sel(param = [a, b, *c])
        # .sel(gene = genes_oi2)
        .sel(phase = "test")
        # .sel(regions = ["10k10k"])
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
# fig, ax = plt.subplots()
# ax.axhline(0)
# plt.boxplot(plotdata[b] - plotdata[a])

# fig, ax = plt.subplots()
# sns.stripplot(plotdata[b]/plotdata[a], ax = ax);
# ax.axhline(0)
# ax.set_yscale("log")
# ax.set_ylim(1/2, 2)
# ax.axhline(0)

# %%
fig, (ax0, ax1, ax3) = plt.subplots(1, 3, figsize = (6, 2))

plotdata["diff"] = plotdata[b] - plotdata[a]
plotdata["reldiff"] = np.clip(plotdata["diff"], 0, 1)/np.clip(plotdata[b], 0.01, 1)
plotdata["tick"] = (np.arange(len(plotdata)) % int(len(plotdata)/10)) == 0
plotdata["i"] = np.arange(len(plotdata))
plotdata["oi"] = plotdata.index.get_level_values("gene").isin(genes_oi1)

cmap = mpl.colormaps["Set1"]

ax0.scatter(np.arange(len(plotdata)), (plotdata["diff"]), c = cmap(plotdata["oi"].astype(int)), s = 3)
ax0.axhline(0, color = "black", linestyle = "--")
ax0.set_xticks(plotdata["i"].loc[plotdata["tick"]])
ax0.set_xticklabels(plotdata[a].loc[plotdata["tick"]].round(2), rotation = 0)

# loess
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], np.arange(len(plotdata)), frac = 0.5)
ax0.plot(z[:, 0], z[:, 1], color = "red")

# actual mean
ax1.scatter(plotdata[a], (plotdata["diff"]), c = cmap(plotdata["oi"]), s = 3)
ax1.axhline(0, color = "black", linestyle = "--")
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata[a], frac = 0.5)
ax1.plot(z[:, 0], z[:, 1], color = "red")


# vs
ax3.scatter(plotdata[a], plotdata[b], c = cmap(plotdata["oi"]), s = 3)
ax3.axhline(0, color = "black", linestyle = "--")

# loess
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
z = lowess(plotdata[b], plotdata[a], frac = 2/3)
ax3.plot(z[:, 0], z[:, 1], color = "red")
ax3.axline((0, 0), slope = 1, color = "black", linestyle = "--")

# lm
import scipy.stats
lm = scipy.stats.linregress(plotdata[a], plotdata[b])
lm.slope

# %%
plotdata = pranks.sel(phase = "test").sel(regions = "100k100k").to_pandas()
# plotdata = pranks.sel(phase = "test").sel(regions = "10k10k").to_pandas()
plotdata = plotdata.loc[:, plotdata.mean(0).sort_values().index]
sns.boxplot(plotdata, orient = "h")

# %%
# Wilcoxon signed-rank test
from scipy.stats import wilcoxon
wilcoxon(
    cors.sel(gene = genes_oi2).sel(phase = "test").sel(regions = "10k10k").sel(param = "radial_binary"),
    cors.sel(gene = genes_oi2).sel(phase = "test").sel(regions = "10k10k").sel(param = "radial_binary_200cellsteps"),
)

# %%
import itertools

from scipy.stats import wilcoxon, ttest_rel

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
