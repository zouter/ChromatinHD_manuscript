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

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
from chromatinhd_manuscript.designs_pred import (
    dataset_folds_peakcaller_predictor_combinations as design_peaks,
)

# design_peaks = design_peaks.loc[design_peaks["predictor"] != "xgboost"].copy()

from chromatinhd_manuscript.designs_pred import (
    dataset_folds_method_combinations as design_methods,
)

from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_folds_method_combinations as design_methods_traintest,
)

design_methods_traintest["dataset"] = design_methods_traintest["testdataset"]
design_methods_traintest["folds"] = "all"

from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_folds_peakcaller_predictor_combinations as design_peaks_traintest,
)

design_peaks_traintest["dataset"] = design_peaks_traintest["testdataset"]
design_peaks_traintest["folds"] = "all"

# %%
design_peaks["method"] = design_peaks["peakcaller"] + "/" + design_peaks["predictor"]
design_peaks_traintest["method"] = (
    design_peaks_traintest["peakcaller"] + "/" + design_peaks_traintest["predictor"]
)
# design_peaks = design_peaks.loc[design_peaks["predictor"] == "lasso"]

# %%
from chromatinhd_manuscript.designs_pred import dataset_folds_baselinemethods_combinations as design_baseline
from chromatinhd_manuscript.designs_pred import dataset_folds_simulation_combinations as design_simulated


# %%
design = pd.concat(
    [design_peaks, design_methods, design_methods_traintest, design_peaks_traintest, design_baseline, design_simulated]
)
design.index = np.arange(len(design))
design.index.name = "design_ix"

# %%
# design = design.loc[((design["folds"].isin(["random_5fold", "all"])))]
design = design.loc[((design["folds"].isin(["5x5"])))]
# design = design.query("layer in ['magic']").copy()
# design = design.query("layer in ['normalized']").copy()
# design = design.query("regions in ['10k10k']").copy()
# design = design.query("regions in ['100k100k']").copy()
design = design.query("regions in ['10k10k', '100k100k']").copy()
design = design.loc[design["peakcaller"] != "stack"]

# %%
design["traindataset"] = [
    x["dataset"] if pd.isnull(x["traindataset"]) else x["traindataset"]
    for _, x in design.iterrows()
]

# %%
assert not design[[col for col in design.columns if not col in ["params"]]].duplicated(keep = False).any(), "Duplicate designs"

# %%
scores = {}
design["found"] = False
for design_ix, design_row in design.iterrows():
    prediction = chd.flow.Flow(
        chd.get_output()
        / "pred"
        / design_row["dataset"]
        / design_row["regions"]
        / design_row["folds"]
        / design_row["layer"]
        / design_row["method"]
    )
    if (prediction.path / "scoring" / "performance" / "genescores.pkl").exists():
        print(prediction.path)
        # print(prediction.path)
        genescores = pd.read_pickle(
            prediction.path / "scoring" / "performance" / "genescores.pkl"
        )

        if isinstance(genescores, xr.Dataset):
            genescores = genescores.mean("model").to_dataframe()

        genescores["design_ix"] = design_ix
        scores[design_ix] = genescores.reset_index()
        design.loc[design_ix, "found"] = True
scores = pd.concat(scores, ignore_index=True)
scores = pd.merge(design, scores, on="design_ix")

scores = scores.reset_index().set_index(
    ["method", "dataset", "regions", "layer", "phase", "gene"]
)
assert not scores.index.duplicated().any(), "scores index is not unique"

dummy_method = "baseline_v42"
scores["cor_diff"] = (
    scores["cor"] - scores.xs(dummy_method, level="method")["cor"]
).reorder_levels(scores.index.names)

design["found"].mean()

# %%
metric_ids = ["cor"]

group_ids = ["method", "dataset", "regions", "layer", "phase"]

meanscores = scores.groupby(group_ids)[[*metric_ids, "design_ix"]].mean()
diffscores = meanscores - meanscores.xs(dummy_method, level="method")
diffscores.columns = diffscores.columns + "_diff"
relscores = np.log(meanscores / meanscores.xs(dummy_method, level="method"))
relscores.columns = relscores.columns + "_rel"

scores_all = meanscores.join(diffscores).join(relscores)

# %%
dataset = "pbmc10k"
# dataset = "hspc"

layer = "normalized"
layer = "magic"

regions = design["regions"].iloc[0]
# regions = "10k10k"
regions = "100k100k"

phase = "test"
# phase = "validation"

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset / "fragments" / "100k100k")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset / "folds" / "5x5")
fold = folds[1]

cors = scores.xs(dataset, level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs(regions, level = "regions")["cor"].unstack().T
cmses = scores.xs(dataset, level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs(regions, level = "regions")["cmse"].unstack().T
ccors = cors * pd.Series(transcriptome.layers["normalized"][:].std(0), index = transcriptome.var.index)[cors.index].values[:, None]

# %%
ms = ["baseline_v42", "v20", "macs2_leiden_0.1_merged/linear"]
plotdata = pd.DataFrame({
    # m:scores.loc[m].loc["pbmc10k"].loc["10k10k"].loc["normalized"].loc["test"]["cor"] for m in ms
    m:scores.loc[m].loc["pbmc10k"].loc["100k100k"].loc["magic"].loc["test"]["cor"] for m in ms
    # m:scores.loc[m].loc["pbmc10k"].loc["100k100k"].loc["normalized"].loc["test"]["cor"] for m in ms
})
m1, m2 = "baseline_v42", "v20"
m1, m2 = "macs2_leiden_0.1_merged/linear", "v20"

fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(plotdata[m1], plotdata[m2], color = "#333", s = 1)
ax.scatter(plotdata[m1].mean(), plotdata[m2].mean(), color = "red", s = 10)
ax.plot([0, 1], [0, 1], color = "black", linestyle = "--")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(m1)
ax.set_ylabel(m2)

# %%
plotdata["diff"] = (plotdata["v20"] - plotdata["macs2_leiden_0.1_merged/linear"])
plotdata.sort_values("diff", ascending = True).head(20).style.bar()

# %% [markdown]
# ## Load dataset and genes

# %%
dataset_name = "pbmc10k"

# %%
transcriptome_original = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset / "transcriptome")
fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset / "fragments" / "10k10k")

# %%
# genes_oi1 = plotdata.sort_values("diff").index[:40]
genes_oi1 = []
genes_oi2 = transcriptome_original.var.query("n_cells > 500").sort_values("dispersions_norm", ascending = False).index[:250]
genes_oi = set(genes_oi1) | set(transcriptome_original.gene_id(["CCL4", "IL1B", "CD79A", "QKI", "CD79B"])) | set(genes_oi2)
print(len(genes_oi))

# %%
gene_oi = transcriptome_original.gene_id("IL1B")

# %%
datasets = {}
for regions_name in ["100k100k", "10k10k"]:
    fragments_original = chd.data.Fragments(chd.get_output() / "datasets" / dataset / "fragments" / regions_name)
    for gene_oi in genes_oi:
        if not (chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / regions_name / "fragments").exists():
            fragments = fragments_original.filter_regions(
                fragments_original.regions.filter(
                    [gene_oi], path=chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / "regions" / regions_name
                ),
                path=chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / regions_name / "fragments",
            )
            fragments.create_regionxcell_indptr()

for gene_oi in genes_oi:
    if not (chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / "transcriptome").exists():
        transcriptome = transcriptome_original.filter_genes(
            [gene_oi], path=chd.get_output() / "datasets" / "mini" / "pbmc10k" / gene_oi / "transcriptome"
        )

# %% [markdown]
# ## Diffexp

# %%
import scanpy as sc

# %%
transcriptome_original.adata.obs["celltype"].value_counts()

# %%
groups = dict(
    T = ["CD4 naive T", "CD8 naive T", "CD4 memory T", "CD8 activated T", "MAIT"],
    T4 = ["CD4 naive T", "CD4 memory T"],
    lymphoid = ["CD4 naive T", "CD8 naive T", "CD4 memory T", "CD8 activated T", "MAIT", "naive B", "NK", "Plasma", "memory B"],
    B = ["naive B", "Plasma", "memory B"],
    myeloid = ["CD14+ Monocytes", "FCGR3A+ Monocytes", "cDCs"],
    mono = ["CD14+ Monocytes", "FCGR3A+ Monocytes"],
    DC = ["cDCs", "pDCs"],
    T8 = ["CD8 naive T", "CD8 activated T"],
    Tnaive = ["CD4 naive T", "CD8 naive T"],
    memory = ["CD4 memory T", "memory B"],
    all = transcriptome_original.adata.obs["celltype"].unique(),
)

diffexps = []
for group_name, celltypes in groups.items():
    adata = transcriptome_original.adata[transcriptome_original.adata.obs["celltype"].isin(celltypes)].copy()
    for celltype in celltypes:
        assert celltype in adata.obs["celltype"].unique(), celltype
        adata.obs["oi"] = (adata.obs["celltype"] == celltype).astype("category")
        sc.tl.rank_genes_groups(adata, groupby = "oi", use_raw=False, method = "t-test")
        diffexp = sc.get.rank_genes_groups_df(adata, group = None).set_index(["names"])
        diffexp.index.names = ["gene"]
        diffexp["significant"] = (diffexp["pvals_adj"] < 0.05) & (diffexp["logfoldchanges"] > 0.5)
        diffexps.append(pd.DataFrame({
            "pval":diffexp.groupby("gene")["pvals_adj"].min(),
            "lfc":diffexp.groupby("gene")["logfoldchanges"].max(),
            "score":diffexp.groupby("gene")["scores"].max(),
            "diffexp":diffexp.groupby("gene")["significant"].any(),
            "celltype":celltype,
            "group":group_name,
        }).reset_index())
diffexps = pd.concat(diffexps, ignore_index = True)

# %%
diffexps["significant"] = (diffexps["pval"] < 0.05) & (diffexps["lfc"] > 1.)
diffexps.groupby(["group", "gene"])["significant"].any().unstack().sum(1).sort_values()

# %%
diffexp_groups = diffexps.groupby(["group", "gene"])["significant"].any().unstack()

# %% [markdown]
# ## Model per gene

# %%
from params import params

# %%
import logging
chd.models.pred.trainer.trainer.logger.handlers = []
chd.models.pred.trainer.trainer.logger.propagate = False

# %%
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/*/*/0e669da3a497e5718efc4ec66f95f176
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/10k10k/*/e4ba9692843da9372d6a79c5570fa0c3
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name}/*/7bd6b1354af76bfa853313c822cb26cb
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/10k10k/ENSG00000158813/7bd6b1354af76bfa853313c822cb26cb

# %%
# encoding = chd.models.pred.model.better.TophatEncoding(50)
# encoding = chd.models.pred.model.better.RadialEncoding(50)
# encoding = chd.models.pred.model.better.LinearBinaryEncoding((1000, 500, 250))
encoding = chd.models.pred.model.better.RadialBinaryEncoding((5000, 1000, 500, 250), scale = 1)
# encoding = chd.models.pred.model.better.SineEncoding2(100)
# encoding = model.fragment_embedder.encoder.to("cpu")
import torch
x = torch.stack([
    # torch.arange(-1000, 1000),
    # torch.arange(-1000, 1000),
    torch.linspace(-100000, 100000, 500),
    torch.linspace(-100000, 100000, 500),
])

y = encoding(x.T)
sns.heatmap(y.detach().numpy())

# model.fragment_embedder.weight1.weight.data[:] = 0.
# model.embedding_to_expression.weight1.weight.data[:] = 0.

# %%
encoding = chd.models.pred.model.encoders.LinearDistanceEncoding()
encoding = chd.models.pred.model.encoders.SplitDistanceEncoding()
encoding = chd.models.pred.model.encoders.DirectDistanceEncoding()
import torch
x = torch.stack([
    torch.arange(-1000, 1000),
    torch.arange(-1000, 1000) + torch.linspace(0, 1000, 1000+1000),
])

y = encoding(x.T)
sns.heatmap(y.detach().numpy())

# %%
import pickle

# %%
for param_id, param in params.items():
    param = param.copy()
    cls = param.pop("cls")

    train_params = {}
    if "n_cells_step" in param:
        train_params["n_cells_step"] = param.pop("n_cells_step")
    if "lr" in param:
        train_params["lr"] = param.pop("lr")
    if "weight_decay" in param:
        train_params["weight_decay"] = param.pop("weight_decay")
    if "n_epochs" in param:
        train_params["n_epochs"] = param.pop("n_epochs")
    if "label" in param:
        param.pop("label")

    for regions_name in ["10k10k", "100k100k"]:
        for gene_oi in genes_oi:
            model_folder = chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name / gene_oi / param_id

            if (model_folder / "performance").exists():
                continue

            print(param_id)
            transcriptome = chd.data.Transcriptome(path=chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")
            fragments = chd.data.Fragments(path=chd.get_output() / "datasets" / "pbmc10k" / "fragments" / regions_name)

            model = cls(
                fragments = fragments,
                transcriptome=transcriptome,
                fold = fold,
                layer = layer,
                regions_oi = [gene_oi],
                **param,
            )
            # raise ValueError()
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
dataset_name = "pbmc10k"

for param_id, param in params_peakcounts.items():
    param = param.copy()

    for regions_name in ["10k10k", "100k100k"]:
        peakcounts = chd.data.peakcounts.PeakCounts(
            path=chd.get_output() / "datasets" / dataset_name / "peakcounts" / param["peakcaller"] / regions_name
        )

        for gene_oi in genes_oi:
            model_folder = chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name / gene_oi / param_id

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
param_summary = pd.DataFrame({**params, **params_peakcounts}).T
param_summary["label"] = [param["label"] if str(param["label"]) != "nan" else param_id for param_id, param in param_summary.iterrows()]

# %%
scores = []
for param_id, param in tqdm.tqdm({**params, **params_peakcounts}.items()):
    for gene_oi in genes_oi:
        for regions_name in ["10k10k", "100k100k"]:
            model_folder = chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name / gene_oi / param_id
            performance_folder = model_folder / "performance"
            if (performance_folder / "scores.pkl").exists():
                score = pickle.load(open(performance_folder / "scores.pkl", "rb"))["cor"]
                score = score.mean("model").to_dataframe().reset_index()
                score["param"] = param_id
                score["gene"] = gene_oi
                score["regions"] = regions_name

                if (model_folder / "trace.pkl").exists():
                    trace = pickle.load(open(model_folder / "trace.pkl", "rb"))
                    score["n_train_checkpoints"] = trace["n_train_checkpoints"]

                scores.append(score)
scores = pd.concat(scores).set_index(["gene", "regions", "param", "phase"])
scores = xr.Dataset.from_dataframe(scores).reindex(param = param_summary.index)
scores["r2"] = scores["cor"]**2

# %%
scores.coords["param"] = pd.Index(param_summary["label"].values, name = "param")

# %%
(scores["n_train_checkpoints"].isin([1000, 2000])).mean(["gene", "regions", "phase"]).to_pandas().sort_values()

# %%
cors = scores["cor"]
r2 = scores["r2"]
ran = ~np.isnan(cors)
ranks = cors.rank("param")
pranks = cors.rank("param", pct = True)

# %%
ran.sel(phase = "test").mean("gene").to_pandas().T.style.bar()

# %%
generanking = r2.sel(param = "peaks_main").sel(regions = "10k10k").sel(phase = "test").to_pandas()
genes_non = generanking.index[generanking > 0.1]
print(len(genes_non))
r2.sel(gene = genes_non).mean("gene").sel(phase = "validation").to_pandas().T.sort_values("10k10k").style.bar()

# %%
r2.sel(gene = genes_oi2).mean("gene").sel(phase = "test").to_pandas().T.sort_values("10k10k").style.bar()

# %%
r2.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()
# ranks.mean("gene").sel(phase = "test").to_pandas().T.sort_values("100k100k").style.bar()

# %%
a = "peaks_main"
a = "radial_binary_1000-31frequencies_splitdistance"
# a = "radial_binary_1000-31frequencies_splitdistance_wd1e-1"
# a = "radial_binary_1000-31frequencies_splitdistance_attn"
# a = "radial_binary_1000-31frequencies_splitdistance_attn_initeye"
# a = "radial_binary_1000-31frequencies_1-1layers_directdistance"
# a = "rolling_500_lasso"
# b = "radial_binary_4000-31frequencies_splitdistance_wd1e-1"
# b = "radial_binary_1000-31frequencies_lineardistance_wd1e-1"
b = "spline_binary_1000-31frequencies_splitdistance"
# b = "sine"
c = []
c = [
    # "radial_binary_1000-31frequencies_lineardistance_wd1e-1",
    # "radial_binary_4000-31frequencies_splitdistance_wd1e-1",
#     "radial_binary_1000-31frequencies_splitdistance_relu",
#     "radial_binary_1000-31frequencies_splitdistance_silu"
]
# c = ["radial_binary_1000-31frequencies_splitdistance", "radial_binary_1000-31frequencies_splitdistance_wd2e-1"]
# b = "sine"
plotdata = (
    # (cors)
    (r2)
        .sel(param = [a, b, *c])
        .sel(gene = genes_oi2)
        .sel(phase = "test")
        # .sel(regions = ["10k10k"])
        # .sel(regions = ["100k100k"])
        .sel(regions = ["100k100k", "10k10k"])
        .stack({"region_gene":["regions", "gene"]})
        .to_pandas().T
).dropna().fillna(0.)
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
plotdata.loc["10k10k"].mean(), plotdata.loc["100k100k"].mean()

# %%
scores.sel(param = "peaks_main", regions = "100k100k", gene = transcriptome_original.gene_id("CCL4"))["cor"].to_pandas()

# %%
scores.sel(param = "spline_binary_1000-31frequencies_splitdistance", regions = "100k100k", gene = transcriptome_original.gene_id("CCL4"))["cor"].to_pandas()

# %%
scores.sel(param = "spline_binary_1000-31frequencies_splitdistance", regions = "100k100k", gene = transcriptome_original.gene_id("IL1B"))["cor"].to_pandas()

# %%
import statsmodels.api as sm
import scipy.stats

fig = chd.grid.Figure(chd.grid.Wrap(padding_width = 0.7, ncol = 4))

plotdata["diff"] = plotdata[b] - plotdata[a]
plotdata["tick"] = (np.arange(len(plotdata)) % int(len(plotdata)/10)) == 0
plotdata["i"] = np.arange(len(plotdata))
plotdata["oi"] = plotdata.index.get_level_values("gene").isin(genes_oi1)
plotdata["oi"] = pd.Categorical(plotdata.index.get_level_values("regions"))
plotdata["dispersions"] = transcriptome.adata.var["dispersions"].loc[plotdata.index.get_level_values("gene")].values
plotdata["means"] = transcriptome.adata.var["means"].loc[plotdata.index.get_level_values("gene")].values
plotdata["dispersions_norm"] = transcriptome.adata.var["dispersions_norm"].loc[plotdata.index.get_level_values("gene")].values
plotdata["log10means"] = np.log10(plotdata["means"])
plotdata["log10dispersions_norm"] = np.log10(plotdata["dispersions_norm"])
n_fragments = pd.Series(fragments.counts.mean(0), index = fragments.var.index)
plotdata["log10n_fragments"] = np.log10(n_fragments.loc[plotdata.index.get_level_values("gene")].values)
n_fragments_std = pd.Series(np.log1p(fragments.counts).std(0), index = fragments.var.index)
plotdata["n_fragments_std"] = n_fragments_std.loc[plotdata.index.get_level_values("gene")].values

cmap = mpl.colormaps["Set1"]

# rank vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(np.arange(len(plotdata)), (plotdata["diff"]), c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a + " rank")
ax.set_ylabel("$\Delta$ r2")
ax.axhline(0, color = "black", linestyle = "--")
ax.set_xticks(plotdata["i"].loc[plotdata["tick"]])
ax.set_xticklabels(plotdata[a].loc[plotdata["tick"]].round(2), rotation = 0)

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], np.arange(len(plotdata)), frac = 0.5)
ax.plot(z[:, 0], z[:, 1], color = "green")

# a vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata[a], (plotdata["diff"]), c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a)
ax.set_ylabel("$\Delta$ r2")
ax.axhline(0, color = "black", linestyle = "--")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata[a], frac = 0.5)
ax.plot(z[:, 0], z[:, 1], color = "green")

lm = scipy.stats.linregress(plotdata[a], plotdata["diff"])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope

# vs
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata[a], plotdata[b], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a)
ax.set_ylabel(b)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

lowess = sm.nonparametric.lowess
z = lowess(plotdata[b], plotdata[a], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")
ax.axline((0, 0), slope = 1, color = "black", linestyle = "--")
ax.set_aspect("equal")

lm = scipy.stats.linregress(plotdata[a], plotdata[b])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope

cut = (1 - lm.intercept) / lm.slope
print(cut)
ax.annotate(f"cut {1-cut:.1%}", (1, 1), (1, 1.1), arrowprops = dict(arrowstyle = "->"), ha = "right")
ax.annotate(f"$r^2$={lm.rvalue**2:.1%}", (0.95, 0.95), ha = "right", va = "top")

# dispersions vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["dispersions"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("dispersion")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["dispersions"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10means"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10means")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10means"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions_norm vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10dispersions_norm"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10dispersions_norm")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10dispersions_norm"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10n_fragments"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10n_fragments")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10n_fragments"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments std
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["n_fragments_std"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("n_fragments_std")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["n_fragments_std"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

fig.plot()

# %%
lm = scipy.stats.linregress(plotdata[a], plotdata[b])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope, lm.intercept, lm.rvalue**2,(1 - lm.intercept) / lm.slope

# %%
# plotdata = pranks.sel(phase = "test").sel(regions = "100k100k").to_pandas()
plotdata = pranks.sel(phase = "test").sel(regions = "10k10k").to_pandas()
plotdata = plotdata.loc[:, plotdata.mean(0).sort_values().index]
plotdata = plotdata.dropna()
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
        if row1["distance_encoder"] == "split" and row2["distance_encoder"] == "linear":
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
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming you have a list of variable-length sequences
sequences = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4, 5, 6])]

# Step 1: Pad sequences to a common length
padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

# Step 2: Create mask
lengths = torch.tensor([len(seq) for seq in sequences])
mask = (padded_sequences != 0)

# Step 3: Apply Self-Attention
n_heads = 2  # Number of attention heads
dim_model = 64  # Dimension of model
multihead_attention = nn.MultiheadAttention(embed_dim=dim_model, num_heads=n_heads, batch_first=True)

embedding = padded_sequences[:, :, None].expand(-1, -1, dim_model).float()

output, _ = multihead_attention(embedding, embedding, embedding, key_padding_mask=~mask)

# You can now use 'output' for further processing.

# %%
embedding.shape

# %%
output.shape

# %%
import torch
from torch import nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        att_w = torch.nn.functional.softmax(self.W(batch_rep).squeeze(-1), -1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep
pooling = SelfAttentionPooling(dim_model)

pooling(output).shape

# %%
plotdata.mean(0).plot.bar()

# %% [markdown]
# ## Per cell type group diff exp

# %%
a = "peaks_main"
b = "radial_binary_1000-31frequencies_splitdistance"

# %%
celltypegroup_genes = {group_name:diffexp_groups.columns[diffexp_groups.loc[group_name]].intersection(genes_oi) for group_name in diffexp_groups.index}

# %%
import scipy.stats
groupscores = []
for group_name, genes in celltypegroup_genes.items():
    plotdata = r2.sel(gene = genes).sel(phase = "test").sel(regions = "100k100k").to_pandas()
    plotdata = plotdata.loc[plotdata.mean(1).sort_values().index, plotdata.mean(0).sort_values().index]

    lm = scipy.stats.linregress(plotdata[a], plotdata[b])

    groupscores.append({
        "group":group_name,
        "r2_mean":plotdata.mean(0)[a],
        "r2_diff":(plotdata.mean(0)[b] - plotdata.mean(0)[a]),
        "r2_diff_ratio":(plotdata.mean(0)[b] - plotdata.mean(0)[a]) / plotdata.mean(0)[b],
        "n_genes":len(genes),
        "slope":lm.slope,
    })
groupscores = pd.DataFrame(groupscores).set_index("group")

# %%
groupscores.sort_values("n_genes").style.bar()

# %%
lm = scipy.stats.linregress(groupscores["n_genes"], groupscores["r2_diff_ratio"])
lm

# %%
fig, ax = plt.subplots()
ax.scatter(groupscores["n_genes"], groupscores["r2_diff_ratio"])
ax.scatter(groupscores.loc[["all"]]["n_genes"], groupscores.loc[["all"]]["r2_diff_ratio"])

# %% [markdown]
# ## Interpret

# %%
param_id = [k for k, param in params.items() if param["label"] == "radial_binary_1000-31frequencies_splitdistance_wd1e-1"][0]
param = params[param_id]

param = param.copy()
cls = param.pop("cls")

train_params = {}
if "n_cells_step" in param:
    train_params["n_cells_step"] = param.pop("n_cells_step")
if "lr" in param:
    train_params["lr"] = param.pop("lr")
if "weight_decay" in param:
    train_params["weight_decay"] = param.pop("weight_decay")
if "n_epochs" in param:
    train_params["n_epochs"] = param.pop("n_epochs")

if "label" in param:
    param.pop("label")

# %%
gene_oi = transcriptome_original.gene_id("CCL4")
# gene_oi = transcriptome_original.gene_id("IL1B")

# %%
regions_name = "100k100k"
# regions_name = "10k10k"

# %%
train_params["lr"] = 1e-4
train_params["n_epochs"] = 10000

# %%
transcriptome = chd.data.Transcriptome(path=chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")
fragments = chd.data.Fragments(path=chd.get_output() / "datasets" / "pbmc10k" / "fragments" / regions_name)

model = cls(
    fragments = fragments,
    transcriptome=transcriptome,
    fold = fold,
    layer = layer,
    regions_oi = [gene_oi],
    **param,
)
model.train_model(**train_params, pbar = True)
# performance = chd.models.pred.interpret.Performance(path = model_folder / "performance")
# performance.score(fragments, transcriptome, [model], [fold], pbar = False)

# %%
model.trace.plot()

# %%
self = model

# %%
fragments.var["ix"] = np.arange(len(fragments.var))
region_ixs = fragments.var["ix"].loc[self.regions_oi].values
minibatcher = chd.loaders.minibatches.Minibatcher(
    np.concatenate([fold["cells_validation"], fold["cells_test"]]),
    region_ixs,
    n_regions_step=10,
    n_cells_step=10000,
    permute_cells=False,
    permute_regions=False,
)

loaders_validation = chd.loaders.LoaderPool(
    chd.loaders.TranscriptomeFragments,
    dict(
        transcriptome=transcriptome,
        fragments=fragments,
        cellxregion_batch_size=minibatcher.cellxregion_batch_size,
        layer=self.layer,
    ),
    n_workers=5,
)

# %%
loaders_validation.initialize(minibatcher)
data = next(iter(loaders_validation))

# %%
model.fragment_embedder.encoder.requires_grad = True

# %%
loss = model.forward_loss(data)
loss.backward()

# %%
oi = (model.fragment_embedder.encoder.scales == model.fragment_embedder.encoder.scales[-1])
locs = model.fragment_embedder.encoder.locs.detach().numpy()
model.fragment_embedder.encoder.locs.shape

# %%
grad = model.fragment_embedder.encoder.embedding.grad
val = model.fragment_embedder.encoder.embedding.detach()
attribution = grad * val
# attribution = attribution.reshape(-1, model.fragment_embedder.encoder.embedding.grad.shape[1] // 2, 2).mean(-1).mean(0)
attribution = attribution[:, attribution.shape[1]//2:].mean(0)

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes = (20, 50, 100, 200, 500))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(path = chd.get_output() / "test" / "interpret", reset = True)
regionmultiwindow.score(fragments, transcriptome, [model], [fold], censorer, regions = [gene_oi])

# %%
regionmultiwindow.interpolate()

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05))
width = 10

# window = None
window = (2500, 5000)

region = fragments.regions.coordinates.loc[gene_oi]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window)
fig.main.add_under(panel_genes)

panel_pileup = chd.models.pred.plot.Pileup.from_regionmultiwindow(
    regionmultiwindow, gene_oi, width=width, window = window
)
fig.main.add_under(panel_pileup)

panel_predictivity = chd.models.pred.plot.Predictivity.from_regionmultiwindow(
    regionmultiwindow, gene_oi, width=width, window = window
)
fig.main.add_under(panel_predictivity)

fig.plot()

# %%
plotdata = regionmultiwindow.get_plotdata(gene_oi)

fig, ax = plt.subplots()
(-plotdata["deltacor"]).plot(zorder = 10)
ax.set_xlim(2500, 5000)
# ax.set_xlim(-1000, 1000)

ax2 = ax.twinx()
ax2.plot(plotdata.index, plotdata["lost"], color = "grey", zorder = -1)

# plotdata = pd.DataFrame({"position":locs, "scale":model.fragment_embedder.encoder.scales.detach().numpy(), "attribution":attribution})
# plotdata = plotdata.loc[plotdata["scale"] == 20]

# ax2 = ax.twinx()
# ax2.plot(plotdata["position"], plotdata["attribution"], color = "orange")
# ax2.axhline(0, color = "orange", linestyle = "--")

# %% [markdown]
# ## R2

# %% [markdown]
# Discussing out of sample R2: https://stats.stackexchange.com/questions/228540/how-to-calculate-out-of-sample-r-squared
