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
#     display_name: Python 3 (ipykernel)
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
# design = design.query("dataset != 'alzheimer'").copy()
# design = design.query("dataset == 'pbmc10k'").copy()
# design = design.query("dataset == 'pbmc3k-pbmc10k'").copy()
# design = design.query("dataset == 'pbmc10k_gran-pbmc10k'").copy()
# design = design.query("dataset == 'pbmc10k_gran-pbmc10k'").copy()

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

cors = scores.xs(dataset, level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs(regions, level = "regions")["cor"].unstack().T
cmses = scores.xs(dataset, level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs(regions, level = "regions")["cmse"].unstack().T
ccors = cors * pd.Series(transcriptome.layers["normalized"][:].std(0), index = transcriptome.var.index)[cors.index].values[:, None]

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(cors["baseline_v42"], cors["v20"], color = "#333", s = 1)
ax.plot([0, 1], [0, 1], color = "black", linestyle = "--")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# %%
cors.mean().to_frame().style.bar()

# %%
cors.loc[transcriptome.gene_id("CD69")].to_frame().style.bar()

# %%
cors_validation = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("validation", level="phase").xs(regions, level = "regions")["cor"].unstack().T
cors_test = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("test", level="phase").xs(regions, level = "regions")["cor"].unstack().T
fig, ax = plt.subplots()
reference = "macs2_leiden_0.1_merged/linear"
reference = "baseline_v42"
ax.scatter(  
    (cors_validation["v20"] - cors_validation[reference]),
    (cors_test["v20"] - cors_test[reference]),
)
(cors_test["v20"] > cors_test[reference]).mean()

# %%
relative_scores = (cors[["macs2_leiden_0.1_merged/linear", "baseline_v42"]] - cors["v20"].values[:, None]).sort_values("macs2_leiden_0.1_merged/linear")
relative_scores["n_fragments"] = pd.Series(np.argsort(np.argsort(fragments.counts.sum(0)))/len(fragments.var), fragments.var.index)
relative_scores.tail(20).style.bar()

# %%
sns.ecdfplot(scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").loc["v20"].loc[regions].loc["validation"]["cor"] - scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").loc["v20"].loc[regions].loc["test"]["cor"])


# %%
def add_v23(cors, cors_validation):
    idx = np.argmax(
        np.stack(
            [
                cors_validation["v20"].values,
                cors_validation["v22"].values,
            ]
        ),
        0,
    )
    v23_cors = pd.Series(
        np.stack(
            [
                cors["v20"].values,
                cors["v22"].values,
            ]
        )[idx, np.arange(len(idx))],
        index=cors.index,
    )
    cors["v23"] = v23_cors.reindex(cors.index, level="gene").values
cors_validation = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("validation", level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
cors_test = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("test", level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
add_v23(cors_test, cors_validation)

# %%
cors_test.mean().to_frame("cor").sort_values("cor").style.bar()

# %%
methods = chdm.methods.prediction_methods
missing_methods = list(set(scores.index.get_level_values("method").unique()) - set(methods.index))
methods = methods.reindex(list(methods.index) + missing_methods)
methods["color"] = methods["color"].fillna("grey")


# %%
def judge(genes_oi):
    fig, (ax, ax1) = plt.subplots(1, 2, figsize = (6, 2))
    plotdata = cors.loc[genes_oi]
    for key, cor in plotdata.items():
        color = methods.loc[key, "color"]
        sns.ecdfplot(cor, label = key, color = color, ax = ax)
    ax.legend(loc='center left', bbox_to_anchor=(3, 0.5))

    import IPython.display
    IPython.display.display(pd.DataFrame({
        "actual":cors.loc[genes_oi].mean(),
        "relative":(1-(cors.loc[genes_oi].mean()/cors.loc[genes_oi, "v23"].values[:, None].mean()).sort_values()),
        "absolute":((cors.loc[genes_oi, "v23"].values[:, None].mean() - cors.loc[genes_oi].mean())),
    }).sort_values("relative").style.bar(vmin = 0))

    plotdata = cors.loc[genes_oi].mean(0)
    plotdata = plotdata.loc[~plotdata.index.str.startswith("simulated")]
    plotdata.plot.bar(ax = ax1)

    print(cors.loc[genes_oi]["baseline_v42"].mean() - cors.loc[genes_oi]["v23"].mean())

def compare_with_baseline(genes_oi, extra_methods = None, title = None):
    plotdata = cors.loc[genes_oi].mean(0)
    if extra_methods is None:
        extra_methods = []
    plotdata = plotdata.loc[["v23", 
    # "macs2_leiden_0.1_merged/linear", 
    "baseline_v42", "simulated_both_10", "simulated_both_50", *extra_methods]]
    fig, ax = plt.subplots(figsize = (2.5, 1.2))
    plotdata.plot.barh(ax = ax, width = 0.8, color = methods.loc[plotdata.index, "color"])
    ax.set_xlabel(f"average correlation (n = {len(genes_oi)})")
    ax.set_yticklabels(["ChromatinHD-pred", "Baseline", "Simulation 10%", "Simulation 50%", *methods.loc[extra_methods, "label"]])
    ax.set_ylabel("")
    ax.set_title(title)


def compare(genes_oi, extra_methods = None, title = None):
    plotdata = cors.loc[genes_oi].mean(0)
    if extra_methods is None:
        extra_methods = []
    plotdata = plotdata.loc[["v23", 
    # "macs2_leiden_0.1_merged/linear", 
    "baseline_v42", *extra_methods]]
    fig, ax = plt.subplots(figsize = (2.5, 1.2))
    plotdata.plot.barh(ax = ax, width = 0.8, color = methods.loc[plotdata.index, "color"])
    ax.set_xlabel(f"average correlation (n = {len(genes_oi)})")
    ax.set_yticklabels(["ChromatinHD-pred", "ArchR 42", *methods.loc[extra_methods, "label"]])
    ax.set_ylabel("")
    ax.set_title(title)


def compare_with_baseline_peaks(genes_oi, title = None):
    plotdata = cors.loc[genes_oi].mean(0)
    plotdata = plotdata.loc[["v23", 
    "macs2_leiden_0.1_merged/linear", 
    "baseline_v42", "simulated_both_10", "simulated_both_50"]]
    fig, ax = plt.subplots(figsize = (2.5, 1.2))
    plotdata.plot.barh(ax = ax, width = 0.8, color = methods.loc[plotdata.index, "color"])
    ax.set_xlabel(f"average correlation (n = {len(genes_oi)})")
    ax.set_yticklabels(["ChromatinHD-pred", "MACS2", "Baseline", "Simulation 10%", "Simulation 50%"])
    ax.set_ylabel("")
    ax.set_title(title)


def compare_simulations(title = None):
    plotdata = cors.mean(0)
    plotdata = plotdata.loc[[
    "simulated_both_10", "simulated_both_50", "simulated_both", "v23"]]
    fig, ax = plt.subplots(figsize = (2.5, 1.2))
    plotdata.plot.barh(ax = ax, width = 0.8, color = methods.loc[plotdata.index, "color"])
    ax.set_xlabel(f"average correlation")
    ax.set_yticklabels(["Simulation 10%", "Simulation 50%", "Simulation 100%", "ChromatinHD"])
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xlim(0, 1)


def compare_baselines(title = None):
    plotdata = cors.mean(0)
    plotdata = plotdata.loc[[
    "simulated_both_10", "simulated_both_50", "baseline_v42", "baseline_v21", "v23"]]
    fig, ax = plt.subplots(figsize = (2.5, 1.2))
    plotdata.plot.barh(ax = ax, width = 0.8, color = methods.loc[plotdata.index, "color"])
    ax.set_xlabel(f"average correlation")
    ax.set_yticklabels(["Simulation 10%", "Simulation 50%", "ArchR 42", "ArchR 21", "ChromatinHD"])
    ax.set_ylabel("")
    ax.set_title(title)


# %%
cors = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("test", level="phase").xs("10k10k", level = "regions")["cor"].unstack().T
cors["v23"] = cors["v20"]

genes_oi = transcriptome.var.sort_values("dispersions_norm").index[:1000]
compare(genes_oi, title = "1000 least differential genes", extra_methods = ["macs2_leiden_0.1_merged/linear"])

genes_oi = transcriptome.var.sort_values("dispersions_norm").index[-1000:]
compare(genes_oi, title = "1000 most differential genes", extra_methods = ["macs2_leiden_0.1_merged/linear"])

cors = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("test", level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
cors["v23"] = cors["v20"]

genes_oi = transcriptome.var.sort_values("dispersions_norm").index[:1000]
compare(genes_oi, title = "1000 least differential genes", extra_methods = ["macs2_leiden_0.1_merged/linear"])

genes_oi = transcriptome.var.sort_values("dispersions_norm").index[-1000:]
compare(genes_oi, title = "1000 most differential genes", extra_methods = ["macs2_leiden_0.1_merged/linear"])

# %%
cors = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
cors["v23"] = cors["v20"]
compare_baselines()

# %%
compare_simulations()

# %%
genes_oi = cors.index

cors = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs("10k10k", level = "regions")["cor"].unstack().T
cors["v23"] = cors["v20"]
compare_with_baseline(genes_oi, title = "10kb-TSS-10kb")

cors = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs(phase, level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
cors["v23"] = cors["v20"]
compare_with_baseline(genes_oi, title = "100kb-TSS-100kb")

# %%
cors_validation = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("validation", level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
cors = scores.xs("pbmc10k", level="dataset").xs(layer, level = "layer").xs("test", level="phase").xs("100k100k", level = "regions")["cor"].unstack().T
add_v23(cors, cors_validation)

# %%
# all genes
genes_oi = cors.index
judge(genes_oi)

# %%
import scipy.stats
genes_oi = transcriptome.var.sort_values("mean", ascending = False).index[:2000]
# genes_oi = pd.Series(True, index = transcriptome.var.index)
cors_ranked = cors.loc[genes_oi, [
    "rolling_500/lasso",
    "baseline_v42",
    # "baseline_v21",
    "genrich/linear",
    "macs2_leiden_0.1_merged/linear",
    # "cellranger/linear",
    "v23",
    # "counter",
]]
cors_ranked = pd.DataFrame(scipy.stats.rankdata(cors_ranked, axis = 1), index = cors_ranked.index, columns = cors_ranked.columns)
cors_ranked = cors_ranked.loc[:, cors_ranked.mean(0).sort_values().index]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height = 0))
for method in cors_ranked.columns:
    panel, ax = fig.main.add_under(polyptich.grid.Panel((1, 0.2)))
    ax.hist(cors_ranked[method], bins = np.arange(cors_ranked.max().max()+2), label = method, alpha = 0.5, color = methods.loc[method, "color"], density = True)
    ax.set_ylabel(methods.loc[method, "label"], rotation = 0, ha = "right", va = "center")
    ax.set_yticks([])
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, cors_ranked.max().max()+1)
fig.plot()

# %%
plotdata = (cors_ranked-1).apply(np.bincount) / len(cors_ranked)
fig, ax = plt.subplots(figsize = (2, 2))
bottom = np.zeros(cors_ranked.shape[1])
palette = sns.color_palette("Greens", n_colors = len(cors_ranked.columns))
for perc, plotdata_perc in plotdata.groupby(np.arange(len(plotdata))):
    ax.barh(
        np.arange(len(plotdata.columns)),
        plotdata_perc.values[0],
        left=bottom,
        color=palette[int(perc)],
        lw = 0,
        height = 0.9,
        label = f"{perc}",
    )
    # plotdata_perc.plot.bar(ax = ax, color = methods.loc[plotdata.columns, "color"], alpha = 0.5, label = f"{perc:.0%}")
    bottom = bottom + plotdata_perc.values[0]
ax.set_yticks(np.arange(len(plotdata.columns)))
ax.set_yticklabels(methods.loc[plotdata.columns, "label"])
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, len(plotdata.columns)-0.5)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.legend(title = "rank", bbox_to_anchor = (1, 1), loc = "upper left")
ax.set_xlabel("Genes")

# %%
genes_oi = (cors_ranked["macs2_leiden_0.1_merged/linear"] == 1)
genes_oi = genes_oi.index[genes_oi]
compare_with_baseline(genes_oi, extra_methods = ["macs2_leiden_0.1_merged/linear", "rolling_500/lasso"], title = f"Genes where MACS2 per cell type merged is the worst (n={len(genes_oi)})")

# %%
genes_oi = (cors_ranked["rolling_500/lasso"] == 1)
genes_oi = genes_oi.index[genes_oi]
compare_with_baseline(genes_oi, extra_methods = ["macs2_leiden_0.1_merged/linear", "rolling_500/lasso"], title = f"Genes where Window 500bp is the worst (n={len(genes_oi)})")

# %%
genes_oi = (cors_ranked["baseline_v42"] == 1)
genes_oi = genes_oi.index[genes_oi]
compare_with_baseline(genes_oi, extra_methods = ["macs2_leiden_0.1_merged/linear", "rolling_500/lasso"], title = f"Genes where baseline is the worst (n={len(genes_oi)})")

# %%
genes_oi = (cors_ranked["v23"] == 1)
genes_oi = genes_oi.index[genes_oi]
compare_with_baseline(genes_oi, extra_methods = ["macs2_leiden_0.1_merged/linear", "rolling_500/lasso"], title = f"Genes where ChromatinHD is the worst (n={len(genes_oi)})")

# %%
# genes_oi = ((cors["macs2_leiden_0.1_merged/linear"] - cors["v23"]) < 0) & (cors.index.isin(transcriptome.var.sort_values("mean", ascending = False).index[:500]))
genes_oi = pd.Series(cors.index.isin(transcriptome.var.sort_values("mean", ascending = False).index[:2000]), cors.index)
genes_oi = genes_oi[genes_oi].index
plotdata = pd.DataFrame({
    "baseline_v42":cors["baseline_v42"],
    "macs2_leiden_0.1_merged/linear":cors["macs2_leiden_0.1_merged/linear"],
    "v23":cors["v23"],
    "diff":(cors["v23"]-cors["macs2_leiden_0.1_merged/linear"]),
}).loc[genes_oi]
fig, ax = plt.subplots()
ax.scatter(plotdata["v23"], plotdata["diff"], cmap = "coolwarm", alpha = 0.5, s = 2)
ax.axhline(0, color = "black", linestyle = "--")
ax.set_xlabel("cor ChromatinHD")
ax.set_ylabel("cor ChromatinHD - MACS2")
# ax.plot([0, 1], [0, 1], color = "black", linestyle = "--")

# %%
# most differential genes
genes_oi = transcriptome.var.sort_values("dispersions_norm").index[-1000:]
judge(genes_oi)
compare_with_baseline(genes_oi, title = "1000 most differential genes")

# %%
# least differential genes
genes_oi = transcriptome.var.sort_values("dispersions_norm").index[:1000]
judge(genes_oi)
compare_with_baseline(genes_oi, title = "1000 least differential genes")

# %%
# most expressed genes
genes_oi = transcriptome.var["means"].sort_values().tail(1000).index
judge(genes_oi)
compare_with_baseline(genes_oi, title = "1000 most expressed genes")

# %%
genes_oi = (cors["baseline_v42"] < 0.05)
print(genes_oi.mean())
compare_with_baseline_peaks(genes_oi, title = "genes for which V42 has cor < 0.05")

# %%
genes_oi = ((cors["macs2_leiden_0.1_merged/linear"] - cors["baseline_v42"]) < 0)
compare_with_baseline_peaks(genes_oi, title = "genes for which MACS2 - baseline < 0")

# %%
genes_oi = ((cors["v20"] - cors["baseline_v42"]) < 0)
print(genes_oi.sum())
compare_with_baseline_peaks(genes_oi, title = "genes for which MACS2 - baseline < 0")

# %%
fig, ax = plt.subplots()

reference_id = "simulated_both"
# reference_id = "simulated_both_10"
# baseline_id = "macs2_leiden_0.1_merged/linear"
baseline_id = "baseline_v42"
observed_id = "v23"

genes_oi =  transcriptome.var["means"].sort_values().tail(1000).index

plotdata = cors.loc[genes_oi]

segs = np.stack([plotdata[[reference_id, reference_id]], plotdata[[baseline_id, observed_id]]], -1)
line_segments = mpl.collections.LineCollection(segs, linestyle='solid', alpha = 0.1, color = "black")
ax.add_collection(line_segments)
ax.scatter(plotdata[reference_id], plotdata[observed_id], c = "k", s = 3)
ax.scatter(plotdata[reference_id], plotdata[baseline_id], c = "k", s = 10, marker = "_", alpha = 0.1)

xs_perfect = np.linspace(0, 1, 100)
ys_perfect = xs_perfect
ax.plot(xs_perfect, ys_perfect, color = "#333", linestyle = "dashed")
# ax.set_xlim(0.2, 0.6)
ax.set_ylim(-0., 0.8)

# %%
fig, ax = plt.subplots()
# x, y = cors.loc[genes_oi, "v20"], cors.loc[genes_oi, "macs2_leiden_0.1_merged/linear"]
x, y = cors.loc[genes_oi, "simulated_both"], cors.loc[genes_oi, "v20"]
plt.scatter(x, y, s = 1)
plt.scatter(x[x>y], y[x>y], color = "red", s = 1)
print((x>y).mean())
ax.plot([0, 1], [0, 1])
fig, ax = plt.subplots()

# %%
genes_oi = cors.index[:1000]

fig, (ax, ax1) = plt.subplots(1, 2, figsize = (6, 2))
plotdata = cors.loc[genes_oi]
for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    sns.ecdfplot(cor, label = key, color = color, ax = ax)
ax.legend(loc='center left', bbox_to_anchor=(3, 0.5))

pd.DataFrame({
    "actual":cors.mean(),
    "relative":(1-(cors.loc[genes_oi].mean()/cors.loc[genes_oi, "v23"].values[:, None].mean()).sort_values()),
    "absolute":((cors.loc[genes_oi, "v23"].values[:, None].mean() - cors.loc[genes_oi].mean())),
}).sort_values("relative").style.bar(vmin = 0)

plotdata = cors.loc[genes_oi].mean(0)
plotdata = plotdata.loc[~plotdata.index.str.startswith("simulated")]
plotdata.plot.bar(ax = ax1)

# %%
genes_oi = cors.index[cors["baseline_v42"] < 0.05]
(1-(cors.loc[genes_oi].mean()/cors.loc[genes_oi, "v23"].values[:, None].mean()).sort_values().to_frame()).style.bar(vmin = 0)

# %%
genes_oi = cors.index[(cors["baseline_v42"] < 0.05)]
(1-(cors.loc[genes_oi].mean()/cors.loc[genes_oi, "v23"].values[:, None].mean()).sort_values().to_frame()).style.bar(vmin = 0)

# %%
fig, ax= plt.subplots(figsize = (4, 2))
plotdata = cors.loc[genes_oi]
for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    sns.ecdfplot(cor, label = key, color = color)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# %%
# gene_ranking = transcriptome.adata.var["dispersions_norm"].sort_values(ascending = False).index
# gene_ranking = transcriptome.adata.var["means"].sort_values(ascending = False).index
# gene_ranking = pd.Series(fragments.counts.mean(0), index = transcriptome.var.index).sort_values(ascending = False).index
# gene_ranking = cors["macs2_leiden_0.1_merged/linear"].sort_values(ascending = False).index
gene_ranking = cors["v23"].sort_values(ascending = False).index
plotdata = cors.loc[gene_ranking]
plotdata = (np.cumsum(plotdata) / np.arange(1, plotdata.shape[0] + 1)[:, None])

fig, ax = plt.subplots()

for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    plt.plot(np.arange(cor.shape[0]), cor, label = key, color = color)
ax.legend()


# %%
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plotdata = cors.loc[gene_ranking]
plotdata = (plotdata["v20"].values[:, None] - plotdata[["baseline_v42", "baseline_v21", "macs2_leiden_0.1_merged/linear"]])

fig, ax = plt.subplots()

for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    ax.scatter(np.arange(cor.shape[0]), cor, label = key, color = color, s= 0.3)
    plotdata_mean = plotdata[key].rolling(1000).mean().fillna(np.cumsum(plotdata[key])/np.arange(1, plotdata.shape[0] + 1))
    ax.plot(np.arange(plotdata.shape[0])[100:], plotdata_mean[100:], color = color)
ax.axhline(0.)
ax.legend()

# %%
methods_info = chdm.methods.prediction_methods.reindex(design["method"].unique())

methods_info["type"] = pd.Categorical(
    methods_info["type"], ["peak", "predefined", "rolling", "ours"]
)
methods_info["predictor"] = pd.Categorical(
    methods_info["predictor"], ["linear", "lasso", "xgboost"]
)
methods_info["subgroup"] = methods_info["type"] != "ours"
methods_info = methods_info.sort_values(["subgroup", "predictor", "type"])

methods_info["ix"] = -np.arange(methods_info.shape[0])

methods_info.loc[pd.isnull(methods_info["color"]), "color"] = "black"
methods_info.loc[pd.isnull(methods_info["label"]), "label"] = methods_info.index[
    pd.isnull(methods_info["label"])
]
methods_info["section"] = [
    predictor if ~pd.isnull(predictor) else type
    for predictor, type in zip(methods_info["predictor"], methods_info["type"])
]

section_info = methods_info.groupby("section").first()

# %%
metrics_info = pd.DataFrame(
    [
        {
            "label": "$\\Delta$ cor\n(method-baseline)",
            "metric": "cor_diff",
            "limits": (-0.01, 0.01),
            "transform": lambda x: x,
            "ticks": [-0.01, 0, 0.01],
            "ticklabels": ["-0.01", "0", "0.01"],
        },
        # {
        #     "label": "cor ratio",
        #     "metric": "cor_rel",
        #     "limits": (np.log(2 / 3), np.log(1.5)),
        #     "transform": lambda x: x,
        #     # "ticks": [-0.01, 0, 0.01],
        #     # "ticklabels": ["-0.01", "0", "0.01"],
        # },
    ]
).set_index("metric")
metrics_info["ix"] = np.arange(len(metrics_info))
metrics_info["ticks"] = metrics_info["ticks"].fillna(
    metrics_info.apply(
        lambda metric_info: [metric_info["limits"][0], 0, metric_info["limits"][1]],
        axis=1,
    )
)
metrics_info["ticklabels"] = metrics_info["ticklabels"].fillna(
    metrics_info.apply(lambda metric_info: metric_info["ticks"], axis=1)
)

# %%
datasets_info = pd.DataFrame(
    index=design.groupby(["dataset", "regions", "layer"]).first().index
)
datasets_info["label"] = datasets_info.index.get_level_values("dataset")
datasets_info = datasets_info.sort_values("label")
datasets_info["ix"] = np.arange(len(datasets_info))
datasets_info["label"] = datasets_info["label"].str.replace("-", "←\n")

# %% [markdown]
# ## Relative

# %%
import scanpy as sc

# %%
# !pip install scikit-misc

# %%
adata = transcriptome.adata.copy()
adata.X = adata.layers["normalized"]
sc.pp.highly_variable_genes(adata)

# %%
from fit_nbinom import fit_nbinom
params = []
for i in tqdm.tqdm(range(adata.var.shape[0])):
    X = np.array(adata.layers["counts"][:, i].todense())[:, 0]
    params.append(fit_nbinom(X))
params = pd.DataFrame(params, index = adata.var.index)

# %%
params["mean"] = adata.var["means"].values

# %%
plt.scatter(np.log(params["mean"]), np.log(params["size"]))

# %%
params["mean_bin"] = np.digitize(np.log(params["mean"]), bins = np.quantile(np.log(params["mean"]), np.linspace(0, 1, 10))) - 1
params["size_mean"] = params.groupby("mean_bin")["size"].median()[params["mean_bin"]].values
params["dispersion_mean"] = params["size_mean"]

# %%
import scipy.stats

def transform_parameters(mu, dispersion, eps = 1e-8):
    # avoids NaNs induced in gradients when mu is very low
    dispersion = np.clip(dispersion, 0, 20.0)

    logits = np.log(mu + eps) - np.log(1 / dispersion + eps)

    total_count = 1 / dispersion

    return total_count, logits

# transcriptome.adata.var = transcriptome.adata.var.copy()
# transcriptome.adata.var["dispersion_expected"] = transcriptome.adata.var["dispersions"] / transcriptome.adata.var["dispersions_norm"]

mu = (fragments.counts / (fragments.counts.mean(0, keepdims=True) + 1e-5)) * params["mean"].values
dispersion = params["dispersion_mean"].values

total_count, logits = transform_parameters(mu, dispersion)
probs = 1/(1+np.exp(logits))

expected_cors = []
for i in tqdm.trange(10):
    expression = np.random.negative_binomial(total_count, probs)
    expected_cors.append(chd.utils.paircor(mu, expression))
expected_cors = np.stack(expected_cors)
observed_cors = performance.genescores["cor"].mean("model").mean("phase").to_pandas()

# %%
expected_cors_mean = expected_cors.mean(0)

# %%
dummy_cors = scores.loc["counter"].loc["pbmc10k"].loc["10k10k"].loc["normalized"].loc["test"]["cor"]
observed_cors = scores.loc["v20"].loc["pbmc10k"].loc["10k10k"].loc["normalized"].loc["test"]["cor"]

# %%
fig, ax= plt.subplots(figsize = (2, 2))
sns.ecdfplot(expected_cors_mean)
sns.ecdfplot(observed_cors)
sns.ecdfplot(dummy_cors)

fig, ax= plt.subplots(figsize = (2, 2))
sns.ecdfplot(expected_cors_mean[-100:])
sns.ecdfplot(observed_cors[-100:])
sns.ecdfplot(dummy_cors[-100:])

# %% [markdown]
# ### Across datasets and metrics

# %%
score_relative_all = scores_all

# %%
panel_width = 5 / 4
panel_resolution = 1 / 4

fig, axes = plt.subplots(
    len(metrics_info),
    len(datasets_info),
    figsize=(
        len(datasets_info) * panel_width,
        len(metrics_info) * len(methods_info) * panel_resolution,
    ),
    gridspec_kw={"wspace": 0.05, "hspace": 0.2},
    squeeze=False,
)

for dataset, dataset_info in datasets_info.iterrows():
    axes_dataset = axes[:, dataset_info["ix"]].tolist()
    for metric, metric_info in metrics_info.iterrows():
        ax = axes_dataset.pop(0)
        ax.set_xlim(metric_info["limits"])
        plotdata = (
            pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [dataset], names=datasets_info.index.names
                )
            )
            .join(score_relative_all)
            .reset_index()
            .query("phase == 'test'")
        )
        plotdata = pd.merge(
            plotdata,
            methods_info,
            on="method",
        )

        ax.barh(
            width=plotdata[metric],
            y=plotdata["ix"],
            color=plotdata["color"],
            lw=0,
            zorder=0,
            # height=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # out of limits values
        metric_limits = metric_info["limits"]
        plotdata_annotate = plotdata.loc[
            (plotdata[metric] < metric_limits[0])
            | (plotdata[metric] > metric_limits[1])
        ]
        transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        metric_transform = metric_info.get("transform", lambda x: x)
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x=0.03 if left else 0.97,
                y=plotdata_row["ix"],
                s=f"{metric_transform(plotdata_row[metric]):+.2f}",
                transform=transform,
                va="center",
                ha="left" if left else "right",
                color="#FFFFFFCC",
                fontsize=6,
            )

    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Datasets
for dataset, dataset_info in datasets_info.iterrows():
    ax = axes[0, dataset_info["ix"]]
    ax.set_title(dataset_info["label"], fontsize=8)

# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[metric_info["ix"], 0]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])

    ax = axes[metric_info["ix"], 0]
    ax.set_xlabel(metric_info["label"])

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"])

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

# Sections
for ax in axes.flatten():
    for section in section_info["ix"]:
        ax.axhline(section + 0.5, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# %%
manuscript.save_figure(fig, "2", "positional_all_scores_datasets")

# %% [markdown]
# ### Averaged per dataset group

# %%
datagroups_info = pd.DataFrame(
    {
        "datagroup": ["within_dataset", "across_dataset", "across_celltypes"],
        "ix": [0, 1, 2],
        "label": [
            "Within dataset",
            "Across datasets\nSame cell types",
            "Across datasets\nDifferent cell types",
        ],
        "datasets": [
            ["brain", "e18brain", "lymphoma", "pbmc10k", "pbmc10k_gran"],
            ["pbmc10k_gran-pbmc10k", "pbmc3k-pbmc10k"],
            ["lymphoma-pbmc10k"],
        ],
    }
).set_index("datagroup")
datasets_info["datagroup"] = (
    datagroups_info.explode("datasets")
    .reset_index()
    .set_index("datasets")
    .loc[datasets_info.index.get_level_values("dataset"), "datagroup"]
    .values
)

# %%
group_ids = [*methods_info.index.names, "datagroup", "phase"]
scores_all["datagroup"] = pd.Categorical(
    (
        datasets_info["datagroup"]
        .reindex(scores_all.reset_index()[datasets_info.index.names])
        .values
    ),
    categories=datagroups_info.index,
)
score_relative_all = scores_all.groupby(group_ids).mean()

# %%
panel_width = 5 / 4
panel_resolution = 1 / 8

fig, axes = plt.subplots(
    len(metrics_info),
    len(datagroups_info),
    figsize=(
        len(datagroups_info) * panel_width,
        len(metrics_info) * len(methods_info) * panel_resolution,
    ),
    gridspec_kw={"wspace": 0.2, "hspace": 0.2},
    squeeze=False,
)

for datagroup, datagroup_info in datagroups_info.iterrows():
    axes_datagroup = axes[:, datagroup_info["ix"]].tolist()
    for metric, metric_info in metrics_info.iterrows():
        ax = axes_datagroup.pop(0)
        ax.set_xlim(metric_info["limits"])
        plotdata = (
            score_relative_all.xs(datagroup, level="datagroup")
            .reset_index()
            .query("phase == 'test'")
        )
        plotdata = pd.merge(
            plotdata,
            methods_info,
            on="method",
        )

        ax.barh(
            width=plotdata[metric],
            y=plotdata["ix"],
            color=plotdata["color"],
            lw=0,
            zorder=0,
            # height=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # out of limits values
        metric_limits = metric_info["limits"]
        plotdata_annotate = plotdata.loc[
            (plotdata[metric] < metric_limits[0])
            | (plotdata[metric] > metric_limits[1])
        ]
        transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        metric_transform = metric_info.get("transform", lambda x: x)
        for _, plotdata_row in plotdata_annotate.iterrows():
            left = plotdata_row[metric] < metric_limits[0]
            ax.text(
                x=0.03 if left else 0.97,
                y=plotdata_row["ix"],
                s=f"{metric_transform(plotdata_row[metric]):+.2f}",
                transform=transform,
                va="center",
                ha="left" if left else "right",
                color="#FFFFFFCC",
                fontsize=6,
            )

        # individual values
        plotdata = scores_all.loc[
            scores_all.index.get_level_values("dataset").isin(
                datagroup_info["datasets"]
            )
        ].query("phase == 'test'")
        plotdata = pd.merge(
            plotdata,
            methods_info,
            on="method",
        )
        ax.scatter(
            plotdata[metric],
            plotdata["ix"],
            # color=plotdata["color"],
            s=2,
            zorder=1,
            marker="|",
            color="#33333388",
        )

    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Datasets
for datagroup, datagroup_info in datagroups_info.iterrows():
    ax = axes[0, datagroup_info["ix"]]
    ax.set_title(datagroup_info["label"], fontsize=8)

# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[metric_info["ix"], 0]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])

    ax = axes[metric_info["ix"], 0]
    ax.set_xlabel(metric_info["label"])

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"], fontsize=8)

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

# Sections
for ax in axes.flatten():
    for section in section_info["ix"]:
        ax.axhline(section + 0.5, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

manuscript.save_figure(fig, "2", "positional_all_scores_datagroups")


# %% [markdown]
# ### Averaged over all datasets
# group_ids = [*methods_info.index.names, "phase"]
# score_relative_all = scores_all.groupby(group_ids).mean()

# %%
panel_width = 5 / 4
panel_resolution = 1 / 8

fig, axes = plt.subplots(
    1,
    len(metrics_info),
    figsize=(
        len(metrics_info) * panel_width,
        len(methods_info) * panel_resolution,
    ),
    gridspec_kw={"wspace": 0.2},
    squeeze=False,
)

for metric_ix, (metric, metric_info) in enumerate(metrics_info.iterrows()):
    ax = axes[0, metric_ix]
    ax.set_xlim(metric_info["limits"])
    plotdata = score_relative_all.reset_index().query("phase == 'test'")
    plotdata = pd.merge(
        plotdata,
        methods_info,
        on="method",
    )

    ax.barh(
        width=plotdata[metric],
        y=plotdata["ix"],
        color=plotdata["color"],
        lw=0,
        zorder=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # out of limits values
    metric_limits = metric_info["limits"]
    plotdata_annotate = plotdata.loc[
        (plotdata[metric] < metric_limits[0]) | (plotdata[metric] > metric_limits[1])
    ]
    transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    metric_transform = metric_info.get("transform", lambda x: x)
    for _, plotdata_row in plotdata_annotate.iterrows():
        left = plotdata_row[metric] < metric_limits[0]
        ax.text(
            x=0.03 if left else 0.97,
            y=plotdata_row["ix"],
            s=f"{metric_transform(plotdata_row[metric]):+.2f}",
            transform=transform,
            va="center",
            ha="left" if left else "right",
            color="#FFFFFFCC",
            fontsize=6,
        )


# Metrics
for metric, metric_info in metrics_info.iterrows():
    ax = axes[0, metric_info["ix"]]
    ax.set_xticks(metric_info["ticks"])
    ax.set_xticklabels(metric_info["ticklabels"])
    ax.set_xlabel(metric_info["label"])
    ax.axvline(0, color="#000000", lw=0.5, zorder=0, dashes=(2, 2))

# Methods
for ax in axes[:, 0]:
    ax.set_yticks(methods_info["ix"])
    ax.set_yticklabels(methods_info["label"], fontsize=8)

for ax in axes.flatten():
    ax.set_ylim(methods_info["ix"].min() - 0.5, 0.5)

manuscript.save_figure(fig, "2", "positional_all_scores")

# %% [markdown]
# ## Compare against CRE methods

# %%
method_oi1 = "v20"
method_oi2 = "macs2_improved/lasso"
# method_oi1 = "rolling_50/linear"
method_oi2 = "rolling_500/linear"
plotdata = pd.DataFrame(
    {
        "cor_b": scores.xs(method_oi2, level="method").xs("validation", level="phase")[
            "cor_diff"
        ],
        "dataset": scores.xs(method_oi2, level="method")
        .xs("validation", level="phase")
        .index.get_level_values("dataset"),
    }
)
plotdata["cor_total"] = scores.xs(method_oi1, level="method").xs(
    "validation", level="phase"
)["cor"]
plotdata["cor_a"] = scores.xs(method_oi1, level="method").xs(
    "validation", level="phase"
)["cor_diff"]
plotdata = plotdata.query("cor_total > 0.05")
plotdata["diff"] = plotdata["cor_a"] - plotdata["cor_b"]
plotdata = plotdata.sample(n=plotdata.shape[0])

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.axline((0, 0), slope=1, dashes=(2, 2), zorder=1, color="#333")
ax.axvline(0, dashes=(1, 1), zorder=1, color="#333")
ax.axhline(0, dashes=(1, 1), zorder=1, color="#333")
plt.scatter(
    plotdata["cor_b"],
    plotdata["cor_a"],
    # c=datasets_info.loc[plotdata["dataset"], "color"],
    alpha=0.5,
    s=1,
)
ax.set_xlim()
ax.set_xlabel(f"$\Delta$ cor {method_oi2}")
ax.set_ylabel(f"$\Delta$ cor {method_oi1}", rotation=0, ha="right", va="center")

# %%
plotdata.loc["pbmc10k"].sort_values("diff", ascending=False).head(20)

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / "pbmc10k"
    / "10k10k"
    / "random_5fold"
    / "v20"
)

scores_dir = prediction.path / "scoring"


# %%
# if you're interested in genes where the difference is small
plotdata.loc["pbmc10k"].assign(abs_diff=lambda x: np.abs(x["diff"]).abs()).query(
    "cor_a > 0.1"
).sort_values("abs_diff", ascending=True).to_csv(
    scores_dir / "difference_with_peakcalling_small.csv"
)

plotdata.loc["pbmc10k"].assign(abs_diff=lambda x: np.abs(x["diff"]).abs()).query(
    "cor_a > 0.1"
).sort_values("abs_diff", ascending=True).head(20)

# %%
# if you're interested in genes where the difference is large
plotdata.loc["pbmc10k"].query("cor_a > 0.1").sort_values(
    "diff", ascending=False
).to_csv(scores_dir / "difference_with_peakcalling_large.csv")

plotdata.loc["pbmc10k"].query("cor_a > 0.1").sort_values("diff", ascending=False).head(
    20
)

# %%

# %%
