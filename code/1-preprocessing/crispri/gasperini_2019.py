# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# ### Download CRISPRi data

# %%
folder = chd.get_output() / "data" / "crispri" / "gasperini_2019"

# %%
file = chd.get_output() / "data" / "crispri" / "gasperini_2019" / "GSE120861_all_deg_results.pilot.txt.gz"
file.parent.mkdir(parents=True, exist_ok=True)

import requests

if not file.exists():
    response = requests.get(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120861/suppl/GSE120861_all_deg_results.pilot.txt.gz"
    )
    with open(file, "wb") as f:
        f.write(response.content)

# %%
file = chd.get_output() / "data" / "crispri" / "gasperini_2019" / "GSE120861_all_deg_results.at_scale.txt.gz"
file.parent.mkdir(parents=True, exist_ok=True)

import requests

if not file.exists():
    response = requests.get(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120861/suppl/GSE120861_all_deg_results.at_scale.txt.gz"
    )
    with open(file, "wb") as f:
        f.write(response.content)

# %%
data = pd.read_table(file)

# %%
data["gene"] = data["pairs4merge"].str.split(":").str[1]

data.loc[data["pvalue.empirical.adjusted"] == "not_applicable", "pvalue.empirical.adjusted"] = 1
data["pvalue.empirical.adjusted"] = data["pvalue.empirical.adjusted"].astype(float)

# %%
data = data.loc[~data["gRNA_group"].str.startswith("scrambled")]
data = data.loc[~data["gRNA_group"].str.startswith("random")]
data = data.loc[~data["gRNA_group"].str.startswith("bassik")]

data = data.loc[data["target_site.chr"] != "NTC"].copy()

data["chrom"] = data["target_site.chr"]
data["start"] = data["target_site.start"].astype(int)
data["end"] = data["target_site.stop"].astype(int)
data["HS_LS_logratio"] = data["beta"]

# %%
import liftover

converter = liftover.get_lifter("hg19", "hg38")
data["start_orig"] = data["start"]
data["end_orig"] = data["end"]

starts = []
ends = []
for chrom, start, end in zip(data["chrom"], data["start_orig"], data["end_orig"]):
    try:
        starts.append(converter[chrom][start][0][1])
    except:
        starts.append(np.nan)
    try:
        ends.append(converter[chrom][end][0][1])
    except:
        ends.append(np.nan)
data["start"] = starts
data["end"] = ends

data = data.loc[~data["start"].isna()].copy()

# %%
dataset_name = "hspc"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "hspc" / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / "hspc" / "fragments" / "100k100k")

# %%
data_orig = []
for gene in fragments.regions.coordinates.index:
    if gene in data["ENSG"].tolist():
        subdata = data.loc[data["ENSG"] == gene]
        region = fragments.regions.coordinates.loc[gene]

        subdata = subdata.loc[subdata["chrom"] == region["chrom"]]
        subdata = subdata.loc[subdata["start"] >= region["start"]]
        subdata = subdata.loc[subdata["end"] <= region["end"]]

        if len(subdata) > 0:
            subdata["HS_LS_logratio"] = subdata["beta"]
            subdata["significant"] = subdata["pvalue.empirical.adjusted"] < 0.05
            subdata["Gene"] = transcriptome.symbol(gene)

            data_orig.append(subdata)
data_orig = pd.concat(data_orig)

# %%
data = data_orig

# %% [markdown]
# ### Store

# %%
data.to_csv(folder / "data.tsv", sep="\t", index=False)

# %% [markdown]
# ### Load interpretation

# %%
dataset_name = "hspc"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "hspc" / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / "hspc" / "fragments" / "100k100k")

# %%
model_folder = chd.get_output() / "pred" / "pbmc10k" / "100k100k" / "5x5" / "normalized" / "v20"

models = [chd.models.pred.model.additive.Model(model_folder / str(model_ix)) for model_ix in range(25)]
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / "pbmc10k" / "folds" / "5x5")

symbols_oi = data.loc[data["significant"]]["Gene"].unique()
genes_oi = transcriptome.gene_id(symbols_oi)
censorer = chd.models.pred.interpret.MultiWindowCensorer(
    fragments.regions.window,
    (
        100,
        500,
    ),
    relative_stride=1,
)

scoring_folder = model_folder / "scoring" / "crispri" / "gasperini_2019"

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(scoring_folder / "regionmultiwindow")
regionmultiwindow.score(fragments, transcriptome, models, folds, censorer, regions=genes_oi)
regionmultiwindow.interpolate()

# %% [markdown]
# ### Check

# %%
data["Significant"] = data["significant"].astype(int)

# %%
joined_all = []
for gene_oi in tqdm.tqdm(genes_oi):
    # for gene_oi in tqdm.tqdm(transcriptome.gene_id(["JUNB"])):
    region = fragments.regions.coordinates.loc[gene_oi]
    symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

    data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
    data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
    data_oi["start"] = data_oi["start"].astype(int)
    data_oi["end"] = data_oi["end"].astype(int)

    data_oi["z"] = (data["HS_LS_logratio"]) / data["HS_LS_logratio"].std()

    data_oi = data_oi.loc[data_oi["start"] > region["start"]]
    data_oi = data_oi.loc[data_oi["end"] < region["end"]]

    if data_oi.shape[0] > 0:
        data_centered = chd.plot.genome.genes.center(data_oi, region)
        data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

        data_centered["bin"] = regionmultiwindow.design.index[
            np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"]) - 1
        ]
        data_binned = data_centered.groupby("bin").mean(numeric_only=True)

        joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
        joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

        joined_shared = joined.loc[joined["HS_LS_logratio"].notnull()].copy()
        joined_shared["gene"] = gene_oi
        joined_all.append(joined_shared)
joined_all = pd.concat(joined_all)

# %%
cutoff = -0.001

# %%
fig, ax = plt.subplots(figsize=(5, 2))
ax.scatter(
    joined_all["deltacor"],
    joined_all["HS_LS_logratio"],
    c=joined_all["Significant"],
    cmap="viridis",
    s=10,
    vmin=0,
    vmax=1,
)
ax.axvline(cutoff)

fig, ax = plt.subplots(figsize=(5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_all["gene"].unique())), index=joined_all["gene"].unique())
ax.scatter(joined_all["deltacor"], joined_all["HS_LS_logratio"], c=colors[joined_all["gene"]])
for gene, color in colors.items():
    ax.scatter([], [], c=color, label=transcriptome.symbol(gene))
ax.axvline(cutoff)

# %%
joined_all["significant_expression"] = joined_all["Significant"] > 0
joined_all["significant_chd"] = joined_all["deltacor"] < -0.005

# %%
confirmed = (joined_all["significant_expression"] & joined_all["significant_chd"]).sum() / joined_all[
    "significant_chd"
].sum()

randoms = []
for i in range(1000):
    randoms.append(
        (
            joined_all.iloc[np.random.permutation(np.arange(len(joined_all)))]["significant_expression"].values
            & joined_all["significant_chd"]
        ).sum()
        / joined_all["significant_chd"].sum()
    )
randoms = np.array(randoms)

fig, ax = plt.subplots(figsize=(5, 2))
ax.hist(randoms, bins=np.linspace(0, 1, 20), density=True)
ax.axvline(confirmed, color="red")
(randoms >= confirmed).mean()

# %%
import fisher

contingency = pd.crosstab(joined_all["significant_expression"], joined_all["significant_chd"])
fisher.pvalue(*contingency.values.flatten())
odds = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / (contingency.iloc[1, 0] * contingency.iloc[0, 1])
odds

# %%
data.groupby("Gene")["significant"].sum().sort_values(ascending=False).head(10)

# %%
gene_oi = transcriptome.gene_id("TMSB4X")
gene_oi = transcriptome.gene_id("LMO2")
gene_oi = transcriptome.gene_id("CXCL2")
# gene_oi = transcriptome.gene_id("MGAT3")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
data_oi["start"] = data_oi["start"].astype(int)
data_oi["end"] = data_oi["end"].astype(int)

data_oi = data_oi.loc[data_oi["start"] > region["start"]]
data_oi = data_oi.loc[data_oi["end"] < region["end"]]

data_centered = chd.plot.genome.genes.center(data_oi, region)
data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

data_centered["bin"] = regionmultiwindow.design.index[
    np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"]) - 1
]
data_binned = data_centered.groupby("bin").mean(numeric_only=True)

# %%
joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 20000]
# window = [-10000, 10000] # TSS
window = [40000, 50000]  # IL10 Enhancer

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width=10, window=window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["HS_LS_logratio"], lw=0, width=binwidth)
ax.set_xlim(*window)
ax.set_ylabel("CRISPRi score", rotation=0, ha="right", va="center")

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["deltacor"], lw=0, width=binwidth)
ax.set_xlim(*window)
ax.set_ylabel("$\Delta cor$", rotation=0, ha="right", va="center")

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["lost"], lw=0, width=binwidth)
ax.set_xlim(*window)
ax.set_ylabel("# fragments", rotation=0, ha="right", va="center")

panel, ax = fig.main.add_under(
    chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window=fragments.regions.window, width=10)
)
ax.set_xlim(*window)

fig.plot()

# %% [markdown]
# ## Slices

# %%
significant_cutoff = np.log(1.2)

joined_genes = {}
for gene_oi in tqdm.tqdm(genes_oi):
    # for gene_oi in tqdm.tqdm(transcriptome.gene_id(["JUNB"])):
    region = fragments.regions.coordinates.loc[gene_oi]
    symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

    data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
    data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
    data_oi["start"] = data_oi["start"].astype(int)
    data_oi["end"] = data_oi["end"].astype(int)

    data_oi = data_oi.loc[data_oi["start"] > region["start"]]
    data_oi = data_oi.loc[data_oi["end"] < region["end"]]

    if data_oi.shape[0] > 0:
        data_centered = chd.plot.genome.genes.center(data_oi, region)
        data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

        data_centered["bin"] = regionmultiwindow.design.index[
            np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"]) - 1
        ]
        data_binned = data_centered.groupby("bin").mean(numeric_only=True)
        data_binned["n_guides"] = data_centered.groupby("bin").size()

        data_binned = data_binned.reindex(regionmultiwindow.design.index)

        joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
        joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

        joined["significant_expression"] = joined["HS_LS_logratio"].abs() > significant_cutoff

        joined_genes[gene_oi] = joined

# %%
# get aupr score
import sklearn.metrics
import chromatinhd.data.peakcounts

# %%
genescores = {}
slicescores = {}
allslicescores = {}

peakcallers = [
    "macs2_leiden_0.1_merged",
    # "macs2_leiden_0.1",
    "cellranger",
    "rolling_500",
]

# %%
for peakcaller in peakcallers:
    peakcounts = chromatinhd.data.peakcounts.PeakCounts(
        path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "100k100k"
    )
    X_peaks = peakcounts.counts
    X_transcriptome = transcriptome.layers["normalized"]

    def extract_data(gene_oi, peaks_oi):
        x = np.array(X_peaks[:, peaks_oi["ix"]].todense())
        y = np.array(X_transcriptome[:, gene_oi["ix"]])
        return x, y

    transcriptome.var["ix"] = np.arange(transcriptome.var.shape[0])

    var_peaks = peakcounts.var
    var_peaks["ix"] = np.arange(var_peaks.shape[0])

    method_name = peakcaller + "/" + "cor"
    genescores[method_name] = []
    slicescores[method_name] = []
    allslicescores[method_name] = []

    for gene_oi in tqdm.tqdm(genes_oi):
        # get correlation between peak and expression
        peak_gene_links_oi = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi].copy()

        peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
        gene = transcriptome.var.loc[gene_oi]

        x, y = extract_data(gene, peaks_oi)

        cors = np.corrcoef(x.T, y[:, None].T)[:-1, -1]
        peakscores = cors

        peak_gene_links_oi["cor"] = cors

        joined_oi = joined_genes[gene_oi]

        cor = np.zeros(len(joined_oi))
        joined_oi["cor"] = 0.0
        for peak_id, peak_gene_link in peak_gene_links_oi.iterrows():
            cor[
                np.arange(
                    np.searchsorted(regionmultiwindow.design["window_start"], peak_gene_link.relative_start),
                    np.searchsorted(regionmultiwindow.design["window_start"], peak_gene_link.relative_end),
                )
            ] = peak_gene_link["cor"]
        joined_oi["cor"] = cor
        joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

        #
        joined_oi["score"] = joined_oi["cor"].abs() + 0.0001 * np.random.randn(len(joined_oi))

        allslicescores[method_name].append(joined_oi.assign(method=method_name, gene=gene_oi))

        if joined_oi["significant_expression"].sum() > 0:
            aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
        else:
            aupr = 1

        # subset on significant ones
        joined_cre = joined_oi.loc[(joined_oi["cor"].abs() >= 0.05)]

        genescores[method_name].append(
            {
                "gene": gene_oi,
                "method": method_name,
                "lfc_mean": joined_cre["HS_LS_logratio"].abs().mean(),
                "n": joined_cre.shape[0],
                "aupr": aupr,
            }
        )

        slicescores[method_name].append(
            pd.DataFrame(
                {
                    "gene": gene_oi,
                    "method": method_name,
                    "lfc": joined_cre["HS_LS_logratio"].abs(),
                    "slice": joined_cre.index,
                    "score": joined_cre["cor"].abs(),
                }
            )
        )

# %%
fig, ax = plt.subplots()
for gene_oi in regionmultiwindow.scores.keys():
    regionmultiwindow.scores[gene_oi].mean("model")["deltacor"].to_pandas().plot()
ax.axhline(-0.001)

# %%
for deltacor_cutoff in [-0.001]:
    method_name = "v20" + "/" + "{:.0e}".format(deltacor_cutoff)
    print(method_name)
    genescores[method_name] = []
    slicescores[method_name] = []
    allslicescores[method_name] = []
    for gene_oi in tqdm.tqdm(genes_oi):
        joined_oi = joined_genes[gene_oi]
        joined_oi["significant_chd"] = joined_oi["deltacor"] < deltacor_cutoff
        joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

        joined_oi["score"] = -joined_oi["deltacor"]

        if joined_oi["significant_expression"].sum() > 0:
            aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
        else:
            aupr = 1

        allslicescores[method_name].append(joined_oi.assign(method=method_name, gene=gene_oi))

        joined_chd = joined_oi.loc[joined_oi["significant_chd"]]

        genescores[method_name].append(
            {
                "gene": gene_oi,
                "method": method_name,
                "lfc_mean": joined_chd["HS_LS_logratio"].abs().mean(),
                "n": joined_chd.shape[0],
                "aupr": aupr,
            }
        )

        slicescores[method_name].append(
            pd.DataFrame(
                {
                    "gene": gene_oi,
                    "method": method_name,
                    "lfc": joined_chd["HS_LS_logratio"].abs(),
                    "score": -joined_chd["deltacor"],
                    "slice": joined_chd.index,
                }
            )
        )

# %%
method_name = "all"
genescores[method_name] = []
slicescores[method_name] = []
allslicescores[method_name] = []
for gene_oi in tqdm.tqdm(genes_oi):
    joined_oi = joined_genes[gene_oi]
    joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
    genescores[method_name].append(
        {
            "gene": gene_oi,
            "method": method_name,
            "lfc_mean": joined_oi["HS_LS_logratio"].abs().mean(),
            "n": joined_chd.shape[0],
        }
    )
    slicescores[method_name].append(
        pd.DataFrame(
            {
                "gene": gene_oi,
                "method": method_name,
                "lfc": joined_oi["HS_LS_logratio"].abs(),
                "score": 0.0,
                "slice": joined_oi.index,
            }
        )
    )

    allslicescores[method_name].append(
        joined_oi.assign(method=method_name, gene=gene_oi, score=np.random.rand(joined_oi.shape[0]))
    )

# %%
allslicescores_stacked = pd.concat(
    [pd.concat(allslicescores[method_name]) for method_name in allslicescores.keys()], ignore_index=True
)
fig, (ax_aupr, ax_auroc) = plt.subplots(1, 2, figsize=(5, 2))

eps = 0.0
eps = 1e-5

methodscores = []
for method_name, slicescores_oi in allslicescores_stacked.groupby("method"):
    slicescores_oi["significant_expression"] = slicescores_oi["HS_LS_logratio"].abs() > np.log(1.2)
    print(len(slicescores_oi), len(slicescores_oi["score"].unique()))
    print(slicescores_oi["significant_expression"].mean())
    aupr = sklearn.metrics.average_precision_score(
        slicescores_oi["significant_expression"],
        slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]),
    )
    auroc = sklearn.metrics.roc_auc_score(slicescores_oi["significant_expression"], slicescores_oi["score"])

    curve = sklearn.metrics.precision_recall_curve(
        slicescores_oi["significant_expression"],
        slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]),
    )
    ax_aupr.plot(curve[1], curve[0], label=method_name)

    curve = sklearn.metrics.roc_curve(
        slicescores_oi["significant_expression"],
        slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]),
    )
    ax_auroc.plot(curve[0], curve[1], label=method_name)
    ax_auroc.plot([0, 1], [0, 1], color="black", linestyle="--")

    methodscores.append(
        {
            "method": method_name,
            "aupr": aupr,
            "auroc": auroc,
        }
    )
methodscores = pd.DataFrame(methodscores)
# legend outside
ax_auroc.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
methodscores.style.bar()

# %%
slicescores_stacked = pd.concat([pd.concat(x) for x in list(slicescores.values())])

# %%
genescores_stacked = pd.concat([pd.DataFrame(genescore) for genescore in genescores.values()], ignore_index=True)
genescores_stacked.groupby("method")["aupr"].mean()

# %%
cumscores = []
fig, ax = plt.subplots()
for method_name, methoddata in slicescores_stacked.groupby("method"):
    ax.plot(
        np.arange(len(methoddata)),
        (methoddata.sort_values("score", ascending=False)["lfc"].cumsum() / (np.arange(len(methoddata)) + 1)),
        label=method_name,
    )
plt.legend()
ax.set_xlim(0, 200)

# %%
fig, ax = plt.subplots()
methods = pd.DataFrame(
    {
        "method": list(genescores.keys()),
        "ix": np.arange(len(genescores)),
    }
).set_index("method")
for method, plotdata in slicescores_stacked.groupby("method"):
    sns.ecdfplot(x="lfc", data=plotdata.dropna(), ax=ax, label=f"{method} (n={plotdata.shape[0]})")
plt.legend()

# %%
slicescores_mean = slicescores_stacked.groupby("method").mean(numeric_only=True)
slicescores_mean["n"] = slicescores_stacked.groupby("method").size()
slicescores_mean.style.bar()

plotdata = slicescores_mean
fig, ax = plt.subplots(figsize=(2, 1.5))
ax.scatter(np.exp(plotdata["lfc"]), plotdata.index, color="#333")
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
# set xtick label rotation
for tick in ax.get_xticklabels(which="minor"):
    tick.set_rotation(45)

# %%
plotdata = slicescores_stacked.groupby(["gene", "method"]).mean(numeric_only=True).unstack()["lfc"].T

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_width=0.1))

cmap = mpl.colormaps["YlGnBu"]
cmap.set_bad("#EEEEEE")

norm = mpl.colors.Normalize(vmin=0)

panel, ax = fig.main.add_under(polyptich.grid.Panel(np.array(plotdata.shape)[::-1] * 0.25))
ax.matshow(plotdata.values, cmap=cmap, aspect="auto", norm=norm)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_xticks(np.arange(len(plotdata.columns)))
ax.set_xticklabels(transcriptome.symbol(plotdata.columns), rotation=45, ha="left")
ax.set_yticks(np.arange(len(plotdata.index)))
ax.set_yticklabels(plotdata.index)

panel, ax = fig.main.add_right(polyptich.grid.Panel([0.25, plotdata.shape[0] * 0.25]))
ax.matshow(slicescores_mean["lfc"].values[:, None], cmap=cmap, aspect="auto", norm=norm)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_xticks([0])
ax.set_xticklabels(["Mean"], rotation=45, ha="left")
ax.set_yticks([])

fig.plot()

# %%
slicescores_stacked.query("method == 'v20/-1e-03'").sort_values("lfc", ascending=False)

# %%
# explore the results of one gene, where is the increase in lfc coming from?
gene_oi = transcriptome.gene_id("CXCL2")
slicescores_stacked.query("method == 'v20/-1e-03'").query("gene == @gene_oi").sort_values("lfc", ascending=False)
