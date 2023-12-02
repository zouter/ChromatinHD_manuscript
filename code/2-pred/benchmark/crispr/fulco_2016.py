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
from chromatinhd_manuscript import crispri

from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")
chd.set_default_device("cuda:0")

# %% [markdown]
# ### Download CRISPRi data

# %%
file = chd.get_output() / "data" / "crispri" / "fulco_2016" / "aag2445_table_s2.xlsx"
file.parent.mkdir(parents=True, exist_ok=True)
import pathlib
file = pathlib.Path("aag2445_table_s2_2.xlsx")

import requests
if not file.exists():
    response = requests.get("https://www.science.org/doi/suppl/10.1126/science.aag2445/suppl_file/aag2445_table_s2.xlsx")
    with open(file, "wb") as f:
        f.write(response.content)

# %%
data_orig = pd.read_excel(file, skiprows=1, sheet_name="Table S2")
data_orig = data_orig.loc[data_orig["Set"].isin(["MYC Tiling", "GATA1 Tiling"])].copy()
data_orig["chrom"] = data_orig["chr"]
data_orig["HS_LS_logratio"] = data_orig["CRISPRi Score"]

data = data_orig.copy()
data["start_orig"] = data["start"]
data["end_orig"] = data["end"]

import liftover
converter = liftover.get_lifter("hg19", "hg38")
data["start"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["start_orig"])]
data["end"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["end_orig"])]

# %%
data["Gene"] = data["Set"].str.split(" ").str[0]

# %%
data["Significant"] = np.abs(data["CRISPRi Score"]) > 0.5

# %% [markdown]
# ### Load interpretation

# %%
dataset_name = "hspc"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")

# dataset_name = "hspc_focus"
# transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
# fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "500k500k")

# %%
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
# folds.sample_cells(fragments, 5, 5)
# fold = folds[0]
# folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "1")
# folds.folds = [fold]

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)

data = data.loc[data["Gene"].isin(symbols_oi)].copy()
data["gene"] = transcriptome.gene_id(data["Gene"]).values

# %%
import chromatinhd.models.pred.model.better
model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / "5x1" / "magic" / "v30"

# %%
performance = chd.models.pred.interpret.Performance(model_folder / "scoring" / "performance")
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(model_folder / "scoring" / "regionmultiwindow",)

# %% [markdown]
# ## Check enrichment

# %%
genes_oi = genes_oi[regionmultiwindow.scores["scored"].sel_xr(genes_oi).all("fold").values]

# %%
# joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 500)
joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 100)
joined_observed = joined_all.loc[~pd.isnull(joined_all["HS_LS_logratio"])].copy()
joined_observed["deltacor_positive"] = np.where(joined_observed["effect"] > 0, joined_observed["deltacor"], 0)

# %%
deltacor_cutoff = -0.001
lfc_cutoff = np.log(1.5)

# %%
joined_observed["significant_expression"] = joined_observed["HS_LS_logratio"].abs() > lfc_cutoff
joined_observed["significant_chd"] = joined_observed["deltacor_positive"] < deltacor_cutoff

# %%
genescores = []
for gene_oi, joined_oi in joined_observed.groupby("gene"):
    confirmed_n  = (joined_oi["significant_expression"] & joined_oi["significant_chd"]).sum()
    total_n = joined_oi["significant_chd"].sum()
    confirmed = confirmed_n / total_n

    randoms = []
    for i in range(1000):
        randoms.append((joined_oi.iloc[np.random.permutation(np.arange(len(joined_oi)))]["significant_expression"].values & joined_oi["significant_chd"]).sum() / joined_oi["significant_chd"].sum())
    randoms = np.array(randoms)

    p = (randoms > confirmed).mean()

    genescores.append({
        "gene": gene_oi,
        "symbol": transcriptome.symbol(gene_oi),
        "observed": confirmed,
        "random": randoms.mean(),
        "p": p
    })

    print(f"{transcriptome.symbol(gene_oi)} Observed {confirmed:.2%} Random {randoms.mean():.2%} p-value {p:.2f}")
genescores = pd.DataFrame(genescores).set_index("gene")
genescores["dispersions_norm"] = transcriptome.var.loc[genescores.index, "dispersions_norm"].values
genescores["dispersions"] = transcriptome.var.loc[genescores.index, "dispersions"].values

genescores.style.bar()

# %%
fig, ax =plt.subplots(figsize = (5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_observed["gene"].unique())), index = joined_observed["gene"].unique())
ax.scatter(joined_observed["deltacor_positive"], joined_observed["HS_LS_logratio"], c = colors[joined_observed["gene"]], s = 2)
for gene, color in colors.items():
    ax.scatter([], [], c = color, label = transcriptome.symbol(gene))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axvline(deltacor_cutoff, color = "red", dashes = (2, 2))
ax.axhline(0, color = "#333", lw = 1, zorder = 0)
ax.axhline(-lfc_cutoff, color = "red", dashes = (2, 2))
ax.set_xlabel("$\Delta$ cor")
ax.set_ylabel("Log-fold change")
ax.set_yticks([-lfc_cutoff, 0, lfc_cutoff])
ax.set_yticklabels(["1/2", "1", "2"])
ax.set_ylabel(f"CRISPRi\nfold enrichment\nhigh vs low RNA", rotation = 0, ha = "right", va = "center")
ax.set_yticks(np.log([0.125, 0.25, 0.5, 1, 2]))
ax.set_yticklabels(["⅛", "¼", "½", 1, 2])
ax.set_xlim(ax.get_xlim())
ax.axvspan(ax.get_xlim()[0], deltacor_cutoff, color = "grey", zorder = -1, alpha = 0.1)
ax.annotate("{} High $\Delta$ cor+ low CRISPRi depletion".format(total_n - confirmed_n), (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top", fontsize = 8)
ax.annotate("{} High $\Delta$ cor+ high CRISPRi depletion".format(confirmed_n), (0.05, 0.05), xycoords = "axes fraction", ha = "left", va = "bottom", fontsize = 8)

# %% [markdown]
# ## Focus on one gene

# %%
gene_oi = transcriptome.gene_id("GATA1")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
joined = joined_all.query("gene == @gene_oi").copy()
joined["score"] = joined["HS_LS_logratio"] * joined["deltacor"]

# %%
regionmultiwindow2 = regionmultiwindow

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.1))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = [-10000, 20000]  # GATA1 region oi
arrows = [{"position": 14000, "orientation": "right", "y": 0.5}]

# window = [-20000, 20000]  # HDAC6 region oi
# arrows = [{"position": 14000, "orientation": "right", "y": 0.5}]

# window = [-260000, -250000]
# arrows = []

# window = [-10000, 10000] # KLF1 TSS
# arrows = [
#     {"position": -3500, "orientation": "left", "y": 0.5},
#     {"position": 1000, "orientation": "left", "y": 0.5},
#     {"position": -650, "orientation": "right", "y": 0.5},
# ]

# window = [-65000, -45000] # CALR upstream
# arrows = []

# window = fragments.regions.window  # all
# arrows = []

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width=10, window=window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Pileup(regionmultiwindow2.get_plotdata(gene_oi), window=window, width=10)
)  # (regionmultiwindow2.get_plotdata(gene_oi), window = window, width = 10))

panel, ax = fig.main.add_under(
    chdm.plotting.Peaks(
        region,
        chd.get_output() / "peaks" / dataset_name,
        window=fragments.regions.window,
        width=10,
        peakcallers=["cellranger", "macs2_improved", "macs2_leiden_0.1", "macs2_leiden_0.1_merged", "encode_screen"],
    )
)
ax.set_xlim(*window)
ax.set_xticks([])

panel, ax = fig.main.add_under(chd.grid.Panel((10, 1)))
ax.bar(
    joined["window_mid"],
    joined["HS_LS_logratio"],
    lw=0,
    width=binwidth,
    color="#333",
)
ax.step(
    joined["window_mid"],
    joined["HS_LS_logratio"].fillna(0.0),
    lw=0.5,
    color="#AAA",
    where="mid",
    zorder=0,
    linestyle="--",
)
ax.set_ylim(*ax.get_ylim()[::-1])
ax.set_xlim(*window)
ax.set_ylabel(f"CRISPRi\nfold enrichment\ngrowth selection", rotation=0, ha="right", va="center")
# ax.set_ylabel(f"CRISPRi\nfold enrichment\nhigh vs low {transcriptome.symbol(gene_oi)}", rotation = 0, ha = "right", va = "center")
ax.set_xticks([])
ax.set_yticks(np.log([0.125, 0.25, 0.5, 1, 2]))
ax.set_yticklabels(["⅛", "¼", "½", 1, 2])

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Predictivity(regionmultiwindow2.get_plotdata(gene_oi), window=window, width=10)
)
for arrow in arrows:
    panel.add_arrow(**arrow)

panel, ax = fig.main.add_under(chd.grid.Panel((10, 0.5)))
# # !wget https://www.encodeproject.org/files/ENCFF010PHG/@@download/ENCFF010PHG.bigWig
import pyBigWig
file = pyBigWig.open("ENCFF010PHG.bigWig")
plotdata = pd.DataFrame({"value":file.values(region["chrom"], region["start"], region["end"])})
plotdata["position"] = np.arange(*fragments.regions.window)[::int(region["strand"])]
ax.plot(plotdata["position"], plotdata["value"], color = "#333")
ax.set_xlim(*window)
ax.set_ylim(0, 20)
ax.set_ylabel("H3K27ac\nsignal", rotation = 0, ha = "right", va = "center")

panel, ax = fig.main.add_under(chd.grid.Panel((10, 0.5)))
# # !wget https://www.encodeproject.org/files/ENCFF814IYI/@@download/ENCFF814IYI.bigWig
import pyBigWig
file = pyBigWig.open("ENCFF814IYI.bigWig")
plotdata = pd.DataFrame({"value":file.values(region["chrom"], region["start"], region["end"])})
plotdata["position"] = np.arange(*fragments.regions.window)[::int(region["strand"])]
ax.plot(plotdata["position"], plotdata["value"], color = "#333")
ax.set_xlim(*window)
ax.set_ylim(0, 20)
ax.set_ylabel("H3K4me3\nsignal", rotation = 0, ha = "right", va = "center")

panel, ax = fig.main.add_under(chd.grid.Panel((10, 0.5)))
# # !wget https://www.encodeproject.org/files/ENCFF242ENK/@@download/ENCFF242ENK.bigWig
import pyBigWig
file = pyBigWig.open("ENCFF242ENK.bigWig")
plotdata = pd.DataFrame({"value":file.values(region["chrom"], region["start"], region["end"])})
plotdata["position"] = np.arange(*fragments.regions.window)[::int(region["strand"])]
ax.plot(plotdata["position"], plotdata["value"], color = "#333")
ax.set_xlim(*window)
ax.set_ylim(0, 20)
ax.set_ylabel("H3K27me\nsignal", rotation = 0, ha = "right", va = "center")

fig.plot()

# %% [markdown]
# ## Slices

# %%
import sklearn.metrics

# %%
genes_oi = list(set(joined_observed["gene"].unique()) & set(regionmultiwindow.scores.keys()))

# %%
window_size = 100
# window_size = 500

joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = window_size)
windows_oi = regionmultiwindow.design.query("window_size == @window_size").index
design = regionmultiwindow.design.loc[windows_oi]

# lfc_cutoff = np.log(1.2)
lfc_cutoff = np.log(1.5)
# lfc_cutoff = np.log(2.0)
joined_all["significant_expression"] = joined_all["HS_LS_logratio"].abs() > lfc_cutoff

joined_genes = {k:v for k, v in joined_all.groupby("gene")}

# %%
import sklearn.metrics
import chromatinhd.data.peakcounts
import scipy.stats

# %%
joined_all.dropna()["significant_expression"].mean()

# %%
genescores = {}
slicescores = {}
allslicescores = {}

# %%
ws = [0, 1, 2, 3, 5, 7, 10 ,20,25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

# %%
deltacor_cutoff = -0.001

method_names = [
    "v30",
    # "v30/positive",
    # "v30/negative",
    # "v30/effect",
    "accessibility"
]
for method_name in method_names:
    print(method_name)
    genescores[method_name] = []
    slicescores[method_name] = []
    allslicescores[method_name] = []
    for gene_oi in tqdm.tqdm(genes_oi):
        for w in ws:
            joined_oi = joined_genes[gene_oi]
            joined_oi["significant_cor"] = joined_oi["deltacor"] < deltacor_cutoff

            joined_oi["score"] = -joined_oi["deltacor"]
            if method_name == "v30":
                joined_oi["score"] = crispri.rolling_max(-joined_oi["deltacor"].values, w)
            if method_name.endswith("positive"):
                joined_oi["score"] = crispri.rolling_max(-np.where(joined_oi["effect"] > 0, joined_oi["deltacor"].values, 0), w)
            elif method_name.endswith("negative"):
                joined_oi["score"] = crispri.rolling_max(-np.where(joined_oi["effect"] < 0, joined_oi["deltacor"].values, 0), w)
            elif method_name.endswith("effect"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["effect"].values * (-joined_oi["deltacor"].values), w)
            elif method_name == "accessibility":
                joined_oi["score"] = crispri.rolling_max(joined_oi["lost"].values, w)

            joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
            
            aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
            auroc = sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], joined_oi["score"])
            
            joined_oi["score_untied"] = joined_oi["score"] + np.random.normal(scale = 1e-5, size = len(joined_oi))
            cor = np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0, 1]
            spearman = scipy.stats.spearmanr(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0]

            confirmed = (joined_oi["significant_expression"] & joined_oi["significant_cor"]).sum() / joined_oi["significant_cor"].sum()

            joined_oi["score_rank"] = joined_oi["score"].rank(ascending = False) / joined_oi.shape[0]

            allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, w = w))

            joined_chd = joined_oi.loc[joined_oi["significant_cor"]]

            genescores[method_name].append({
                "gene": gene_oi,
                "method": method_name,
                "lfc_mean": joined_chd["HS_LS_logratio"].abs().mean(),
                "n": joined_chd.shape[0],
                "aupr":aupr,
                "auroc":auroc,
                "cor":cor,
                "spearman":spearman,
                "confirmed":confirmed,
                "w":w,
            })

            slicescores[method_name].append(pd.DataFrame({
                "gene": gene_oi,
                "method": method_name,
                "lfc": joined_chd["HS_LS_logratio"].abs(),
                "score": joined_chd["score"],
                "slice":joined_chd.index,
                "w":w,
            }))

# %%
gene_oi = transcriptome.gene_id("GATA1")
joined_oi = joined_genes[gene_oi]
joined_oi["significant_cor"] = joined_oi["deltacor"] < deltacor_cutoff

joined_oi["score"] = -joined_oi["deltacor"]
joined_oi["score_original"] = joined_oi["score"]
joined_oi["score"] = crispri.rolling_max(-joined_oi["deltacor"].values, 10)
joined_oi_all = joined_oi.copy()

joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

fig, ax = plt.subplots(figsize = (10, 2))
ax.plot(joined_oi_all["window_mid"], np.abs(joined_oi_all["HS_LS_logratio"]))
ax.axhline(lfc_cutoff, color = "blue", linestyle = "--", lw = 1)

ax2 = ax.twinx()
ax2.plot(joined_oi_all["window_mid"], -joined_oi_all["deltacor"], color = "green")
ax2.plot(joined_oi_all["window_mid"], joined_oi_all["score"], color = "red")

ax.set_xlim([-10000, 20000])
# ax.set_xlim([-10000, 10000])

# %%
peakcallers = [
    "macs2_leiden_0.1_merged",
    # "macs2_leiden_0.1",
    "cellranger",
    "rolling_500",
    # "genrich",
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

    for method_name in [peakcaller + "/linear_absolute"]:
    # for method_name in [peakcaller + "/linear", peakcaller + "/linear_absolute", peakcaller + "/linear_positive"]:
        genescores[method_name] = []
        slicescores[method_name] = []
        allslicescores[method_name] = []

        for gene_oi in tqdm.tqdm(genes_oi):
            # get correlation between peak and expression
            peak_gene_links_oi = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi].copy()

            peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
            gene = transcriptome.var.loc[gene_oi]

            x, y = extract_data(gene, peaks_oi)

            x = x.T
            y = y[:, None].T

            cors = np.corrcoef(x, y)[:-1, -1]
            cors[np.isnan(cors)] = 0
            peakscores = cors
            
            peak_gene_links_oi["cor"] = cors

            joined = joined_genes[gene_oi]

            cor = np.zeros(len(joined))
            joined["cor"] = 0.
            for peak_id, peak_gene_link in peak_gene_links_oi.iterrows():
                cor[np.arange(np.searchsorted(design["window_start"], peak_gene_link.relative_start), np.searchsorted(design["window_start"], peak_gene_link.relative_end))] = peak_gene_link["cor"]
            joined["cor"] = cor

            for w in ws:
                joined_oi = joined.copy()

                if method_name.endswith("linear"):
                    joined_oi["score"] = crispri.rolling_max(joined_oi["cor"].values, w)
                elif method_name.endswith("linear_absolute"):
                    joined_oi["score"] = crispri.rolling_max(np.abs(joined_oi["cor"].abs().values), w)
                elif method_name.endswith("linear_positive"):
                    joined_oi["score"] = crispri.rolling_max(np.where(joined_oi["cor"] > 0, joined_oi["cor"].values, 0), w)

                joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

                aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
                auroc = sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], joined_oi["score"])

                joined_oi["score_untied"] = joined_oi["score"] + np.random.normal(scale = 1e-3, size = len(joined_oi))
                cor = np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0, 1]
                spearman = scipy.stats.spearmanr(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0]
                
                joined_oi["significant_cor"] = joined_oi["cor"].abs() >= 0.05

                joined_oi["score_rank"] = joined_oi["score"].rank(ascending = False) / joined_oi.shape[0]

                allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, w = w))

                joined_cre = joined_oi.loc[joined_oi["significant_cor"]]
                
                genescores[method_name].append({
                    "gene": gene_oi,
                    "method": method_name,
                    "lfc_mean": joined_cre["HS_LS_logratio"].abs().mean(),
                    "n": joined_cre.shape[0],
                    "aupr":aupr,
                    "auroc":auroc,
                    "cor":cor,
                    "spearman":spearman,
                    "w":w,
                })

                slicescores[method_name].append(pd.DataFrame({
                    "gene": gene_oi,
                    "method": method_name,
                    "lfc": joined_cre["HS_LS_logratio"].abs(),
                    "slice":joined_cre.index,
                    "score": joined_cre["cor"].abs(),
                    "w":w,
                }))

# %%
method_name = "all"
genescores[method_name] = []
slicescores[method_name] = []
allslicescores[method_name] = []
for gene_oi in tqdm.tqdm(genes_oi):
    joined_oi = joined_genes[gene_oi]
    joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
    
    aupr = []
    auroc = []
    for i in range(100):
        aupr.append(sklearn.metrics.average_precision_score(joined_oi["significant_expression"], np.random.rand(joined_oi.shape[0])))
        auroc.append(sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], np.random.rand(joined_oi.shape[0])))
    aupr = np.mean(aupr)
    auroc = np.mean(auroc)

    joined_oi["score"] = np.random.rand(joined_oi.shape[0])
    joined_oi["score_rank"] = joined_oi["score"].rank(ascending = False) / joined_oi.shape[0]

    genescores[method_name].append({
        "gene": gene_oi,
        "method": method_name,
        "lfc_mean": joined_oi["HS_LS_logratio"].abs().mean(),
        "n": joined_chd.shape[0],
        "aupr":aupr,
        "auroc":auroc,
        "cor":np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score"] + np.random.normal(0, 1e-5, size = len(joined_oi)))[0, 1],
    })
    slicescores[method_name].append(pd.DataFrame({
        "gene": gene_oi,
        "method": method_name,
        "lfc": joined_oi["HS_LS_logratio"].abs(),
        "score":0.,
        "slice":joined_oi.index,
    }))

    allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, score = np.random.rand(joined_oi.shape[0]), w = 0))

# %% [markdown]
# ### All genes together

# %%
allslicescores_stacked = pd.concat([pd.concat(allslicescores[method_name]) for method_name in allslicescores.keys()], ignore_index=True)
slicescores_stacked = pd.concat([pd.concat(slicescores[method_name]) for method_name in slicescores.keys()], ignore_index=True)

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for method, plotdata in allslicescores_stacked.query("w == 20").groupby("method"):
    panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
    cors = []
    for gene, plotdata in plotdata.groupby("gene"):
        ax.set_title(method)
        ax.scatter(-plotdata["HS_LS_logratio"], plotdata["score"], s = 1)
        cors.append(np.corrcoef(-plotdata["HS_LS_logratio"], plotdata["score"] + np.random.normal(scale = 1e-5, size = len(plotdata)))[0, 1])
    ax.annotate(f"cor = {sum(cors)/len(cors):.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")

fig.plot()

# %%
eps = 1e-5

methodscores = []
for (method_name, w), slicescores_oi in allslicescores_stacked.groupby(["method", "w"]):
    aupr = sklearn.metrics.average_precision_score(slicescores_oi["significant_expression"], -slicescores_oi["score_rank"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    auroc = sklearn.metrics.roc_auc_score(slicescores_oi["significant_expression"], -slicescores_oi["score_rank"])

    methodscores.append({
        "method": method_name,
        "aupr": aupr,
        "auroc": auroc,
        "cor": np.corrcoef(slicescores_oi["HS_LS_logratio"].abs(), -slicescores_oi["score_rank"])[0, 1],
        "w":w
    })
methodscores = pd.DataFrame(methodscores)
methodscores["r2"] = methodscores["cor"]**2

# %%
methodscores.loc[(methodscores["method"] == "all") | (methodscores["w"] == 5)].style.bar()

# %%
prediction_methods = chdm.methods.prediction_methods
prediction_methods.loc["all"] = {
    "color":"grey",
    "label":"Random"
}
prediction_methods.loc["accessibility"] = {
    "color":"grey",
    "label":"Accessibility"
}
for method_name in ["v30", "v30/positive", "v30/negative", "v30/effect"]:
    prediction_methods.loc[method_name] = {
        "color":"blue",
        "label":"ChromatinHD " + (method_name.split("/")[1] if "/" in method_name else "")
    }
for peakcaller in peakcallers:
    prediction_methods.loc[peakcaller + "/linear"] = prediction_methods.loc[peakcaller + "/linear"]
    prediction_methods.loc[peakcaller + "/linear_absolute"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_absolute", "label"] += " abs"
    prediction_methods.loc[peakcaller + "/linear_positive"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_positive", "label"] += " pos"

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width = 0.2))

plotdata = methodscores.loc[(methodscores["w"] == 5) | (methodscores["method"] == "all")].copy()
methods = prediction_methods.loc[methodscores["method"].unique()]

panel, ax = fig.main.add_under(chd.grid.Panel((2, len(methods)*0.3)))
color = prediction_methods.reindex(plotdata["method"])["color"]
ax.barh(plotdata["method"], plotdata["aupr"], color = color)
ax.set_xlim(0., 1)
ax.set_xlabel("AUPRC")
ax.set_yticks(np.arange(len(plotdata["method"])))
ax.set_yticklabels(prediction_methods.reindex(plotdata["method"])["label"])

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["auroc"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("AUROC")
ax.set_title("Fulco et al. 2019")

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["cor"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("cor")

fig.plot()

# %%
fig, (ax_aupr, ax_auroc) = plt.subplots(1, 2, figsize = (5, 2))

eps = 1e-5

allslicescores_oi = allslicescores_stacked.loc[(allslicescores_stacked["w"] == 10) | (allslicescores_stacked["method"] == "all")]

for (method_name, w), slicescores_oi in allslicescores_oi.groupby(["method", "w"]):
    slicescores_oi["significant_expression"] = slicescores_oi["HS_LS_logratio"].abs() > lfc_cutoff

    curve = sklearn.metrics.precision_recall_curve(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    ax_aupr.plot(curve[1], curve[0], label = method_name, color = prediction_methods.loc[method_name, "color"])

    curve = sklearn.metrics.roc_curve(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    ax_auroc.plot(curve[0], curve[1], label = method_name, color = prediction_methods.loc[method_name, "color"])
    ax_auroc.plot([0, 1], [0, 1], color = "black", linestyle = "--")
ax_auroc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
