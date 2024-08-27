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
#     display_name: Python 3
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

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0)
window_width = window[1] - window[0]

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output() / "prediction_positional" / dataset_name / promoter_name / splitter / prediction_name
)

# %%
bed_folder = chd.get_output() / "bed" / "gm1282_structural_chipseq"

import mgzip
import pickle

files = pd.read_csv(bed_folder / "files.csv", index_col=0)
# files = files[files["target"].isin(["CTCF", "RAD21", "SPI1", "POLR2A"])]

values = {}
for accession in tqdm.tqdm(files.index):
    values[accession] = pickle.load(mgzip.open(str(bed_folder / f"values_{accession}.pickle"), "rb"))

# %% [markdown]
# # Check spot

# %%
means = {}
for accession, target in zip(files.index, files["target"]):
    means[target] = np.nanmean(np.stack(values[accession].values()), 0)

# %%
fig, ax = plt.subplots()
for target, mean in means.items():
    ax.plot(mean, label=target, alpha=0.5)
ax.legend()
ax.set_yscale("log")

# %%
sns.heatmap(np.log(pd.DataFrame(means)))

# %% [markdown]
# ## Merge values based on step size of windows
# %%
from collections import defaultdict

values_merged = defaultdict(dict)
step = 100
pad = (5000 // step) * 2
indices = np.arange(0, window_width, step)
for accession, values_accession in tqdm.tqdm(values.items()):
    x = np.stack(list(values[accession].values()))
    N = step

    y = []
    for i in indices:
        y.append(np.nanmean(x[:, i : i + step], 1))
    y = np.stack(y, 1)
    # y = np.exp(np.convolve(np.log(x + 0.1), np.ones(N) / N, mode="same"))
    y = np.pad(y, ((0, 0), (pad, pad)), constant_values=np.nan)
    y = pd.DataFrame(y, index=list([k[1] for k in values_accession.keys()]))
    values_merged[accession] = y


# %%
genes_oi = transcriptome.gene_id(["CD3E"])

# %%
scores_folder = prediction.path / "scoring" / "pairwindow_gene" / genes_oi[0]
window_design = pd.read_pickle(scores_folder / "design.pkl")
window_to_windowix = window_design["ix"]
assert window_design.index[1] - window_design.index[0] == step

# %%
# Calculate for each gene and window the amount of overlapping SNPs

genewindowscores = []
genewindowpads = []
randomgenewindowpads = []
foci = []

distance_cutoff = 1000

# genes_oi = transcriptome.var.index[:100]
genes_oi = transcriptome.var.index[:5000]
# genes_oi = transcriptome.var.index[:2000]
# genes_oi = transcriptome.gene_id(
#     ["BCL2", "CD74", "LYN", "TNFRSF13C", "PAX5", "IRF8", "IRF4"]
# )
# genes_oi = transcriptome.gene_id(["CD74"])
for gene in tqdm.tqdm(genes_oi):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    promoter = promoters.loc[gene]

    if interaction_file.exists():
        interaction = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
        interaction = interaction.rename(columns={0: "cor"})

        interaction["right"] = (interaction["window2"] > interaction["window1"]) * interaction["cor"] > 0
        interaction["abscor"] = np.clip(interaction["cor"].abs(), 0, 1)
        windowscores = (
            interaction.query("distance > 1000")
            .sort_values("cor", ascending=False)
            .groupby("window1")[["cor", "deltacor1", "effect1", "right", "abscor"]]
            .mean(numeric_only=True)
        )
        windowscores.index.name = "window"
        windowscores["ix"] = window_to_windowix.loc[windowscores.index].values

        foci_gene = []
        for window, windowscore in windowscores.sort_values("deltacor1").iterrows():
            if windowscore["deltacor1"] > -1e-4:
                break
            add = True
            for focus in foci_gene:
                if abs(window - focus["window"]) < distance_cutoff:
                    add = False
                    break
            if add:
                focus = {"window": window, **windowscore.to_dict()}
                foci_gene.append(focus)
        foci_gene = pd.DataFrame(foci_gene)

        if len(foci_gene) == 0:
            continue

        foci_gene["ix"] = window_to_windowix.loc[foci_gene["window"]].values

        for accession in files.index:
            value = values_merged[accession].loc[gene]
            score = windowscores.copy()
            score["value"] = value[windowscores["ix"].values]

            foci.append(foci_gene.assign(gene=gene, accession=accession).reset_index())

            for a, b in zip(
                foci_gene["ix"].values,
                foci_gene["ix"].values + pad * 2,
            ):
                genewindowpads.append(value[a:b])

# genewindowscores = pd.concat(genewindowscores, sort=False, ignore_index=True)
genewindowpads = np.stack(genewindowpads)

# %%
if isinstance(foci, list):
    foci = pd.concat(foci, sort=False, ignore_index=True)
foci["ix"] = np.arange(len(foci))


# %%
foci["ix"] = np.arange(len(foci))
accession_oi = files.index[files["target"] == "IRF4"][0]

fig, ax = plt.subplots()
x = np.arange(-pad * step, pad * step, step)

foci_oi = foci.loc[
    (foci["accession"] == accession_oi)
    # & (foci["right"] < 0.5)
    # & (foci["window"] > -0)
    & (foci["deltacor1"] < foci["deltacor1"].mean())
    & True
]

pads_oi = genewindowpads[foci_oi["ix"]]

ax.plot(x, np.nanmean(pads_oi, 0))
ax.axvline(0)

foci_oi2 = foci.loc[
    (foci["accession"] == accession_oi)
    # & (foci["window"] > 0)
    # & (foci["window"] > 0)
    & (foci["deltacor1"] > foci["deltacor1"].mean())
    # & (genewindowscores["right"] > 0.5)
    & True
]

pads_oi2 = genewindowpads[foci_oi2["ix"]]

ax.plot(x, np.nanmean(pads_oi2, 0))

fig, ax = plt.subplots()
ax.scatter(x, np.nanmean(pads_oi, 0) / np.nanmean(pads_oi2, 0))
ax.set_xlim(-2500, 2500)


# %%
foci["ix"] = np.arange(len(foci))
accession_oi = files.index[files["target"] == "SPI1"][0]

# %%
import scanpy as sc

transcriptome.adata.obs["oi"] = pd.Categorical(
    np.array(["noi", "oi"])[transcriptome.adata.obs["celltype"].isin(["naive B", "memory B"]).values.astype(int)]
)
sc.tl.rank_genes_groups(transcriptome.adata, groupby="oi")
diffexp = (
    sc.get.rank_genes_groups_df(
        transcriptome.adata,
        # group="CD14+ Monocytes",
        # group="naive B",
        # group="memory B",
        group="oi",
    )
    .rename(columns={"names": "gene"})
    .assign(symbol=lambda x: transcriptome.var.loc[x["gene"], "symbol"].values)
    .set_index("gene")
)
genes_diffexp = diffexp.query("logfoldchanges > 0.1").index

# %%

ratios = {}
for accession_oi in tqdm.tqdm(files.index):
    ratios_accession = []
    for gene, foci_gene in tqdm.tqdm(
        (
            foci
            # .query("window < -1000")
            # .query("window > 1000")
            .query("accession == @accession_oi").groupby("gene")
        ),
        total=len(foci["gene"].unique()),
    ):
        if len(foci_gene) <= 5:
            continue
        # if gene not in genes_diffexp:
        #     continue
        foci_oi = foci_gene.loc[
            True
            & (foci_gene["abscor"] > foci_gene["abscor"].median())
            # & (foci_gene["deltacor1"] < foci_gene["deltacor1"].median())
            & True
        ]

        pads_oi = genewindowpads[foci_oi["ix"]]

        foci_oi2 = foci_gene.loc[
            True
            # & (foci_gene["deltacor1"] >= foci_gene["deltacor1"].median())
            & (foci_gene["abscor"] < foci_gene["abscor"].median())
            & True
        ]

        pads_oi2 = genewindowpads[foci_oi2["ix"]]

        ratio = np.nanmean(pads_oi + 1e-8, 0) / np.nanmean(pads_oi2 + 1e-8, 0)
        ratios_accession.append(ratio)
    ratios[accession_oi] = ratios_accession

# %%
len(ratios_accession)


# %%
def smooth_spline_fit(x, y, x_smooth):
    import rpy2.robjects as robjects

    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)

    r_smooth_spline = robjects.r["smooth.spline"]
    spline1 = r_smooth_spline(x=r_x, y=r_y)
    ySpline = np.array(robjects.r["predict"](spline1, robjects.FloatVector(x_smooth)).rx2("y"))

    return ySpline


# %%
fig, ax = plt.subplots()
for accession_oi, ratios_accession in ratios.items():
    y = np.exp(np.nanmean(np.log(np.stack(ratios_accession)), 0))
    ax.plot(
        x,
        y,
        label=files.loc[accession_oi, "target"],
        # s=1,
    )
    # ax.plot(x, smooth_spline_fit(x, y, x))
ax.axhline(1, dashes=(2, 2), color="black")
ax.set_yscale("log")
ax.set_xlim(-5000, 5000)
ax.legend()

# %%
fig, ax = plt.subplots(figsize=(4, 2))

enrichments = {}
for accession_oi, ratios_accession in ratios.items():
    enrichments[accession_oi] = np.exp(np.nanmean(np.log(np.stack(ratios_accession)), 0))
enrichments = pd.DataFrame(enrichments).T
enrichments.index = files.loc[enrichments.index, "target"]
norm = mpl.colors.LogNorm(vmin=0.5, vmax=2)
cmap = mpl.cm.PiYG_r
ax.matshow(enrichments, cmap=cmap, norm=norm, aspect="auto")
ax.set_yticks(np.arange(len(enrichments.index)))
ax.set_yticklabels(enrichments.index)
desired_x_ticks = np.array([-5000, -2500, 0, 2500, 5000]).astype(int)
ax.set_xticks(np.interp(desired_x_ticks, x, np.arange(len(x))).astype(int))
ax.set_xlabel("Distance from predictive region (kb)")
ax.xaxis.set_label_position("top")
ax.set_xticklabels([f"{str(x/1000).rstrip('0').rstrip('.')}" for x in desired_x_ticks])
ax.set_xlim(*np.interp([-5000, 5000], x, np.arange(len(x))).astype(int))
ax.axvline(np.interp(0, x, np.arange(len(x))), color="black", lw=1, dashes=(2, 2))
# ax.set_xticklabels(x[ax.get_xticks().astype(int)])

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), extend="both")
cbar.set_ticks([0.5, 1, 2])
cbar.set_ticklabels(["½", "1", "2"])
cbar.set_ticklabels([], minor=True)
cbar.set_label(
    "Ratio ChIP-seq signal\nstrongly co-predictive (cor > mean(cor))\nversus weakly co-predictive (cor ≤ mean(cor))\nenhancers",
    rotation=0,
    ha="left",
    va="center",
)

manuscript.save_figure(fig, "6", "loop_enrichment")
# %%
