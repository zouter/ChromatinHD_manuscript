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
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %% [markdown]
# ## Load ChromatinHD data

# %%
# dataset_name = "pbmc10k"
dataset_name = "hspc"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

regions_name = "100k100k"
# regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %% [markdown]
# ## Load CRISPRi data

# %%
folder = chd.get_output() / "data" / "crispri" / "fulco_2016"
data = pd.read_table(folder / "data.tsv", sep="\t")

# %%
folder = chd.get_output() / "data" / "crispri" / "fulco_2019"
data = pd.read_table(folder / "data.tsv", sep="\t")

# %%
folder = chd.get_output() / "data" / "crispri" / "fulco_2019"
folder2 = chd.get_output() / "data" / "crispri" / "fulco_2019"
data = pd.concat([
    pd.read_table(folder / "data.tsv", sep="\t"),
    pd.read_table(folder2 / "data.tsv", sep="\t")
])

# %%
folder = chd.get_output() / "data" / "crispri" / "gasperini_2019"
data = pd.read_table(folder / "data.tsv", sep="\t")

# %%
folder = chd.get_output() / "data" / "crispri" / "nasser_2021"
data = pd.read_table(folder / "data.tsv", sep="\t")

# %%
folder = chd.get_output() / "data" / "crispri" / "simeonov_2017"
data = pd.read_table(folder / "data.tsv", sep="\t")

# %% [markdown]
# ### Filter

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)
# genes_oi = genes_oi[(transcriptome.var.loc[genes_oi, "dispersions_norm"] > -0.5).values]
# genes_oi = transcriptome.gene_id(["GATA1", "H1FX", "KLF1", "CALR", "HDAC6", "PQBP1", "NUCB1", "FTL"])
print(len(genes_oi))

data = data.loc[data["Gene"].isin(symbols_oi)].copy()
data["gene"] = transcriptome.gene_id(data["Gene"]).values

# %% [markdown]
# ## Mean vs variance

# %%
probs_mean_bins = pd.DataFrame(
    {"cut_exp":[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., 4, 8, np.inf]}
)
probs_mean_bins["cut"] = np.log(probs_mean_bins["cut_exp"])
probs_mean_bins["label"] = ["<" + str(probs_mean_bins["cut_exp"][0])] + ["≥" + str(x) for x in probs_mean_bins["cut_exp"].astype(str)[:-1]]

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": list(np.log(np.round(np.logspace(np.log(1.25), np.log(2.5), 7, base = np.e), 5))) + [np.inf]}
)
clusterprobs_diff_bins["cut_exp"] = np.exp(clusterprobs_diff_bins["cut"])
clusterprobs_diff_bins["label"] = ["<" + str(np.round(clusterprobs_diff_bins["cut_exp"][0], 1))] + ["≥" + str(x) for x in np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1]]
clusterprobs_diff_bins["do_label"] = True
clusterprobs_diff_bins

# %%
regions = fragments.regions

# %%
step = 50
design = pd.DataFrame({
    "start": np.arange(fragments.regions.window[0], fragments.regions.window[1], step),
})
design["end"] = design["start"] + step
design["mid"] = (design["start"] + design["end"]) / 2

# %%
scores = []

for region_id in tqdm.tqdm(genes_oi):
    region = regions.coordinates.loc[region_id]

    # calculate crispri metrics
    data_oi = data.loc[(data["chrom"] == region["chrom"]) & (data["gene"] == region_id)].copy()
    data_oi["start"] = data_oi["start"].astype(int)
    data_oi["end"] = data_oi["end"].astype(int)

    data_oi = data_oi.loc[data_oi["start"] > region["start"]]
    data_oi = data_oi.loc[data_oi["end"] < region["end"]]

    data_centered = chd.plot.genome.genes.center(data_oi, region)
    data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

    if len(data_centered) == 0:
        continue

    data_centered["bin"] = design["mid"].values[np.digitize(data_centered["mid"], design["end"])]
    # data_centered["bin"] = design["window_mid"].values[np.digitize(data_centered["mid"], design["window_mid"]) - 1]
    data_binned = data_centered.groupby("bin").mean(numeric_only=True)
    data_binned = data_binned.reindex(design["mid"])

    data_binned["crispr_significant"] = data_binned["HS_LS_logratio"] < np.log(0.8)

    # calculate chd metrics
    y = regionpositional.get_interpolated(region_id, desired_x = design["mid"].values)

    ymean = y.mean(0)
    z = y - ymean
    zmax = z.std(0)

    # add chd diff metrics to data binned
    data_binned["clusterprobs_diff"] = zmax
    data_binned["clusterprobs_diff_bin"] = np.digitize(data_binned["clusterprobs_diff"], clusterprobs_diff_bins["cut"])
    data_binned["probs_mean"] = ymean
    data_binned["probs_mean_bin"] = np.digitize(data_binned["probs_mean"], probs_mean_bins["cut"])

    scores.append(data_binned.assign(region_id=region_id))
scores = pd.concat(scores)

# %%
import chromatinhd_manuscript.crispri

# %%
ratio_unstacked = scores.dropna().groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["HS_LS_logratio"].mean().unstack().T
size_unstacked = scores.dropna().groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["HS_LS_logratio"].size().unstack().T
significant_unstacked = scores.dropna().groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["crispr_significant"].mean().unstack().T
ratio_unstacked[size_unstacked < 10] = np.nan
significant_unstacked[size_unstacked < 10] = np.nan

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG_r
odds_max = 1.5
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(ratio_unstacked, cmap=cmap, norm=norm)
ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"], rotation = 90)

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="CRISPRi score", extend="both", ticks=[np.log(1/odds_max), 0, np.log(odds_max)])
# cbar.ax.set_yticklabels([f"1/{odds_max}", "1", f"{odds_max}"])

ax.set_yticklabels([])
ax.set_ylabel("")
ax.set_xlabel("")
ax.tick_params(top = False)
ax.xaxis.set_ticks_position('bottom')
sns.despine(fig, ax)

# fig.save_figure()

# manuscript.save_figure(fig, "4", "mean_vs_variance_crispr")

# %%
plotdata = np.log(
        scores.dropna().groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["crispr_significant"].mean().unstack().T
        / scores.dropna()["crispr_significant"].mean()
    )
plotdata[size_unstacked < 5] = np.nan

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG
odds_max = 4.
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(plotdata, cmap=cmap, norm=norm)

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="CRISPRi score", extend="both", ticks=[np.log(1/odds_max), 0, np.log(odds_max)])
# cbar.ax.set_yticklabels([f"1/{odds_max}", "1", f"{odds_max}"])

ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(1, len(probs_mean_bins))-0.5)
ax.set_yticklabels(probs_mean_bins["cut_exp"][:-1])

# ax.set_xlabel("Accessibility\nfold change")

# ax.set_ylabel("")
# ax.set_yticks(np.arange(1, len(probs_mean_bins))-0.5)
# ax.set_yticklabels([])

ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, len(clusterprobs_diff_bins)) - 0.5, minor=True)
ax.set_xticklabels(np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1], minor=True, rotation = 90)
ax.set_xticks([])
ax.set_xticklabels([])

sns.despine(fig, ax)

# fig.save_figure()

manuscript.save_figure(fig, "4", "mean_vs_variance_crispr")

# %%
fig_colorbar = plt.figure(figsize=(2.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=norm, cmap=cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal", ticks=[np.log(1/odds_max), 0, np.log(odds_max)], extend = "both")
ax_colorbar.set_xticklabels([f"1/{odds_max:.0f}", "1", f"{odds_max:.0f}"])
colorbar.set_label("Odds ratio")
manuscript.save_figure(fig_colorbar, "4", "colorbar_crispr")

# %%
scores["missing"] = scores["HS_LS_logratio"].isnull()
sns.heatmap(scores.groupby(["clusterprobs_diff_bin", "probs_mean_bin"])["missing"].mean().unstack().T, vmax = 1, vmin = 0)

# %%
sns.heatmap(np.log(scores.groupby(["clusterprobs_diff_bin", "probs_mean_bin"]).size().unstack().T))

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
panel, ax = fig.main.add_under(polyptich.grid.Panel((2, 2)))

ax.scatter(scores["clusterprobs_diff"], scores["HS_LS_logratio"], s = 1)
for clusterprobs_diff_bin in clusterprobs_diff_bins["cut"]:
    ax.axvline(clusterprobs_diff_bin, color = "black", alpha = 0.5)
ax.set_xlabel("probs_diff")

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(scores["probs_mean"], scores["HS_LS_logratio"], s = 1)
for probs_mean_bin in probs_mean_bins["cut"]:
    ax.axvline(probs_mean_bin, color = "black", alpha = 0.5)
ax.set_xlabel("probs_mean")

fig.plot()

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())
panel, ax = fig.main.add_under(polyptich.grid.Panel((2, 2)))

ax.scatter(data_binned["clusterprobs_diff"], data_binned["HS_LS_logratio"], s = 1)
for clusterprobs_diff_bin in clusterprobs_diff_bins["cut"]:
    ax.axvline(clusterprobs_diff_bin, color = "black", alpha = 0.5)
ax.set_xlabel("probs_diff")
lm = sns.regplot(x = data_binned["clusterprobs_diff"], y = data_binned["HS_LS_logratio"], scatter = False, ax = ax)

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(data_binned["probs_mean"], data_binned["HS_LS_logratio"], s = 1)
for probs_mean_bin in probs_mean_bins["cut"]:
    ax.axvline(probs_mean_bin, color = "black", alpha = 0.5)
ax.set_xlabel("probs_mean")
lm = sns.regplot(x = data_binned["probs_mean"], y = data_binned["HS_LS_logratio"], scatter = False, ax = ax)

fig.plot()

# %% [markdown]
# ## ABCD

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)

# %%
# !wget https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6886585/bin/NIHMS1541544-supplement-2.xlsx -O {chd.get_output() / "NIHMS1541544-supplement-2.xlsx"}

# %%
data_cre = pd.read_excel(chd.get_output() / "NIHMS1541544-supplement-2.xlsx", sheet_name="Supplementary Table 6a", skiprows=1)

# %%
data_cre["HS_LS_logratio"] = np.clip(data_cre["Fraction change in gene expr"], -np.inf, 0)

# %%
import scipy.interpolate
def interpolate_dynamics(dynamics, start, end, measure = "mean"):
    start = min(max(int(start), dynamics.index.min()), dynamics.index.max())
    end = max(min(int(end), dynamics.index.max()), dynamics.index.min())

    if start == end:
        return np.nan
        
    if measure == "mean":
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)(np.arange(start, end)).mean()
    elif measure == "max":
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)(np.arange(start, end)).max()


# %%
import chromatinhd.data.peakcounts

# %%
scores = []
for gene in genes_oi:
    symbol = transcriptome.var["symbol"][gene]
    data_oi = data_cre.query("Gene == @symbol").copy()
    data_oi = chd.data.peakcounts.plot.center_peaks(
        data_oi, fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
    )

    dynamics = regionpositional.probs[gene].std("cluster").to_pandas()

    dynamics_cres = []
    for i, row in data_oi.iterrows():
        dynamics_cres.append(interpolate_dynamics(dynamics, row["start"], row["end"]))
    data_oi["dynamics"] = dynamics_cres
    # data_oi["dynamics"] = np.log(dynamics_cres)
    # data_oi["dynamics"] = dynamics_cres
    data_oi = data_oi.dropna(subset=["dynamics"])
    data_oi["dynamics"] = data_oi["dynamics"]# / data_oi["dynamics"].max()

    cor_abc = np.corrcoef(data_oi["ABC Score"], data_oi["HS_LS_logratio"])[0, 1]
    # cor_abc = spearman(data_oi["ABC Score"], data_oi["HS_LS_logratio"])
    cor_dynamics = np.corrcoef(data_oi["dynamics"], data_oi["HS_LS_logratio"])[0, 1]
    # cor_dynamics = spearman(data_oi["dynamics"], data_oi["HS_LS_logratio"])

    data_oi["ABCD Score"] = data_oi["ABC Score"] * data_oi["dynamics"]
    cor_abcd = np.corrcoef(data_oi["ABCD Score"], data_oi["HS_LS_logratio"])[0, 1]
    # cor_abcd = spearman(data_oi["ABCD Score"], data_oi["HS_LS_logratio"])

    data_cre.loc[data_oi.index, "dynamics"] = data_oi["dynamics"]
    data_cre.loc[data_oi.index, "ABCD Score"] = data_oi["ABCD Score"]

    scores.append(
        {
            "cor_abc": cor_abc,
            "cor_dynamics": cor_dynamics,
            "cor_abcd": cor_abcd,
            "gene": gene,
            "symbol": symbol,
        }
    )
scores = pd.DataFrame(scores)
data_abcd = data_cre.dropna(subset=["ABCD Score"]).copy()

# %%
data_abcd["dynamics"].plot.hist()

# %%
data_cre["distance"] = (data_cre["start"] - data_cre["Gene TSS"]).abs()

model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_cre[["ABC Score"]], data_cre["Significant"])
prediction = model.predict_proba(data_cre[["ABC Score"]])[:, 1]
print("ABC Score= ", sklearn.metrics.average_precision_score(data_cre["Significant"], prediction))

model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_cre[["distance"]], data_cre["Significant"])
prediction = model.predict_proba(data_cre[["distance"]])[:, 1]
print("pure distance= ", sklearn.metrics.average_precision_score(data_cre["Significant"], prediction))

# %%
data_abcd_significant = data_abcd.loc[data_abcd["Significant"]]

data_abcd["significant"] = data_abcd["Adjusted p-value"] < 0.05
data_abcd_significant = data_abcd.loc[data_abcd["significant"]]

# %%
sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABC Score"]), sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABCD Score"])

# %%
import sklearn.ensemble
import sklearn.model_selection
# model = sklearn.ensemble.RandomForestClassifier().fit(data_abcd[["ABC Score", "dynamics"]], data_abcd["significant"])
model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_abcd[["ABC Score", "dynamics"]], data_abcd["significant"])
prediction = model.predict_proba(data_abcd[["ABC Score", "dynamics"]])[:, 1]
print(sklearn.metrics.roc_auc_score(data_abcd["significant"], prediction), sklearn.metrics.average_precision_score(data_abcd["significant"], prediction))

model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_abcd[["ABC Score"]], data_abcd["significant"])
prediction = model.predict_proba(data_abcd[["ABC Score"]])[:, 1]
print(sklearn.metrics.roc_auc_score(data_abcd["significant"], prediction), sklearn.metrics.average_precision_score(data_abcd["significant"], prediction))

model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_abcd[["ABCD Score"]], data_abcd["significant"])
prediction = model.predict_proba(data_abcd[["ABCD Score"]])[:, 1]
print(sklearn.metrics.roc_auc_score(data_abcd["significant"], prediction), sklearn.metrics.average_precision_score(data_abcd["significant"], prediction))

# %%
spearman(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["ABC Score"]), spearman(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["ABCD Score"]), spearman(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["dynamics"])

# %%
import sklearn.linear_model

# %%
fig, ax = plt.subplots()

model = sklearn.linear_model.LinearRegression().fit(data_abcd_significant[["ABC Score", "dynamics"]], data_abcd_significant["HS_LS_logratio"])
prediction = model.predict(data_abcd_significant[["ABC Score", "dynamics"]])
# print(spearman(data_abcd_significant["HS_LS_logratio"], prediction))
print(np.corrcoef(data_abcd_significant["HS_LS_logratio"], prediction)[0, 1])

ax.scatter(data_abcd_significant["HS_LS_logratio"], prediction)

model = sklearn.linear_model.LinearRegression().fit(data_abcd_significant[["ABC Score"]], data_abcd_significant["HS_LS_logratio"])
prediction = model.predict(data_abcd_significant[["ABC Score"]])
# print(spearman(data_abcd_significant["HS_LS_logratio"], prediction))
print(np.corrcoef(data_abcd_significant["HS_LS_logratio"], prediction)[0, 1])

ax.scatter(data_abcd_significant["HS_LS_logratio"], prediction)

model = sklearn.linear_model.LinearRegression().fit(data_abcd_significant[["ABCD Score"]], data_abcd_significant["HS_LS_logratio"])
prediction = model.predict(data_abcd_significant[["ABCD Score"]])
# print(spearman(data_abcd_significant["HS_LS_logratio"], prediction))
print(np.corrcoef(data_abcd_significant["HS_LS_logratio"], prediction)[0, 1])

ax.scatter(data_abcd_significant["HS_LS_logratio"], prediction)

# %%
model = sklearn.linear_model.LinearRegression().fit(data_abcd[["ABC Score"]], data_abcd["HS_LS_logratio"])
prediction = model.predict(data_abcd[["ABC Score"]])
print(spearman(data_abcd["HS_LS_logratio"], prediction))

fig, ax = plt.subplots()
ax.scatter(data_abcd["HS_LS_logratio"], prediction)

# %%
model = sklearn.linear_model.LinearRegression().fit(data_abcd[["ABC Score", "dynamics"]], data_abcd["HS_LS_logratio"])
prediction = model.predict(data_abcd[["ABC Score", "dynamics"]])
print(spearman(data_abcd["HS_LS_logratio"], prediction))

fig, ax = plt.subplots()
ax.scatter(data_abcd["HS_LS_logratio"], prediction)

# %%
scores.mean()

# %%
fig, ax = plt.subplots()
ax.scatter(scores["cor_abc"], scores["cor_dynamics"])


# %%
scores.style.bar()

# %%
spearman(data_oi["ABC Score"] * data_oi["dynamics"], data_oi["HS_LS_logratio"])

# %%
np.corrcoef(data_cre["ABC Score"], data_cre["HS_LS_logratio"])

# %%
np.corrcoef(data_abcd["ABC Score"], data_abcd["HS_LS_logratio"])

# %%
np.corrcoef(data_abcd["ABCD Score"], data_abcd["HS_LS_logratio"])

# %%
fig, ax = plt.subplots()
ax.scatter(np.log(data_abcd["ABC Score"]), data_abcd["HS_LS_logratio"], s = 1)

# %%
np.corrcoef(np.log(data_abcd["ABC Score"] + 1e-1), data_abcd["HS_LS_logratio"])

# %%
np.corrcoef(np.log(data_abcd["ABCD Score"] + 1e-1), data_abcd["HS_LS_logratio"])

# %%
np.corrcoef(data_abcd["dynamics"], data_abcd["HS_LS_logratio"])

# %%
fig, ax = plt.subplots()
ax.scatter(np.log(data_abcd["ABCD Score"]), data_abcd["HS_LS_logratio"], s = 1)

# %% [markdown]
# ### ABCD 2

# %%
import pyranges
import pybedtools

# %%
suptable_path = chd.get_output() / "NIHMS1541544-supplement-2.xlsx"
if not suptable_path.exists():
    # !wget https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6886585/bin/NIHMS1541544-supplement-2.xlsx -O {suptable_path}

# %%
data_cre = pd.read_excel(suptable_path, sheet_name="Supplementary Table 6a", skiprows=1)

# %%
data_cre["HS_LS_logratio"] = np.clip(data_cre["Fraction change in gene expr"], -np.inf, 0)

# %%
pr = pybedtools.BedTool.from_dataframe(
    fragments.regions.coordinates.query("start > 0").reset_index()[["chrom", "start", "end", "gene"]]
)

# %%
pr_cre =  pybedtools.BedTool.from_dataframe(
    data_cre.reset_index()[["chr", "start", "end", "index"]]
)

# %%
# intersection.to_dataframe().drop(columns = ["start", "end"])

# %%
intersection = pr.intersect(pr_cre, wa = True, wb = True,F = 1.0)
intersected = intersection.to_dataframe().drop(columns = ["start", "end"]).rename(columns = {"thickEnd": "index", "name":"gene", "strand":"start", "thickStart":"end"})
# intersected = intersected.groupby("index").first()

# %%
import chromatinhd.data.peakcounts
def spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]


# %%
intersected = chd.data.peakcounts.plot.center_multiple_peaks(
    intersected, fragments.regions.coordinates
)


# %%
# clusters = ["B-cell precursors", "GMP", "HSPC", "Erythrocyte precursors", "Erythroblast"]
# clusters = clustering.var.index
# clusters = clustering.var.index[-5:]
clusters = ["Megakaryocyte-erythrocyte gradient", "Megakaryocyte", "Unknown 1", "Unknown 2"]

assert all([x in clustering.var.index for x in clusters])

# %%
dynamics_genes = {gene:regionpositional.probs[gene].sel(cluster = clusters).std("cluster").to_pandas() for gene in intersected["gene"].unique()}
means_genes = {gene:regionpositional.probs[gene].sel(cluster = clusters).mean("cluster").to_pandas() for gene in intersected["gene"].unique()}

# %%
import scipy.interpolate
def interpolate_means(means, start, end):
    start = min(max(int(start), means.index.min()), means.index.max())
    end = max(min(int(end), means.index.max()), means.index.min())

    if start == end:
        return np.nan
        
    return scipy.interpolate.interp1d(means.index, means.values)(np.arange(start, end)).mean()

def interpolate_dynamics(dynamics, start, end, measure = "mean", means = None):
    start = min(max(int(start), dynamics.index.min()), dynamics.index.max())
    end = max(min(int(end), dynamics.index.max()), dynamics.index.min())

    if start == end:
        return np.nan
        
    if measure == "mean":
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)(np.arange(start, end)).mean()
    elif measure == "max":
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)(np.arange(start, end)).max()
    elif measure == "median":
        return np.median(scipy.interpolate.interp1d(dynamics.index, dynamics.values)(np.arange(start, end)))
    elif measure == "mid":
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)((start + end) / 2)
    elif measure == "center":
        mid = (start + end) / 2
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)(np.linspace(mid - 10, mid + 10, 100)).mean()
    elif measure == "top":
        means_oi = means.loc[(means.index >= start) & (means.index <= end)]
        mid = means_oi.idxmax()
        return scipy.interpolate.interp1d(dynamics.index, dynamics.values)(mid)


# %%
dynamics_cres = []
means_cres = []
for _, row in intersected.iterrows():
    dynamics = dynamics_genes[row["gene"]]
    means = means_genes[row["gene"]]
    means_cres.append(interpolate_means(means, row["start"], row["end"]))
    dynamics_cres.append(interpolate_dynamics(dynamics, row["start"], row["end"], measure = "center", means = means))
data_cre.loc[intersected["index"], "dynamics"] = dynamics_cres
data_cre.loc[intersected["index"], "means"] = means_cres

# %%
np.corrcoef(data_abcd["H3K27ac (RPM)"], data_abcd["means"])[0, 1], np.corrcoef(data_abcd["DHS (RPM)"], data_abcd["means"])[0, 1]

# %%
data_abcd["dynamics"].plot.hist()

# %%
data_abcd = data_cre.dropna(subset=["dynamics"]).copy()
# data_abcd.loc[data_abcd["means"] < -2, "dynamics"] = 0.5
# data_abcd = data_abcd.loc[data_abcd["means"] > -2]
data_abcd["ABCD Score"] = data_abcd["ABC Score"] * data_abcd["dynamics"]

# %%
import sklearn.metrics

# %%
for level in [None, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1]:
    if level is not None:
        data_abcd["significant"] = (data_abcd["Adjusted p-value"] < level) & (data_abcd["HS_LS_logratio"] < -0.1)
    else:
        data_abcd["significant"] = data_abcd["Significant"]
    print(
        level,
        sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABCD Score"]),
        sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABC Score"]) - sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABCD Score"])
    )

# %%
fig, ax = plt.subplots(figsize = (3, 3))
level = 0.05
data_abcd["significant"] = (data_abcd["Adjusted p-value"] < level) & (data_abcd["HS_LS_logratio"] < -0.0)
# data_abcd["significant"] = data_abcd["Significant"]
# data_abcd["significant"] = data_abcd["Fraction change in gene expr"] < -0.1
curve = sklearn.metrics.precision_recall_curve(data_abcd["significant"], data_abcd["ABC Score"])
ax.plot(curve[1], curve[0]) 
aupr = sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABC Score"])
ax.annotate(f"AUPR = {aupr:.2f}", (0.5, 0.5), xycoords = "axes fraction", ha = "center", va = "center")
curve = sklearn.metrics.precision_recall_curve(data_abcd["significant"], data_abcd["ABCD Score"])
ax.plot(curve[1], curve[0])
aupr = sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABCD Score"])
ax.annotate(f"AUPR = {aupr:.2f}", (0.5, 0.4), xycoords = "axes fraction", ha = "center", va = "center")

# %%
data_abcd_significant = data_abcd.loc[data_abcd["Significant"]]

data_abcd["significant"] = data_abcd["Significant"]
data_abcd["significant"] = data_abcd["Adjusted p-value"] < 0.05
data_abcd_significant = data_abcd.loc[data_abcd["significant"]]

# %%
np.corrcoef(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["ABC Score"])[0, 1], np.corrcoef(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["ABCD Score"])[0, 1]

# %%
spearman(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["ABC Score"]), spearman(data_abcd_significant["HS_LS_logratio"], data_abcd_significant["ABCD Score"])

# %%
sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABC Score"]), sklearn.metrics.average_precision_score(data_abcd["significant"], data_abcd["ABCD Score"])

# %%
sklearn.metrics.roc_auc_score(data_abcd["significant"], data_abcd["ABC Score"]), sklearn.metrics.roc_auc_score(data_abcd["significant"], data_abcd["ABCD Score"])

# %%
import sklearn.ensemble
import sklearn.model_selection
# model = sklearn.ensemble.RandomForestClassifier().fit(data_abcd[["ABC Score", "dynamics"]], data_abcd["significant"])
model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_abcd[["ABC Score", "dynamics"]], data_abcd["significant"])
prediction = model.predict_proba(data_abcd[["ABC Score", "dynamics"]])[:, 1]
print(sklearn.metrics.roc_auc_score(data_abcd["significant"], prediction), sklearn.metrics.average_precision_score(data_abcd["significant"], prediction))

model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_abcd[["ABC Score"]], data_abcd["significant"])
prediction = model.predict_proba(data_abcd[["ABC Score"]])[:, 1]
print(sklearn.metrics.roc_auc_score(data_abcd["significant"], prediction), sklearn.metrics.average_precision_score(data_abcd["significant"], prediction))

model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(data_abcd[["ABCD Score"]], data_abcd["significant"])
prediction = model.predict_proba(data_abcd[["ABCD Score"]])[:, 1]
print(sklearn.metrics.roc_auc_score(data_abcd["significant"], prediction), sklearn.metrics.average_precision_score(data_abcd["significant"], prediction))

# %%
(data_cre["end"] - data_cre["start"]).value_counts().sort_values(ascending = False).head(10)
