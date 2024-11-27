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
import pickle

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_method_combinations as design_methods,
)
from chromatinhd_manuscript.diff_params import params
from chromatinhd_manuscript.designs_qtl import design as design_qtl


# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
from chromatinhd_manuscript.designs_diff import dataset_latent_method_combinations as design_chd
design_chd = design_chd.query("method == 'v31'").query("splitter == '5x1'")
from chromatinhd_manuscript.designs_diff import dataset_latent_peakcaller_diffexp_combinations as design_diffexp
from chromatinhd_manuscript.designs_qtl import design as design_qtl

design = pd.concat([
    design_chd.rename(columns = {"method":"reference_method", "splitter":"reference_splitter"}).merge(design_chd),
    design_chd.rename(columns = {"method":"reference_method", "splitter":"reference_splitter"}).merge(design_diffexp)
], ignore_index = True).merge(design_qtl)

design = design.query("regions == '100k100k'")

design["method"] = [f"{row.peakcaller}/{row.diffexp}" if not pd.isnull(row.peakcaller) else "chd" for _, row in design.iterrows()]

design.shape

# %%
design = design.query("dataset != 'pbmc20k'")

# %%
scores = {}
for design_ix, design_row in tqdm.tqdm(design.iterrows(), total=len(design)):
    dataset_name = design_row["dataset"]
    regions_name = design_row["regions"]
    peakcaller = design_row["peakcaller"]
    diffexp = design_row["diffexp"]
    motifscan_name = design_row["motifscan"]

    if pd.isnull(peakcaller):
        enrichment_folder = (
            chd.get_output()
            / "diff"
            / dataset_name
            / regions_name
            / "5x1"
            / "v31"
            / "scoring"
            / "regionpositional"
            / "top"
            / "-1-1.5"
            / "enrichment"
            / motifscan_name
        )
    else:
        enrichment_folder = (
            chd.get_output()
            / "diff"
            / dataset_name
            / regions_name
            / "5x1"
            / "v31"
            / "scoring"
            / "regionpositional"
            / "top"
            / "-1-1.5"
            / "enrichment"
            / motifscan_name
            / peakcaller
            / diffexp
        )

    try:
        score = pickle.load((enrichment_folder / "scores.pkl").open("rb"))
        scores[design_ix] = score
    except FileNotFoundError:
        continue
scores = pd.DataFrame.from_dict(scores, orient="index")
scores.index.name = "design_ix"

# %%
methods = design.groupby(["method", "peakcaller", "diffexp"], dropna = False).first().index.to_frame().set_index("method")
methods["color"] = ["black" if pd.isnull(row.peakcaller) else "red" for _, row in methods.iterrows()]

settings = design.groupby(["dataset", "regions", "motifscan"]).first().index.to_frame()
settings["ix"] = np.arange(len(settings))

# %%
scores["rel_matched"] = scores["matched"] / scores["total_snps"].max()

# %%
plotdata = scores.join(design).groupby(["dataset", "regions", "motifscan", "method"])["odds"].mean().unstack().T
resolution = 0.3
fig, ax = plt.subplots(figsize=(10, plotdata.shape[0] * resolution))
for setting_id, setting_data in plotdata.items():
    ix = settings.loc[setting_id, "ix"]
    color = methods.loc[setting_data.index, "color"]
    ax.scatter(setting_data, [ix] * len(setting_data), color=color)
    ax.axhline(ix, color="gray", linestyle="--")
# ax.set_xscale("log")
# ax.set_xlim(1)

# %%
# plotdata.loc["chd"]/plotdata.loc["macs2_improved/wilcoxon"]

# %%
plotdata.style.bar(vmin = 1)

# %%
scores_joined = scores.join(design).groupby(["dataset", "regions", "motifscan", "method"])["odds"].mean().unstack().T

# %%
plotdata = scores_joined.dropna()
# plotdata = plotdata.fillna(plotdata.median(0))

plotdata = plotdata.loc[plotdata.mean(1).sort_values().index]
fig, ax = plt.subplots()
plotdata.mean(1).plot.barh(ax=ax)
ax.set_xlim(1)

# %%
main_features = ["dataset", "regions", "motifscan"]

# %%
plotdata = scores.join(design).groupby(["dataset", "regions", "motifscan", "method"])["odds"].mean().unstack().T
resolution = 0.1
fig, ax = plt.subplots(figsize=(10, plotdata.shape[0] * resolution))
for setting_id, setting_data in plotdata.items():
    ix = settings.loc[setting_id, "ix"]
    color = methods.loc[setting_data.index, "color"]
    ax.scatter(setting_data, [ix] * len(setting_data), color=color)

# %%
motifscan_info = pd.DataFrame(index = design["motifscan"].unique())
def determine_group(x):
    if "main" in x:
        return "gwas_main"
    elif "gwas" in x:
        return "gwas_ld"
    elif "causaldb" in x:
        return "causaldb"
    elif "gtex_caveman" in x and "differential" in x:
        return "gtex_caveman_differential"
    elif "gtex_caviar" in x and "differential" in x:
        return "gtex_caviar_differential"
    elif "differential" in x:
        return "gtex_caviar_differential"
    elif "gtex_caveman" in x:
        return "gtex_caveman"
    elif "gtex_caviar" in x:
        return "gtex_caviar"
    elif "gtex" in x:
        return "gtex_main"
motifscan_info["group"] = [determine_group(x) for x in motifscan_info.index]

# %%
scores_joined = scores.join(design).groupby(["dataset", "regions", "motifscan", "method"]).mean()
scores_common = scores_joined.reset_index()
scores_common["group"] = [motifscan_info.loc[x, "group"] for x in scores_common["motifscan"]]
scores_common["scored"] = ~pd.isnull(scores_common["odds"])

plotdata = scores_common.groupby(["group", "method"]).mean(numeric_only = True).reset_index()
plotdata["n"] = scores_common.groupby(["group", "method"]).size().values

plotdata_mean = plotdata.groupby("method").mean(numeric_only = True).reset_index()
plotdata_mean["n"] = plotdata.groupby("method").size().values
plotdata_mean = plotdata_mean.loc[plotdata_mean["n"] == plotdata_mean["n"].max()]
plotdata_mean["group"] = "mean"
plotdata = plotdata.loc[plotdata["method"].isin(plotdata_mean["method"])]

plotdata = pd.concat([
    plotdata,
    plotdata_mean
])

plotdata_groupby = plotdata.groupby("group")


# %%
motifscan_ids = motifscan_info.query("group in 'gwas_main'").index

# %%
(
    scores_joined.query("method == 'chd'").droplevel("method").query("motifscan in @motifscan_ids")["matched"]/
    scores_joined.query("method == 'macs2_improved/wilcoxon'").droplevel("method").query("motifscan in @motifscan_ids")["matched"]
    
)

# %%
y_info = pd.DataFrame(index = [*motifscan_info["group"].unique(), "mean"])
y_info["ix"] = np.arange(len(y_info))
y_info["label"] = y_info.index
y_info.loc["gwas_main", "label"] = "GWAS reported"
y_info.loc["causaldb", "label"] = "GWAS causaldb"
y_info.loc["gwas_ld", "label"] = "GWAS >0.9 R²"
y_info.loc["gtex_main", "label"] = "GTEx >0.9 R²"
y_info.loc["gtex_caviar", "label"] = "GTEx caviar"
y_info.loc["gtex_caveman", "label"] = "GTEx caveman"
y_info.loc["gtex_caviar", "label"] = "GTEx caviar"
y_info.loc["gtex_caveman", "label"] = "GTEx caveman"
y_info.loc["gtex_caviar_differential", "label"] = "GTEx caviar diff."
y_info.loc["gtex_caveman_differential", "label"] = "GTEx caveman diff."
y_info.loc["mean", "label"] = "Mean"
# y_info.loc["gtex_ca", "label"] = "GWAS causaldb"
# y_info.loc["causaldb", "label"] = "GWAS causaldb"

# %%
differential_methods = chdm.methods.differential_methods
differential_methods.loc["chd", "type"] = "ours"
# differential_methods = pd.DataFrame(index = plotdata.index).join(differential_methods)
# differential_methods["color"] = "black"
# differential_methods["type"] = "chd"

# %%
import matplotlib as mpl
import seaborn as sns
import chromatinhd.plot.quasirandom

# %%
fig, ax = plt.subplots(figsize=(4, len(plotdata_groupby) * 0.25))

score = "odds"
# score = "r2"
# score = "scored"
# score = "cor"

for dataset, plotdata_dataset in plotdata_groupby:
    x = y_info.loc[dataset]["ix"]

    plotdata_dataset = plotdata_dataset.loc[plotdata_dataset["n"] == plotdata_dataset["n"].max()].copy()

    plotdata_dataset[score] = plotdata_dataset[score].fillna(0.)
    y = np.array(chd.plot.quasirandom.offsetr(np.array(plotdata_dataset[score].values.tolist()), adjust=0.1)) * 0.8 + x
    plotdata_dataset["y"] = y

    plotdata_dataset["color"] = differential_methods.loc[
        plotdata_dataset["method"], "color"
    ].values
    plotdata_dataset.loc[plotdata_dataset["method"] == "v32", "color"] = "pink"
    plotdata_dataset.loc[plotdata_dataset["method"] == "v31", "color"] = "turquoise"

    ax.axhspan(x - 0.5, x + 0.5, color="#33333308" if dataset != "mean" else "#33333311", ec="white", zorder = -2)
    ax.scatter(plotdata_dataset[score], plotdata_dataset["y"], s=5, color=plotdata_dataset["color"], lw=0)

    plotdata_dataset["type"] = differential_methods.loc[
        plotdata_dataset["method"], "type"
    ].values
    plotdata_top = plotdata_dataset.sort_values(score, ascending=False).groupby("type").first()
    for i, (type, plotdata_top_) in enumerate(plotdata_top.groupby("type")):
        ax.plot(
            [plotdata_top_[score]] * 2,
            [x - 0.45, x + 0.45],
            color=plotdata_top_["color"].values[0],
            lw=2,
            zorder=0,
            alpha=0.9,
            solid_capstyle = "butt",
        )
    plotdata_ours = plotdata_dataset.loc[plotdata_dataset["method"] == "chd"].iloc[0]
    plotdata_top_others = plotdata_top.loc[plotdata_top.index != "ours"]

    try:
        plotdata_others_max = plotdata_top_others.loc[plotdata_top_others[score].idxmax()]
        rectangle = mpl.patches.Rectangle(
            (plotdata_others_max[score], x-0.45),
            plotdata_ours[score] - plotdata_others_max[score],
            0.9,
            fc=plotdata_ours["color"],
            ec="none",
            zorder=-1,
            alpha=1/3,
        )
        ax.add_patch(rectangle)
    except ValueError:
        pass
ax.set_xlim(1)
ax.set_yticks(y_info["ix"])
ax.set_yticklabels(y_info["label"])
for ticklabel in ax.get_yticklabels():
    if ticklabel.get_text() == "Mean":
        ticklabel.set_fontweight("bold")
    else:
        ticklabel.set_fontsize(9)
ax.set_xlabel("Average odds ratio")
sns.despine(ax=ax, left=True, bottom=True)
ax.set_ylim(-0.5, y_info["ix"].max() + 0.5)

manuscript.save_figure(fig, "3", "aggregate_qtl_odds")

# %%
plotdata_mean["peakcaller"] = plotdata_mean["method"].str.split("/").str[0]
plotdata_mean.to_csv(chd.get_output() / "aggregate_qtl_enrichment.csv")

# %%
plotdata_datasets = scores_joined.groupby(["method", "dataset"])[["odds"]].mean().reset_index()
plotdata_datasets["peakcaller"] = plotdata_datasets["method"].str.split("/").str[0]
plotdata_datasets.to_csv(chd.get_output() / "aggregate_qtl_enrichment_datasets.csv")

# %%
plotdata_mean_gwas = scores_joined.loc[~scores_joined.index.get_level_values("motifscan").str.contains("gtex")].groupby(["method"])[["odds"]].mean().reset_index()
plotdata_mean_gwas["peakcaller"] = plotdata_mean_gwas["method"].str.split("/").str[0]
plotdata_mean_gwas.to_csv(chd.get_output() / "aggregate_gwas_enrichment.csv")

# %%
plotdata_mean_eqtl = scores_joined.loc[scores_joined.index.get_level_values("motifscan").str.contains("gtex")].groupby(["method"])[["odds"]].mean().reset_index()
plotdata_mean_eqtl["peakcaller"] = plotdata_mean_eqtl["method"].str.split("/").str[0]
plotdata_mean_eqtl.to_csv(chd.get_output() / "aggregate_eqtl_enrichment.csv")

# %%
odds_chd = plotdata_mean.query('method == "chd"')["odds"].iloc[0]
odds_nonchd = plotdata_mean.query('method != "chd"').sort_values("odds", ascending = False)["odds"].iloc[0]

# %%
print(f"We found that ChromatinHD DARs are more strongly enriched for both types of natural variation, with an average odds-ratio of " + "{:.1f}".format(odds_chd) + " compared to the next best approach (MACS2 shared peaks, " + "{:.1f}".format(odds_nonchd) + ") (FIG:aggregate_qtl). GWAS QTLs uniquely within ChromatinHD DARs were associated to 'atypical' differential accessibility in the periphery of larger peaks (rs443623, FIG:example_HLADPA1), broad strong accessibility changes (rs875741, FIG:example_CPEB4), and intra-CRE variability (rs7668673, FIG:example_TBC1D14). Some of these associations were validated using allele-specific binding data complemented with changes in predicted binding affinity (FIG:example_HLADPA1, example_CPEB4, example_TBC1D14).")

# %%
