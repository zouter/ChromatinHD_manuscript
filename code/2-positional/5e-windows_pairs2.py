# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")


# %%
device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"
prediction_name = "v21"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %% [markdown]
# ## Subset

# ##
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)
genes_all_oi = transcriptome.var.index[
    (nothing_scoring.genescores.sel(phase="test").mean("model").mean("i")["cor"] > 0.1)
]
transcriptome.var.loc[genes_all_oi].head(30)

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

# symbol = "SELL"
# symbol = "BACH2"
# symbol = "CTLA4"
# symbol = "SPI1"
# symbol = "IL1B"
# symbol = "TCF3"
# symbol = "PCDH9"
# symbol = "EBF1"
# symbol = "QKI"
# symbol = "NKG7"
# symbol = "CCL4"
# symbol = "TCF4"
# symbol = "TSHZ2"
# symbol = "IL1B"
# symbol = "PAX5"
# symbol = "CUX2"
# symbol = "CD79A"
# symbol = "RALGPS2"
# symbol = "RHEX"
# symbol = "PTPRS"
symbol = "RGS7"
symbol = "CD74"
# symbol = "PLXNA4"
# symbol = "TNFRSF21"
# symbol = "MEF2C"
# symbol = "BCL2"
# symbol = "CCL4"
# symbol = "EBF1"
# symbol = transcriptome.symbol("ENSG00000175985")
print(symbol)
genes_oi = transcriptome.var["symbol"] == symbol
gene = transcriptome.var.index[genes_oi][0]

gene_ix = transcriptome.gene_ix(symbol)
gene = transcriptome.var.iloc[gene_ix].name

# %%
sc.pl.umap(transcriptome.adata, color=gene, use_raw=False, show=False)

# %% [markdown]
# ## Window

# %% [markdown]
# ### Load

# %%
scores_folder = prediction.path / "scoring" / "window_gene" / gene
window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
sns.scatterplot(
    x=window_scoring.genescores["deltacor"]
    .sel(phase="test")
    .mean("model")
    .values.flatten(),
    y=window_scoring.genescores["deltacor"]
    .sel(phase="validation")
    .mean("model")
    .values.flatten(),
    hue=np.log1p(
        window_scoring.genescores["lost"]
        .sel(phase="validation")
        .mean("model")
        .values.flatten()
    ),
)

# %%
# genescores["cor"].mean("model").sel(phase = "train").sel(gene = transcriptome.gene("IL1B")).plot()
# genescores["cor"].mean("model").sel(phase = "validation").sel(gene = transcriptome.gene("CTLA4")).plot()
fig, ax = plt.subplots()
window_scoring.genescores["deltacor"].sel(phase="validation").sel(gene=gene).mean(
    "model"
).plot(ax=ax)
window_scoring.genescores["deltacor"].sel(phase="test").sel(gene=gene).mean(
    "model"
).plot(ax=ax, color="blue")
ax.yaxis_inverted()
ax2 = ax.twinx()
window_scoring.genescores["retained"].sel(phase="test").mean("gene").mean("model").plot(
    ax=ax2, color="red"
)
window_scoring.genescores["retained"].sel(phase="validation").mean("gene").mean(
    "model"
).plot(ax=ax2, color="orange")
ax2.yaxis_inverted()

# %% [markdown]
# ## Pairwindow

scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
windowpair_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# import pathlib

# scores_folder = (
#     pathlib.Path(str(prediction.path).replace("v20", "v21"))
#     / "scoring"
#     / "pairwindow_gene"
#     / gene
# )
# windowpair_scoring2 = chd.scoring.prediction.Scoring.load(scores_folder)

scores_folder = prediction.path / "scoring" / "pairwindow_gene_baseline" / gene
windowpair_baseline_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
retained_additive = pd.Series(
    1
    - (
        (
            1
            - (
                window_scoring.genescores["retained"]
                .sel(gene=gene)
                .mean("model")
                .sel(
                    window=windowpair_scoring.design["window2"].values,
                    phase="validation",
                )
            ).values
        )
        + (
            1
            - (
                window_scoring.genescores["retained"]
                .sel(gene=gene)
                .mean("model")
                .sel(
                    window=windowpair_scoring.design["window1"].values,
                    phase="validation",
                )
            ).values
        )
    ),
    windowpair_scoring.design.index,
)

# %%
# because some fragments may be in two windows, we need to use a baseline to correct for this
additive_baseline = windowpair_baseline_scoring.genescores["cor"].sel(gene=gene)
additive_base1 = (
    window_scoring.genescores["cor"]
    .sel(gene=gene)
    .sel(window=windowpair_scoring.design["window1"].values)
).reindex_like(additive_baseline)
additive_base2 = (
    window_scoring.genescores["cor"]
    .sel(gene=gene)
    .sel(window=windowpair_scoring.design["window2"].values)
).reindex_like(additive_baseline)

deltacor1 = additive_base1.values - additive_baseline
deltacor2 = additive_base2.values - additive_baseline
deltacor_additive = (
    additive_base1.values + additive_base2.values
) - 2 * additive_baseline
deltacor_interacting = (
    windowpair_scoring.genescores["cor"].sel(gene=gene) - additive_baseline
)

# %%
# because some fragments may be in two windows, we need to use a baseline to correct for this
additive_baseline = windowpair_baseline_scoring.genescores["effect"].sel(gene=gene)
additive_base1 = (
    window_scoring.genescores["effect"]
    .sel(gene=gene)
    .sel(window=windowpair_scoring.design["window1"].values)
).reindex_like(additive_baseline)
additive_base2 = (
    window_scoring.genescores["effect"]
    .sel(gene=gene)
    .sel(window=windowpair_scoring.design["window2"].values)
).reindex_like(additive_baseline)

effect1 = additive_base1.values - additive_baseline
effect2 = additive_base2.values - additive_baseline
effect_additive = (
    additive_base1.values + additive_base2.values
) - 2 * additive_baseline / 2
effect_interacting = windowpair_scoring.genescores["effect"].sel(gene=gene)

deltaeffect_interacting = effect_interacting - effect_additive
# effect_interacting = (
#     windowpair_scoring.genescores["effect"].sel(gene=gene) - additive_baseline
# )

# %%
# phase = "test"
phase = "validation"

interaction = windowpair_scoring.design.copy()
interaction["deltacor"] = (
    deltacor_interacting.sel(phase=phase).mean("model").to_pandas()
)
interaction["deltacor1"] = deltacor1.sel(phase=phase).mean("model").to_pandas()
interaction["deltacor2"] = deltacor2.sel(phase=phase).mean("model").to_pandas()
interaction["effect1"] = effect1.sel(phase=phase).mean("model").to_pandas()
interaction["effect2"] = effect2.sel(phase=phase).mean("model").to_pandas()
interaction["effect"] = effect_interacting.sel(phase=phase).mean("model").to_pandas()

additive = windowpair_scoring.design.copy()
additive["deltacor"] = deltacor_additive.sel(phase=phase).mean("model").to_pandas()
additive["effect"] = effect_additive.sel(phase=phase).mean("model").to_pandas()

# %%
sns.heatmap(interaction.set_index(["window_mid1", "window_mid2"])["effect"].unstack())

# %%
sns.heatmap(interaction.set_index(["window_mid1", "window_mid2"])["deltacor"].unstack())

# %%
interaction["deltacor_interaction"] = interaction["deltacor"] - additive["deltacor"]
interaction["effect_interaction"] = interaction["effect"] - additive["effect"]

interaction_effect_matrix = interaction.set_index(["window_mid1", "window_mid2"])[
    "effect_interaction"
].unstack()
interaction_effect_matrix = interaction_effect_matrix.fillna(
    0
) + interaction_effect_matrix.T.fillna(0)

additive_effect_matrix = additive.set_index(["window_mid1", "window_mid2"])[
    "effect"
].unstack()
additive_effect_matrix = additive_effect_matrix.fillna(
    0
) + additive_effect_matrix.T.fillna(0)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(interaction_effect_matrix)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(additive_effect_matrix)

# %%
x = deltacor_interacting.sel(phase="test").values - deltacor_additive.sel(phase="test")
# %%
def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


# %%
import scipy.stats

scores_statistical = []
for i in range(x.shape[1]):
    y = x[:, i]
    if y.std() < 1e-10:
        scores_statistical.append(1)
    else:
        scores_statistical.append(scipy.stats.ttest_1samp(y, 0).pvalue)
scores_statistical = pd.DataFrame({"pval": scores_statistical})
scores_statistical["qval"] = fdr(scores_statistical["pval"])
interaction["pval"] = scores_statistical["pval"].values
interaction["qval"] = scores_statistical["qval"].values

# %%
radius = (
    (
        windowpair_scoring.design["window_end1"]
        - windowpair_scoring.design["window_start1"]
    )
)[0] / 2

# %%
interaction["effect_prod"] = np.prod(interaction[["effect1", "effect2"]], 1)

# %%
fig, ax = plt.subplots()
ax.scatter(
    interaction.query("qval < 0.05")["effect_prod"],
    interaction.query("qval < 0.05")["effect_interaction"],
)

# %%
import scipy.stats


interaction["effect_prod"] = np.prod(interaction[["effect1", "effect2"]], 1)

lm = scipy.stats.linregress(
    interaction["effect_prod"], interaction["effect_interaction"]
)

interaction["effect_prod_corrected"] = (
    interaction["effect_interaction"]
    - lm.intercept
    - lm.slope * interaction["effect_prod"]
)

# %%
plt.scatter(interaction["effect_prod"], interaction["effect_interaction"])
plt.axline(
    [0, lm.intercept],
    slope=lm.slope,
)

# %%
promoter = promoters.loc[gene]
genome_folder = folder_data_preproc / "genome"

# %%
main = chd.grid.Grid(padding_height=0.1)
fig = chd.grid.Figure(main)

panel_width = 8

plotdata_predictive = (
    window_scoring.genescores.sel(gene=gene).sel(phase="test").mean("model").to_pandas()
)
plotdata_predictive["position"] = plotdata_predictive.index

panel_genes = chdm.plotting.genes.Genes(
    promoter, genome_folder, window, width=panel_width
)
panel_genes = main.add_under(panel_genes)

panel_predictive = chd.predictive.plot.Predictive(
    plotdata_predictive, window, panel_width
)
panel_predictive = main.add_under(panel_predictive, padding=0)

panel_interaction = main.add_under(chd.grid.Panel((panel_width, panel_width / 2)))
ax = panel_interaction.ax

norm = mpl.colors.Normalize(-0.001, 0.001)
# norm = mpl.colors.CenteredNorm()

cmap = mpl.cm.RdBu

for windowpair, plotdata_row in interaction.iterrows():
    # if plotdata_row["pval"] >= 1.0:
    if plotdata_row["qval"] >= 0.1:
        continue
    window_mid1 = plotdata_row["window_mid1"]
    window_mid2 = plotdata_row["window_mid2"]
    patch = mpl.patches.RegularPolygon(
        (
            window_mid1 + (window_mid2 - window_mid1) / 2,
            (window_mid2 - window_mid1) / 2,
        ),
        4,
        radius=radius,
        orientation=np.pi / 2,
        ec=None,
        lw=0,
        # fc=cmap(norm(plotdata_row["deltacor_interaction_corrected"])),
        # fc=cmap(norm(plotdata_row["effect_interaction"])),
        # fc=cmap(norm(plotdata_row["effect_prod_corrected"])),
        fc=cmap(norm(plotdata_row["deltacor_interaction"])),
    )
    ax.add_patch(patch)
ax.set_ylim((window[1] - window[0]) / 2)
ax.set_xlim(*window)

for x in np.linspace(*window, 16):
    x2 = x
    x1 = x2 + (window[0] - x2) / 2
    y2 = 0
    y1 = x2 - x1

    if np.isclose(x1, window[1]) or (np.isclose(x2, window[1])):
        color = "black"
        lw = 1
        zorder = 10
    else:
        color = "#eee"
        lw = 0.5
        zorder = -1
    ax.plot(
        [x1, x2],
        [y1, y2],
        zorder=zorder,
        color=color,
        lw=lw,
    )
    ax.plot(
        [-x1, -x2],
        [y1, y2],
        zorder=zorder,
        color=color,
        lw=lw,
    )
ax.axis("off")

fig.plot()

# %%
