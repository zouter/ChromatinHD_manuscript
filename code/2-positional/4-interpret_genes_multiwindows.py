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
import IPython

if IPython.get_ipython():
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
    IPython.get_ipython().run_line_magic(
        "config", "InlineBackend.figure_format='retina'"
    )

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import tqdm.auto as tqdm

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
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initidefault"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "10k10k", np.array([-10000, 10000])
# prediction_name = "v20"

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

# %%
import pathlib

# x = pd.read_csv(
#     pathlib.PosixPath(
#         "/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/prediction_positional/pbmc10k/10k10k/permutations_5fold5repeat/v20/scoring/windows"
#     )
#     / "correlation_nfragments_deltacor_low.csv"
# )
x = pd.read_csv(
    pathlib.PosixPath(
        "/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/prediction_positional/pbmc10k/10k10k/random_5fold/v20/scoring/"
    )
    / "difference_with_peakcalling_large.csv"
    # / "difference_with_peakcalling_small.csv"
)
x["symbol"] = transcriptome.symbol(x["gene"]).values
n = 0
for gene in x["gene"]:
    try:
        scores_folder = prediction.path / "scoring" / "multiwindow_gene" / gene
        multiwindow_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

        print(transcriptome.symbol(gene), x.query("gene == @gene").iloc[0]["cor_a"])

        n += 1
        if n > 20:
            break
    except:
        continue


# %% [markdown]
## Plot

# %%

genome_folder = folder_data_preproc / "genome"
peaks_folder = chd.get_output() / "peaks" / dataset_name


def plot_genes(symbols, resolution=0.0003):
    # load interpolateds
    genes = transcriptome.gene_id(symbols)
    interpolateds = {}
    for gene in genes:
        scores_folder = prediction.path / "scoring" / "multiwindow_gene" / gene
        interpolated = pickle.load((scores_folder / "interpolated.pkl").open("rb"))
        interpolateds[gene] = interpolated

    # actual plotting
    panel_width = (window[1] - window[0]) * resolution

    main = chd.grid.Grid(padding_width=0.1, padding_height=0.15)
    fig = chd.grid.Figure(main)

    for i, symbol in enumerate(symbols):
        print(symbol)
        gene = transcriptome.gene_id(symbol)
        if gene not in interpolateds:
            raise ValueError(f"Gene {symbol} not in interpolateds")
        interpolated = interpolateds[gene]
        promoter = promoters.loc[gene]
        plotdata_predictive = pd.DataFrame(
            {
                "deltacor": interpolated["deltacor_test"].mean(0),
                "lost": interpolated["lost"].mean(0),
                "position": pd.Series(np.arange(*window), name="position"),
            }
        )

        grid_gene = main[i, 0] = chd.grid.Grid(padding_width=0.1, padding_height=0.05)

        genes_panel = grid_gene[0, 0] = chdm.plotting.Genes(
            promoter,
            genome_folder=genome_folder,
            window=window,
            width=panel_width,
        )
        if main.nrow > 1:
            genes_panel.ax.set_xlabel("")
            genes_panel.ax.set_xticks([])

        peaks_panel = grid_gene[2, 0] = chdm.plotting.Peaks(
            promoter,
            peaks_folder,
            window=window,
            width=panel_width,
            row_height=0.4,
        )

        predictive_panel = grid_gene[1, 0] = chd.predictive.plot.Predictive(
            plotdata_predictive,
            window,
            panel_width,
        )
    fig.plot()
    return fig


# %% [markdown]
# Main example

fig = plot_genes(["NKG7", "CCL4", "QKI"])
manuscript.save_figure(fig, "3", "examples_predictive_main")

# %% [markdown]
# Supplementary examples

fig = plot_genes(["NAMPT", "PHEX", "CXCL8", "ZEB1"])
manuscript.save_figure(fig, "3", "examples_predictive_supplementary")

# %%
fig = plot_genes(["TBC1D2"])

# %%
fig = plot_genes(["BCL2"])

# %% [markdown]
# ## Variants/haplotypes

# %%
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
promoter = promoters.loc[gene]

# %%
motifscan_name = "gwas_immune"

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on="snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

# %%
association_oi = association.loc[
    (association["chr"] == promoter["chr"])
    & (association["pos"] >= promoter["start"])
    & (association["pos"] <= promoter["end"])
].copy()

# %%
association_oi["position"] = (association_oi["pos"] - promoter["tss"]) * promoter[
    "strand"
]

# %%
variants = pd.DataFrame(
    {
        "disease/trait": association_oi.groupby("snp")["disease/trait"].apply(list),
        "snp_main_first": association_oi.groupby("snp")["snp_main"].first(),
    }
)
variants = variants.join(snp_info)
variants["position"] = (variants["start"] - promoter["tss"]) * promoter["strand"]

haplotypes = (
    association_oi.groupby("snp_main")["snp"]
    .apply(lambda x: sorted(set(x)))
    .to_frame("snps")
)
haplotypes["color"] = sns.color_palette("hls", n_colors=len(haplotypes))

# %% [markdown]
# ### Compare to individual position ranking

# %%
fig, ax = plt.subplots(figsize=(20, 3))
ax.plot(
    positions_oi * promoter["strand"] + promoter["tss"],
    deltacor_test_interpolated.mean(0),
)
ax2 = ax.twinx()
ax2.plot(
    positions_oi * promoter["strand"] + promoter["tss"],
    retained_interpolated.mean(0),
    color="red",
    alpha=0.6,
)
ax2.set_ylabel("retained")

for _, variant in variants.iterrows():
    ax.scatter(
        variant["position"] * promoter["strand"] + promoter["tss"],
        0.9,
        color=haplotypes.loc[variant["snp_main_first"], "color"],
        s=100,
        marker="|",
        transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
    )

ax.invert_yaxis()
ax2.invert_yaxis()

# %%
import gseapy

# %%
rnk = pd.Series(0, index=pd.Series(list("abcdefghijklmnop")))
rnk.values[:] = -np.arange(len(rnk))
genesets = {"hi": ["a", "b", "c"]}

# %%
rnk = -pd.Series(deltacor_test_interpolated.mean(0), index=positions_oi.astype(str))
genesets = {"hi": np.unique(variants["position"].astype(str).values)}

# %%
# ranked = gseapy.prerank(rnk, genesets, min_size = 0)

# %%
rnk_sorted = pd.Series(np.sort(np.log(rnk)), index=rnk.index)
# rnk_sorted = pd.Series(np.sort(rnk), index = rnk.index)
fig, ax = plt.subplots()
sns.ecdfplot(rnk_sorted, ax=ax)
sns.ecdfplot(
    rnk_sorted[variants["position"].astype(int).astype(str)], ax=ax, color="orange"
)
for _, motifdatum in variants.iterrows():
    rnk_motif = rnk_sorted[str(int(motifdatum["position"]))]
    q = np.searchsorted(rnk_sorted, rnk_motif) / len(rnk_sorted)
    ax.scatter([rnk_motif], [q], color="red")
    # ax.scatter(motifdatum["position"], 0, color = "red", s = 5, marker = "|")
