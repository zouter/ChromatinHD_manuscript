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
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

# %%
import chromatinhd as chd
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

regions_name = "10k10k"
regions_name = "100k100k"

# fragments
regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / regions_name)
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
counts = np.bincount(
    fragments.mapping[:, 0] * fragments.n_regions + fragments.mapping[:, 1],
    minlength=fragments.n_regions * fragments.n_cells,
)

# %%
bincounts = np.bincount(counts, minlength=10)
bindensity = bincounts / bincounts.sum()


# %%
perc_zero = bindensity[0].sum()
perc_only_one = bindensity[1].sum()
perc_only_two = bindensity[2].sum()
perc_more_than_one = bindensity[2:].sum()

# %%
fig, ax = plt.subplots(figsize=(2, 1))
bins = np.arange(0, 10)
ax.bar(bins, bindensity[: len(bins)], width=1, lw=1)
ax.annotate(
    f"{perc_zero:.0%}",
    (0.0 - 0.4, perc_zero),
    # (0, 0),
    # textcoords="offset points",
    va="bottom",
    ha="left",
)
ax.annotate(
    f"{perc_only_one:.0%}",
    (1.0 - 0.4, perc_only_one),
    # (0, 0),
    # textcoords="offset points",
    va="bottom",
    ha="left",
)
ax.annotate(
    f"{perc_more_than_one:.0%} →",
    (2.0 - 0.4, perc_only_two),
    # (0, 0),
    # textcoords="offset points",
    va="bottom",
    ha="left",
)
sns.despine(ax=ax)
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))

if regions_name == "100k100k":
    # ax.set_ylabel("Fraction of cells", ha = "left")
    ax.set_xlabel(f"Number of fragments\nper cell and region")
else:
    ax.set_xticklabels([])

manuscript.save_figure(fig, "5", "fragments_per_cell_and_gene_" + regions_name)

# %%
import IPython

IPython.display.Markdown(
    f"""Note that in the current multiome data, the co-occurence of multiple fragments close to a gene (-10kb and +10kb from TSS) within the same cell is relatively rare, with {perc_more_than_one:.1%} of cells containing more than one fragment within this window, compared to {perc_only_one:.1%} of cells containing only a single fragment."""
)

# %%
