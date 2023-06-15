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

import torch

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

#
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "100k100k", np.array([-100000, 100000])

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
counts = torch.bincount(
    fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1],
    minlength=fragments.n_genes * fragments.n_cells,
)

# %%
bincounts = torch.bincount(counts, minlength=10)
bindensity = bincounts / bincounts.sum()


# %%
perc_zero = bindensity[0].sum()
perc_only_one = bindensity[1].sum()
perc_only_two = bindensity[2].sum()
perc_more_than_one = bindensity[2:].sum()

# %%
fig, ax = plt.subplots(figsize=(2, 2))
bins = np.arange(0, 10)
ax.bar(bins, bindensity[: len(bins)])
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
ax.set_xlabel(
    f"Number of fragments\nper cell and gene in\n{window_width/1000:.0f}kb window around TSS"
)
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_ylabel("Fraction of cells")

manuscript.save_figure(fig, "5", "fragments_per_cell_and_gene_" + promoter_name)

# %%
import IPython

IPython.display.Markdown(
    f"""Note that in the current multiome data, the co-occurence of multiple fragments close to a gene (-10kb and +10kb from TSS) within the same cell is relatively rare, with {perc_more_than_one:.1%} of cells containing more than one fragment within this window, compared to {perc_only_one:.1%} of cells containing only a single fragment."""
)

# %%
