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

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %%
from design import (
    dataset_splitter_method_combinations as design_methods,
)
design_methods["group"] = "chd"

design = pd.concat([design_methods], axis=0, ignore_index=True)

# %%
scores = {}

for design_ix, design_row in design.query("group == 'chd'").iterrows():
    prediction = chd.flow.Flow(
        chd.get_output() / "diff" / design_row.dataset / design_row.regions / design_row.splitter / design_row.clustering / design_row.method,
    )
    performance = chd.models.diff.interpret.Performance(prediction.path / "scoring" / "performance")

    try:
        scores[design_ix] = performance.scores.sel_xr()
        # raise ValueError("Scores already exist")
    except FileNotFoundError:
        continue

scores = xr.concat(scores.values(), dim = pd.Index(scores.keys(), name = "design_ix"))

# %% [markdown]
# ## Overall

# %%
design_oi = design
# design_oi = design.query("regions == '10k10k'")
design_oi = design.query("regions == '100k100k'")

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi.mean("fold").sel(phase = "test")["scored"].mean("gene").to_pandas()
plotdata.index = design.iloc[plotdata.index]["method"].values
plotdata.plot.barh()

# %%
import scipy.stats

# %%
scores_oi = scores.sel(design_ix = scores.coords["design_ix"].isin(design_oi.index))
plotdata = scores_oi.mean("fold").sel(phase = "test")["likelihood"].mean("gene").to_pandas()
plotdata = plotdata[plotdata != 0]
# plotdata = plotdata - plotdata.min()
plotdata.index = design.iloc[plotdata.index]["method"].values
fig, ax = plt.subplots()
ax.scatter(plotdata, plotdata.index)

# %% [markdown]
# ## Pairwise comparison

# %%
design_oi = design.query("regions == '100k100k'")
# design_oi = design.query("regions == '10k10k'")

# %%
a = design_oi.query("method == 'binary_50bw'").index[0]
# a = design_oi.query("method == 'binary_1000-50bw_earlystop-no'").index[0]
b = design_oi.query("method == 'binary_1000-50bw'").index[0]

plotdata = scores["likelihood"].sel(phase = "test").mean("fold").to_pandas().T
# plotdata = plotdata.loc[(plotdata[b] != 0) & (plotdata[a] != 0)] # remove zero values

# %%
(plotdata[b] - plotdata[a]).mean(), (plotdata[b] > plotdata[a]).mean()

# %%
dataset_name = "pbmc10k/subsets/top250"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
regions_name = "100k100k"
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
import statsmodels.api as sm
import scipy.stats

fig = chd.grid.Figure(chd.grid.Wrap(padding_width = 0.7, ncol = 4))

plotdata = plotdata.sort_values(a)
plotdata["diff"] = plotdata[b] - plotdata[a]
plotdata["tick"] = (np.arange(len(plotdata)) % int(len(plotdata)/10)) == 0
plotdata["i"] = np.arange(len(plotdata))
plotdata["dispersions"] = transcriptome.var["dispersions"].loc[plotdata.index.get_level_values("gene")].values
plotdata["means"] = transcriptome.var["means"].loc[plotdata.index.get_level_values("gene")].values
plotdata["dispersions_norm"] = transcriptome.var["dispersions_norm"].loc[plotdata.index.get_level_values("gene")].values
plotdata["log10means"] = np.log10(plotdata["means"])
plotdata["log10dispersions_norm"] = np.log10(plotdata["dispersions_norm"])
n_fragments = pd.Series(fragments.counts.mean(0), index = fragments.var.index)
plotdata["log10n_fragments"] = np.log10(n_fragments.loc[plotdata.index.get_level_values("gene")].values)
n_fragments_std = pd.Series(np.log1p(fragments.counts).std(0), index = fragments.var.index)
plotdata["n_fragments_std"] = n_fragments_std.loc[plotdata.index.get_level_values("gene")].values
plotdata["oi"] = pd.Categorical([True] * len(plotdata), categories = [True, False])
plotdata["kurtosis"] = (scipy.stats.kurtosis(transcriptome.layers["magic"]))
plotdata["kurtosis_rank"] = scipy.stats.rankdata(plotdata["kurtosis"])

a_label = str(a)
b_label = str(b)

cmap = mpl.colormaps["Set1"]

# rank vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(np.arange(len(plotdata)), (plotdata["diff"]), c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a_label + " rank")
ax.set_ylabel("$\Delta$ r2")
ax.axhline(0, color = "black", linestyle = "--")
ax.set_xticks(plotdata["i"].loc[plotdata["tick"]])
ax.set_xticklabels(plotdata[a].loc[plotdata["tick"]].round(2).values, rotation = 90)

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], np.arange(len(plotdata)), frac = 0.5)
ax.plot(z[:, 0], z[:, 1], color = "green")

# a vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata[a], (plotdata["diff"]), c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a_label)
ax.set_ylabel("$\Delta$ r2")
ax.axhline(0, color = "black", linestyle = "--")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata[a], frac = 0.5)
ax.plot(z[:, 0], z[:, 1], color = "green")

lm = scipy.stats.linregress(plotdata[a], plotdata["diff"])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope

# vs
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata[a], plotdata[b], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel(a_label)
ax.set_ylabel(b_label)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

lowess = sm.nonparametric.lowess
z = lowess(plotdata[b], plotdata[a], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")
ax.axline((0, 0), slope = 1, color = "black", linestyle = "--")
ax.set_aspect("equal")

lm = scipy.stats.linregress(plotdata[a], plotdata[b])
ax.axline((0, 0), slope = lm.slope, color = "cyan", linestyle = "--")
lm.slope

cut = (1 - lm.intercept) / lm.slope
print(cut)
ax.annotate(f"cut {1-cut:.1%}", (1, 1), (1, 1.1), arrowprops = dict(arrowstyle = "->"), ha = "right")
ax.annotate(f"$r^2$={lm.rvalue**2:.1%}", (0.95, 0.95), ha = "right", va = "top")

# dispersions vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["dispersions"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("dispersion")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["dispersions"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10means"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10means")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10means"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# dispersions_norm vs diff
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10dispersions_norm"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10dispersions_norm")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10dispersions_norm"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["log10n_fragments"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("log10n_fragments")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["log10n_fragments"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

# n fragments std
panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
ax.scatter(plotdata["kurtosis"], plotdata["diff"], c = cmap(plotdata["oi"].cat.codes), s = 3)
ax.set_xlabel("kurtosis")
ax.set_ylabel("$\Delta$ r2")

lowess = sm.nonparametric.lowess
z = lowess(plotdata["diff"], plotdata["kurtosis"], frac = 2/3)
ax.plot(z[:, 0], z[:, 1], color = "green")

fig.plot()

# %%
plotdata.loc[transcriptome.gene_id("EBF1")]["log10n_fragments"]
