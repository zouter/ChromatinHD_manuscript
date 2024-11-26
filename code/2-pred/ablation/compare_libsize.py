# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

import scanpy as sc

# %%
# dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "pbmc10k"
dataset_name = "hspc"
splitter = "5x1"
layer = "magic"
regions_name = "100k100k"

method_name = 'radial_binary_1000-31frequencies_splitdistance_wd1e-1_linearlib'
dataset_folder = chd.get_output() / "datasets" / dataset_name
folds = chd.data.folds.Folds(dataset_folder / "folds" / splitter)

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
transcriptome.adata.obs["nfragments"] = fragments.counts.sum(1)
transcriptome.adata.obs["lognfragments"] = np.log1p(transcriptome.adata.obs["nfragments"])
plt.hist(np.log1p(transcriptome.adata.obs["nfragments"]))
sc.pl.umap(transcriptome.adata, color = "lognfragments")

# %%
# gene_oi = transcriptome.gene_id("GATA1")
# gene_oi = transcriptome.gene_id("KLF1")
# gene_oi = transcriptome.gene_id("CCL4")
# gene_oi = transcriptome.gene_id("ZSWIM5")
gene_oi = transcriptome.gene_id("GATA1")

# %%
from params import params

# %%
method_names = [
    "radial_binary_1000-31frequencies_residualfull_lne2e_linearlib",
    "radial_binary_1000-31frequencies_residualfull_lne2e"
]

for method_name in method_names:
    prediction = chd.flow.Flow(
        chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / method_name
    )
    performance = chd.models.pred.interpret.Performance(prediction.path / "scoring" / "performance")

    censorer = chd.models.pred.interpret.censorers.MultiWindowCensorer(fragments.regions.window)
    regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
        path = prediction.path / "scoring" / "regionmultiwindow",
        folds = folds, transcriptome = transcriptome, fragments = fragments, censorer = censorer
    )

    if not regionmultiwindow.scores["scored"].sel_xr(gene_oi).all():
        method_info = params[method_name]

        for region in [gene_oi]:
            models = chd.models.pred.model.better.Models.create(
                fragments=fragments,
                transcriptome=transcriptome,
                folds=folds,
                # reset = True,
                model_params={**method_info["model_params"], "layer": layer},
                train_params=method_info["train_params"],
                regions_oi=[region],
            )
            models.train_models()
            regionmultiwindow.score(models)


# %%
plotdatas = {}
for method_name in method_names:
    prediction = chd.flow.Flow(
        chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / method_name
    )
    regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
        path = prediction.path / "scoring" / "regionmultiwindow"
    )
    regionmultiwindow.interpolate()

    plotdatas[method_name] = (regionmultiwindow.get_plotdata(gene_oi))

# %%
fig, ax = plt.subplots()
for method_name, plotdata in plotdatas.items():
    ax.plot(plotdata.index, -plotdata["deltacor"], label = method_name)
ax.legend()
ax.plot(plotdata.index, plotdatas[method_names[0]]["deltacor"] - plotdatas[method_names[1]]["deltacor"], color = "black")
print((plotdatas[method_names[0]]["deltacor"] - plotdatas[method_names[1]]["deltacor"]).sum())
# ax.set_xlim(-5000, 5000)
ax.set_xlim(-10000, 20000)
