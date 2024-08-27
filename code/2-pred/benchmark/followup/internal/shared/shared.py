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
dataset_name = "pbmc10k"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "10k10k")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")
fold = folds[1]

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5_cellxregion").sample_cellxregion(fragments, 5, 5)
fold = folds[0]

# %%
from params import params
import pickle

# %%
import logging
chd.models.pred.trainer.trainer.logger.handlers = []
chd.models.pred.trainer.trainer.logger.propagate = False

# %%
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/*/48c5b3dccdabeed174f01d1282276f58
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/10k10k/*/e4ba9692843da9372d6a79c5570fa0c3
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name}/*/7bd6b1354af76bfa853313c822cb26cb
# # !rm -r {chd.get_output() / "models" / "mini" / "pbmc10k"}/10k10k/ENSG00000158813/7bd6b1354af76bfa853313c822cb26cb

# %%
layer = "magic"

# %%
# minibatch = 

# %%
for param_id, param in params.items():
    param = param.copy()
    cls = param.pop("cls")

    train_params = {}
    if "n_cells_step" in param:
        train_params["n_cells_step"] = param.pop("n_cells_step")
    if "lr" in param:
        train_params["lr"] = param.pop("lr")

    if "label" in param:
        param.pop("label")
    for regions_name in ["10k10k", "100k100k"]:
        model_folder = chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name / param_id

        fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

        if (model_folder / "performance").exists():
            continue

        model = cls(
            fragments = fragments,
            transcriptome=transcriptome,
            fold = fold,
            layer = layer,
            **param,
        )
        model.train_model(**train_params, pbar = True, n_epochs = 50)
        performance = chd.models.pred.interpret.Performance(path = model_folder / "performance")
        performance.score(fragments, transcriptome, [model], [fold], pbar = False)

        trace = {
            "n_train_checkpoints":len(model.trace.train_steps)
        }
        pickle.dump(trace, open(model_folder / "trace.pkl", "wb"))

# %% [markdown]
# ## Compare

# %%
model.trace.plot()

# %%
from params import params
import pickle

# %%
param_summary = pd.DataFrame({**params}).T
param_summary["label"] = [param["label"] if str(param["label"]) != "nan" else param_id for param_id, param in param_summary.iterrows()]

# %%
genescores = []
for param_id, param in {**params}.items():
    for regions_name in ["10k10k", "100k100k"]:
        model_folder = chd.get_output() / "models" / "mini" / "pbmc10k" / regions_name / param_id
        print(model_folder)
        performance_folder = model_folder / "performance"
        if (performance_folder / "scores.pkl").exists():
            genescore = pickle.load(open(performance_folder / "genescores.pkl", "rb"))
            genescore = genescore.mean("model").to_dataframe().reset_index()
            genescore["param"] = param_id
            genescore["regions"] = regions_name
            genescores.append(genescore)
genescores = pd.concat(genescores).set_index(["gene", "regions", "param", "phase"])
genescores = xr.Dataset.from_dataframe(genescores)
genescores.coords["param"] = param_summary["label"].values

# %%
genes_test = transcriptome.var.index[fold["regions_test"]]
genes_train = transcriptome.var.index[fold["regions_train"]]
genes_validation = transcriptome.var.index[fold["regions_validation"]]

# %%
param_summary["color"] = sns.color_palette("tab10", len(param_summary))
param_info = param_summary.set_index("label")

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.set_title("10k10k")
for param in genescores.coords["param"].values:
    sns.ecdfplot(genescores.sel(regions = "10k10k", param = param, gene = genes_test)["cor"].mean(["phase"]).to_pandas(), ax = ax, label = param, color = param_info.loc[param, "color"])
    sns.ecdfplot(genescores.sel(regions = "10k10k", param = param, gene = genes_train)["cor"].mean(["phase"]).to_pandas(), ax = ax, label = param, linestyle = "--", color = param_info.loc[param, "color"])

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.set_title("100k100k")
for param in genescores.coords["param"].values:
    sns.ecdfplot(genescores.sel(regions = "100k100k", param = param, gene = genes_test)["cor"].mean(["phase"]).to_pandas(), ax = ax, label = param)

ax.legend(loc = "center left", bbox_to_anchor = (1, 0.5), ncol = 1)

fig.plot()

# %%
(genescores.sel(regions = "100k100k", gene = genes_train)["cor"]**2).mean(["gene", "phase"]).to_pandas()

# %%
(genescores.sel(regions = "10k10k", gene = genes_validation)["cor"]**2).sel(phase = "test").mean("gene").to_pandas()
