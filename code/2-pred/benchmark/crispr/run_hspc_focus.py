import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd_manuscript as chdm

from chromatinhd_manuscript import crispri

from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

chd.set_default_device("cuda:0")

dataset_name = "hspc_focus"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "500k500k")

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")

genes_oi = ["ENSG00000102145", "ENSG00000105610", "ENSG00000179218", "ENSG00000184897"]

##
from chromatinhd_manuscript.pred_params import params

model_folder = chd.get_output() / "pred" / dataset_name / "500k500k" / "5x5" / "magic" / "v34"
model_params = params["v34"]["model_params"]
train_params = params["v34"]["train_params"]
##

models = chd.models.pred.model.better.Models.create(
    fragments=fragments,
    transcriptome=transcriptome,
    folds=folds,
    path=model_folder,
    # reset=True,
    model_params=model_params,
    train_params=train_params,
)
print(models.path)

models.train_models(regions_oi=genes_oi)

# interpret
censorer = chd.models.pred.interpret.MultiWindowCensorer(
    fragments.regions.window,
    relative_stride=1,
)

scoring_folder = model_folder / "scoring" / "crispri" / "fulco_2019"

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
    path=scoring_folder / "regionmultiwindow",
    folds=folds,
    transcriptome=transcriptome,
    fragments=fragments,
    censorer=censorer,
    overwrite=True,
)
regionmultiwindow.score(models, regions=genes_oi)
regionmultiwindow.interpolate()
