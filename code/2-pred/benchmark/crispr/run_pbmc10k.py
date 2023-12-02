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

import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd_manuscript as chdm

from chromatinhd_manuscript import crispri

from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

chd.set_default_device("cuda:0")

dataset_name = "pbmc10k"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")

genes_oi = ['ENSG00000005844', 'ENSG00000160223', 'ENSG00000170525', 'ENSG00000159128', 'ENSG00000134954', 'ENSG00000101017', 'ENSG00000134460', 'ENSG00000136573']

import chromatinhd.models.pred.model.better

from chromatinhd_manuscript.pred_params import params
model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / "5x5" / "magic" / "v31"
model_params = params["v31"]["model_params"]
train_params = params["v31"]["train_params"]

models = chd.models.pred.model.better.Models.create(
    fragments=fragments,
    transcriptome=transcriptome,
    folds=folds,
    path=model_folder,
    # reset = True,
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
    path = scoring_folder / "regionmultiwindow",
    folds = folds,
    transcriptome = transcriptome,
    fragments = fragments,
    censorer = censorer,
    # reset = True
)
regionmultiwindow.score(models, regions=genes_oi)
regionmultiwindow.interpolate()
