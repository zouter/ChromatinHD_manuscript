import os
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

dataset_name = "pbmc10k"
regions_name = "100k100k"

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")
fold = folds[0]

import chromatinhd.models.pred.model.better

models = chd.models.pred.model.better.Models(chd.get_output() / "pred/pbmc10k/100k100k/5x5/magic/v33")


censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, window_sizes=(100,))
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
    folds,
    transcriptome,
    fragments,
    censorer,
    path=models.path / "scoring" / "regionmultiwindow_100",
    # overwrite=True,
)
print(regionmultiwindow.scores["scored"].sel_xr().all("fold").sum())

extrema_interleaved = [10, 110, 170, 270, 390, 470, 590, 690, 770]
cuts = [0, *(extrema_interleaved[:-1] + np.diff(extrema_interleaved) / 2), 99999]
sizes = pd.DataFrame(
    {
        "start": cuts[:-1],
        "end": cuts[1:],
        "length": np.diff(cuts),
        "mid": [*(cuts[:-2] + np.diff(cuts)[:-1] / 2), cuts[-2] + 10],
    }
)

sizes = pd.DataFrame({"start": [0, 60], "end": [60, 140], "length": [60, 80], "mid": [30, 100]})

regionsizewindow = chd.models.pred.interpret.RegionSizeWindow(models.path / "scoring" / "regionsizewindow")

# force = True
force = False
for gene_oi in tqdm.tqdm(transcriptome.var.index[::-1]):
    print(gene_oi)
    # for gene_oi in [transcriptome.gene_id("CCL4")]:
    if regionmultiwindow.scores["scored"].sel_xr(gene_oi).all():
        if force or (gene_oi not in regionsizewindow.scores):
            windows_oi = regionmultiwindow.design.query("window_size == 100").index
            # windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]
            windows_oi = windows_oi[
                (
                    regionmultiwindow.scores["deltacor"]
                    .sel_xr(gene_oi)
                    .sel(phase="test")
                    .sel(window=windows_oi.tolist())
                    .mean("fold")
                    < -0.0001
                )
            ]
            print(len(windows_oi))
            if len(windows_oi) > 0:
                design_windows = regionmultiwindow.censorer.design.loc[windows_oi.tolist()]
                design_windows

                censorer = chd.models.pred.interpret.censorers.WindowSizeCensorer(design_windows, sizes)
                regionsizewindow.score(models, regions=[gene_oi], censorer=censorer, device="cpu", force=force)
