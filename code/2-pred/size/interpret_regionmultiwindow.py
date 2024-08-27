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

force = False
force = True

# for gene_oi in tqdm.tqdm(transcriptome.var.index[::-1]):
for gene_oi in ["ENSG00000100014"]:
    if models.trained(gene_oi):
        if force or (not regionmultiwindow.scores["scored"].sel_xr(gene_oi).all()):
            regionmultiwindow.score(models, regions=[gene_oi], force=force)
