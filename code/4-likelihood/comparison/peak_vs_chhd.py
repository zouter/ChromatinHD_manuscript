import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdmpbmc10k_eqtl

import chromatinhd as chd
import tempfile


folder_root = chd.get_output()
folder_data = folder_root / "data"

from chromatinhd_manuscript.designs import dataset_latent_peakcaller_diffexp_method_motifscan_enricher_combinations as design
promoter_name = "10k10k"
def get_score_folder(x):
    return chd.get_output() / "prediction_likelihood" / x.dataset / promoter_name / x.latent / str(x.method) / "scoring" / x.peakcaller / x.diffexp / x.motifscan / x.enricher
design["score_folder"] = design.apply(get_score_folder, axis = 1)


design = 