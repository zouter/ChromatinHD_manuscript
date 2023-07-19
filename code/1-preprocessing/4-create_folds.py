# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "GSE198467_H3K27ac"
# dataset_name = "GSE198467_single_modality_H3K27me3"
# dataset_name = "hspc"
# dataset_name = "pbmc10k_eqtl"

folder_data_preproc = folder_data / dataset_name

# %%
promoter_name = "10k10k"
# promoter_name = "100k100k"

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
(fragments.path / "folds").mkdir(exist_ok=True)

# %% [markdown]
# ### Random cells with train/validation/test

# %%
n_folds = 5
n_folds = 20

# %%
# train/test split
cells_all = np.arange(fragments.n_cells)

cell_bins = np.floor((np.arange(len(cells_all)) / (len(cells_all) / n_folds)))

folds = []
for i in range(n_folds):
    cells_train = cells_all[cell_bins != i]
    cells_validation_test = cells_all[cell_bins == i]
    cells_validation = cells_validation_test[: (len(cells_validation_test) // 2)]
    cells_test = cells_validation_test[(len(cells_validation_test) // 2) :]

    folds.append(
        {
            "cells_train": cells_train,
            "cells_validation": cells_validation,
            "cells_test": cells_test,
        }
    )
pickle.dump(
    folds,
    (fragments.path / "folds" / ("random_" + str(n_folds) + "fold.pkl")).open("wb"),
)
if n_folds == 5:
    pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %% [markdown]
# ### Random cells with train/validation/test but with constant percentage of cells being validation/test

# %%
n_folds = 5
n_repeats = 5

# %%
# train/test split
folds = []

for repeat_ix in range(n_repeats):
    generator = np.random.RandomState(repeat_ix)

    cells_all = generator.permutation(fragments.n_cells)

    cell_bins = np.floor((np.arange(len(cells_all)) / (len(cells_all) / n_folds)))

    for i in range(n_folds):
        cells_train = cells_all[cell_bins != i]
        cells_validation_test = cells_all[cell_bins == i]
        cells_validation = cells_validation_test[: (len(cells_validation_test) // 2)]
        cells_test = cells_validation_test[(len(cells_validation_test) // 2) :]

        folds.append(
            {
                "cells_train": cells_train,
                "cells_validation": cells_validation,
                "cells_test": cells_test,
            }
        )
pickle.dump(
    folds,
    (
        fragments.path
        / "folds"
        / ("permutations_" + str(n_folds) + "fold" + str(n_repeats) + "repeat.pkl")
    ).open("wb"),
)
print(("permutations_" + str(n_folds) + "fold" + str(n_repeats) + "repeat"))

# %% [markdown]
# ### Based on a latent

# %%
# latent_name = "leiden_0.1"
latent_name = "celltype"
latent_name = "overexpression"

latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

# %%
folds = []
perc_validation = 0.2
for latent_dimension in latent.columns:
    cells_train_validation = np.where(1 - latent[latent_dimension])[0]
    cells_test = np.where(latent[latent_dimension])[0]

    split = int(len(cells_train_validation) * perc_validation)
    cells_validation = cells_train_validation[:split]
    cells_train = cells_train_validation[split:]

    print(len(cells_train), len(cells_validation), len(cells_test))

    folds.append(
        {
            "cells_train": cells_train,
            "cells_validation": cells_validation,
            "cells_test": cells_test,
        }
    )
pickle.dump(folds, (fragments.path / "folds" / (latent_name + ".pkl")).open("wb"))

# %% [markdown]
# ### All

# %%
n_bins = 1
splitter = "all"
folds = [{"cells_test": np.arange(fragments.n_cells)}]
pickle.dump(folds, (fragments.path / "folds" / (splitter + ".pkl")).open("wb"))

# %% [markdown]
# ### All train

# %%
n_bins = 1
splitter = "all_train"

cells_train_validation = np.arange(fragments.n_cells)

validation_perc = 0.2
split = int(len(cells_train_validation) * perc_validation)
cells_validation = cells_train_validation[:split]
cells_train = cells_train_validation[split:]

folds = [{"cells_train": cells_train, "cells_validation": cells_validation}]
pickle.dump(folds, (fragments.path / "folds" / (splitter + ".pkl")).open("wb"))

# %%

# %%

# %%

# %%

# %%
