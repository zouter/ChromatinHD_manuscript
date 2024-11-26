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
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
dataset_name = "pbmc10k"
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "10k10k")
regions = fragments.regions

transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
folds = chd.data.folds.Folds(dataset_folder / "folds" / "random_5fold")

# %%
cellxregion_ix = fragments.mapping[:, 0] * fragments.n_regions + fragments.mapping[:, 1]
counts = np.bincount(cellxregion_ix, minlength = fragments.n_cells * fragments.n_regions).reshape(fragments.n_cells, fragments.n_regions)
sparsity = np.mean(counts == 0, axis = 0)
plt.hist(sparsity)

# %%
expression = transcriptome.layers["normalized"][:]
idealcors = []
randomcors = []
randomzmse = []
for i in range(50):
    predicted = (expression) * (np.random.uniform(0, 1, expression.shape) > sparsity)
    idealcors.append(chd.utils.paircor(expression, predicted))
    randomcors.append(chd.utils.paircor(expression, np.random.normal(size = expression.shape)))
    randomzmse.append(chd.utils.pairzmse(expression, predicted))
randomcors = np.array(randomcors)
idealcors = np.array(idealcors)
randomzmse = np.array(randomzmse)

# %%
randomcors = np.array(randomcors)
randomcors_mean = randomcors.mean(0)
randomr2_mean = (randomcors**2).mean(0)

randomzmse = np.array(randomzmse)
randomzmse_mean = randomzmse.mean(0)

idealcors = np.array(idealcors)
idealcors_mean = idealcors.mean(0)
idealr2_mean = (idealcors**2).mean(0)

# %%
transcriptome.var["sparsity"] = sparsity

# %%
plt.scatter(sparsity, idealcors_mean)
plt.scatter(sparsity, randomcors_mean)

# %%
sparsity_expression = (transcriptome.layers["normalized"][:] == 0).mean(0)

# %%
expression = counts
idealcors = []
randomcors = []
randomzmse = []
for i in range(3):
    predicted = (expression) * (np.random.uniform(0, 1, expression.shape) > sparsity_expression)
    idealcors.append(chd.utils.paircor(expression, predicted))
idealcors = np.array(idealcors)

# %%
idealcors = np.array(idealcors)
idealcors_mean = idealcors.mean(0)
idealr2_mean = (idealcors**2).mean(0)

# %%
sns.ecdfplot(idealcors_mean)

# %%
plt.scatter(sparsity_expression, idealr2_mean)
