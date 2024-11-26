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

# %% [markdown]
# # Preprocess

# %%
import polyptich as pp
pp.setup_ipython()

# %%
import chromatinhd as chd

# %%
# for dataset_name in ["pbmc10k", "hspc", "e18brain", "lymphoma", "pbmc10k_gran", "pbmc10kx", "liver", "hepatocytes", "pbmc3k-pbmc10k", "lymphoma-pbmc10k", "pbmc10k_gran-pbmc10k", "pbmc10kx-pbmc10k", "hspc_cycling", "hspc_meg_cycling", "hspc_gmp_cycling"]:
for dataset_name in ["liverkia_lsecs"]:
    dataset_folder = chd.get_output() / "datasets" / dataset_name
    fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")
    folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x1", reset = True)
    folds.sample_cells(fragments, 5)

    folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x5", reset = True)
    folds.sample_cells(fragments, 5, 5)

# %%
