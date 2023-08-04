#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import torch
import numpy as np
import pandas as pd
import chromatinhd as chd
import chromatinhd_manuscript.plot_functions as pf

import matplotlib.pyplot as plt

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"
dataset_name = "simulated"
dataset_name = "myeloid"
fragment_dir = folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}"
df_latent_file = folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv"

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

fragments = chd.data.Fragments(fragment_dir / promoter_name)
fragments.window = window
fragments.create_cut_data()

# %%
coordinates = fragments.coordinates
coordinates = coordinates + 10000
coordinates = coordinates / 20000

mapping = fragments.mapping
mapping_cutsites = torch.bincount(mapping[:, 1]) * 2

#%%
# calculate the range that contains 90% of the data
sorted_tensor, _ = torch.sort(mapping_cutsites)
ten_percent = mapping_cutsites.numel() // 10
min_val, max_val = sorted_tensor[ten_percent], sorted_tensor[-ten_percent]

values, bins, _ = plt.hist(mapping_cutsites.numpy(), bins=50, color="blue", alpha=0.75)
percentages = values / np.sum(values) * 100

import seaborn as sns
sns.set_style("white")
sns.set_context("paper", font_scale=1.4)
fig, ax = plt.subplots(dpi=300)
ax.bar(bins[:-1], percentages, width=np.diff(bins), color="blue", alpha=0.75)
ax.set_title("Percentage of values per bin")
ax.set_xlabel("Number of cut sites")
ax.set_ylabel("%")
ax.axvline(min_val, color='r', linestyle='--')
ax.axvline(max_val, color='r', linestyle='--')

sns.despine()

# fig.savefig(folder_data_preproc / f'plots/n_cutsites.png')

# %%
