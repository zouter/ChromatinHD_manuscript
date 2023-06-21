#%%
import os
import torch
import imageio
import numpy as np
import pandas as pd
import chromatinhd as chd

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# %%
# set folder paths
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name
promoter_name, window = "10k10k", np.array([-10000, 10000])

#%%
dir_model_continuous = folder_data_preproc / "evaluate_pseudo_continuous_tensors"
dir_model_quantile = folder_data_preproc / "quantile_time_evaluate_pseudo"
# dir_plots_continuous = folder_data_preproc / "plots/evaluate_pseudo_continuous"
# dir_plots_cutsites_continuous = folder_data_preproc / "plots/cut_sites_evaluate_pseudo_continuous"

global_vars = globals()
dirs = {key: value for key, value in global_vars.items() if key.startswith('dir_')}

# %%
files_dict = {key: [filename.split('.')[0] for filename in sorted(os.listdir(directory))] for key, directory in dirs.items()}
n_files_dict = {key: len(directory) for key, directory in files_dict.items()}

# %%
comparison_dict = {}
for key1, files_list1 in files_dict.items():
    for key2, files_list2 in files_dict.items():
        if key1 != key2:
            combination_key = f"{key1} vs {key2}"
            different_items = list(set(files_list1) ^ set(files_list2))
            comparison_dict[combination_key] = different_items
# %%
print(n_files_dict)
print(comparison_dict)
n_comparison_dict = {key: len(value) for key, value in comparison_dict.items()}
print(n_comparison_dict)

# %%
