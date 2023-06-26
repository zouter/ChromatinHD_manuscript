#%%
import os
import sys
import numpy as np
import pandas as pd
import chromatinhd as chd

# set folder paths
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

dir_1 = folder_data_preproc / sys.argv[1]
dir_2 = folder_data_preproc / sys.argv[2]
dir_out = folder_data_preproc / sys.argv[3]

if not os.path.isdir(dir_1):
    dir_1 = folder_data_preproc / "evaluate_pseudo_continuous_tensors"
    dir_2 = folder_data_preproc / "evaluate_pseudo_continuous_tensors_128"
    dir_out = folder_data_preproc / "diff_data_frames"

os.makedirs(dir_out, exist_ok=True)

# Get the list of CSV files in the directories
files_1 = [file for file in os.listdir(dir_1) if file.endswith(".csv")]
files_2 = [file for file in os.listdir(dir_2) if file.endswith(".csv")]

#%%
# Iterate over the matching CSV files and calculate the difference
for csv_file in set(files_1).intersection(files_2):
    path_1 = dir_1 / csv_file
    path_2 = dir_2 / csv_file

    df_1 = pd.read_csv(path_1, header=None)
    df_2 = pd.read_csv(path_2, header=None)

    df_diff = df_1 - df_2

    path_out = dir_out / csv_file
    df_diff.to_csv(path_out, index=False, header=False)

# %%
