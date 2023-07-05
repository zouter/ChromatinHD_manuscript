#%%
import os
import torch
import numpy as np
import pandas as pd
import chromatinhd as chd
import chromatinhd_manuscript.plot_functions as pf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %%
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc_backup"
dataset_name = "myeloid"
dataset_name = "simulated"

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_file = promoter_name + "_simulated" if dataset_name == "simulated" else promoter_name
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_file + ".csv"), index_col = 0)

#%%
# find all dir that have suffix '_minmax' in dir 'plots'
dirs_out = sorted([file for file in os.listdir(folder_data_preproc / 'plots') if '_minmax' in file])
# remove suffix '_minmax'
dirs_out = [file.replace('_minmax', '') for file in dirs_out]

# find all dir that have prefix 'likelihood_continuous'
dirs_in = sorted([file for file in os.listdir(folder_data_preproc) if 'likelihood_continuous' in file and '_vs_' not in file])
# remove all items in dirs_out from dirs_in
dirs_in = [file for file in dirs_in if file not in dirs_out]

def find_minmax(file):
    tensor = np.loadtxt(file, delimiter=',')

    rows, cols = tensor.shape
    minima = []
    maxima = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = tensor[i-1:i+2, j-1:j+2].flatten()
            center = tensor[i, j]
            if np.all(center <= neighbors):
                minima.append((i, j))
            elif np.all(center >= neighbors):
                maxima.append((i, j))

    minima_df = pd.DataFrame(minima, columns=['y', 'x'])
    minima_df['type'] = 'minima'

    maxima_df = pd.DataFrame(maxima, columns=['y', 'x'])
    maxima_df['type'] = 'maxima'

    result_df = pd.concat([minima_df, maxima_df], ignore_index=True)
    result_df['y'] /= rows
    result_df['x'] /= cols

    return result_df

#%%
dirs_in = ['likelihood_continuous_128_fold_0']

for dir_sub in dirs_in:
    dir_csv = folder_data_preproc / dir_sub
    dir_plot = folder_data_preproc / "plots" / f"{dir_sub}_minmax"
    os.makedirs(dir_plot, exist_ok=True)

    files = sorted([file for file in os.listdir(dir_csv) if '.csv' in file])
    for file in files:
        gene = file.replace('.csv', '')
        filename = dir_csv / file
        df_minmax = find_minmax(filename)
        pf.minmax(gene, df_minmax, dir_plot, show=False)

        print(dir_sub, gene)

print('End of script')

# def minmax(file):
#     tensor = np.loadtxt(file, delimiter=',')

#     rows, cols = tensor.shape
#     minima = []
#     maxima = []
#     minis = []
#     maxis = []

#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#             neighbors = tensor[i-1:i+2, j-1:j+2].flatten()
#             center = tensor[i, j]
#             if np.all(center <= neighbors):
#                 minima.append((i, j))
#                 minis.append(tensor[i, j].item())
#             elif np.all(center >= neighbors):
#                 maxima.append((i, j))
#                 maxis.append(tensor[i, j].item())

#     try:
#         minima_y, minima_x = zip(*minima)
#         minima_y = np.array(minima_y) / rows
#         minima_x = np.array(minima_x) / cols
#     except ValueError:
#         minima_y, minima_x = [], []

#     try:
#         maxima_y, maxima_x = zip(*maxima)
#         maxima_y = np.array(maxima_y) / rows
#         maxima_x = np.array(maxima_x) / cols
#     except ValueError:
#         maxima_y, maxima_x = [], []

#     # Ensure arrays have the same length
#     max_length = max(len(minima_y), len(minima_x), len(maxima_y), len(maxima_x))
#     minima_y = np.pad(minima_y, (0, max_length - len(minima_y)))
#     minima_x = np.pad(minima_x, (0, max_length - len(minima_x)))
#     maxima_y = np.pad(maxima_y, (0, max_length - len(maxima_y)))
#     maxima_x = np.pad(maxima_x, (0, max_length - len(maxima_x)))
    
#     data = {
#         'minima_y': minima_y,
#         'minima_x': minima_x,
#         'maxima_y': maxima_y,
#         'maxima_x': maxima_x
#     }

#     return pd.DataFrame(data)

# %%
# def find_local_minima_maxima(tensor):
#     rows, cols = tensor.shape
#     local_minima = []
#     local_maxima = []
#     minis = []
#     maxis = []

#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#             neighbors = tensor[i-1:i+2, j-1:j+2].flatten()
#             center = tensor[i, j]
#             if np.all(center <= neighbors):
#                 local_minima.append((i, j))
#                 minis.append(tensor[i, j].item())
#             elif np.all(center >= neighbors):
#                 local_maxima.append((i, j))
#                 maxis.append(tensor[i, j].item())

#     return local_minima, local_maxima, minis, maxis

# def minmax(tensor, minima, maxima, gene, plot_dir):
#     try:
#         minima_y, minima_x = zip(*minima)
#         minima_y = np.array(minima_y) / tensor.shape[0]
#         minima_x = np.array(minima_x) / tensor.shape[1]
#     except ValueError:
#         minima_y, minima_x = [], []

#     try:
#         maxima_y, maxima_x = zip(*maxima)
#         maxima_y = np.array(maxima_y) / tensor.shape[0]
#         maxima_x = np.array(maxima_x) / tensor.shape[1]
#     except ValueError:
#         maxima_y, maxima_x = [], []

#     file_name = plot_dir / f"{gene}.png"

#     plt.figure(figsize=(5, 5))
#     plt.scatter(minima_x, minima_y, c='red', s=1, linewidths=0.5, label='Minima')
#     plt.scatter(maxima_x, maxima_y, c='blue', s=1, linewidths=0.5, label='Maxima')
#     plt.title(str(plot_dir).split('/')[-1])
#     plt.xlabel('Positions')
#     plt.ylabel('Latent Time')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.savefig(file_name, dpi=200)