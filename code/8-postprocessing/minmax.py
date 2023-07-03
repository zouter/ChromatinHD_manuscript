#%%
import os
import torch
import numpy as np
import pandas as pd
import chromatinhd as chd

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

# %%
def find_local_minima_maxima(tensor):
    rows, cols = tensor.shape
    local_minima = []
    local_maxima = []
    minis = []
    maxis = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = tensor[i-1:i+2, j-1:j+2].flatten()
            center = tensor[i, j]
            if np.all(center <= neighbors):
                local_minima.append((i, j))
                minis.append(tensor[i, j].item())
            elif np.all(center >= neighbors):
                local_maxima.append((i, j))
                maxis.append(tensor[i, j].item())

    return local_minima, local_maxima, minis, maxis

def plot_minmax(tensor, minima, maxima, minis, maxis, gene, plot_dir):
    try:
        minima_y, minima_x = zip(*minima)
        minima_y = np.array(minima_y) / tensor.shape[0]
        minima_x = np.array(minima_x) / tensor.shape[1]
        # minis = np.array(minis)
        # minis = (minis - minis.min()) / (minis.max() - minis.min())
    except ValueError:
        minima_y, minima_x = [], []
        # minis = []

    try:
        maxima_y, maxima_x = zip(*maxima)
        maxima_y = np.array(maxima_y) / tensor.shape[0]
        maxima_x = np.array(maxima_x) / tensor.shape[1]
        # maxis = np.array(maxis)
        # maxis = (maxis - maxis.min()) / (maxis.max() - maxis.min())
    except ValueError:
        maxima_y, maxima_x = [], []
        # maxis = []

    file_name = plot_dir / f"{gene}.png"

    plt.figure(figsize=(5, 5))
    plt.scatter(minima_x, minima_y, c='red', s=1, linewidths=0.5, label='Minima')
    plt.scatter(maxima_x, maxima_y, c='blue', s=1, linewidths=0.5, label='Maxima')
    plt.title(str(plot_dir).split('/')[-1])
    plt.xlabel('Positions')
    plt.ylabel('Latent Time')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(file_name, dpi=200)

#%%
# find all dir that have suffix '_minmax' in dir 'plots'
dirs_out = sorted([file for file in os.listdir(folder_data_preproc / 'plots') if '_minmax' in file])
# remove suffix '_minmax'
dirs_out = [file.replace('_minmax', '') for file in dirs_out]

# find all dir that have prefix 'likelihood_continuous'
dirs_in = sorted([file for file in os.listdir(folder_data_preproc) if 'likelihood_continuous' in file and '_vs_' not in file])
# remove all items in dirs_out from dirs_in
dirs_in = [file for file in dirs_in if file not in dirs_out]

#%%
for data_dir in dirs_in:
    print(f"{data_dir=}")
    # data_dir = 'likelihood_continuous_simulated_128_64_32_fold_0'
    csv_dir = folder_data_preproc / data_dir
    plot_dir = folder_data_preproc / "plots" / f"{data_dir}_minmax"
    os.makedirs(plot_dir, exist_ok=True)

    # list all files in csv_dir
    files = sorted([file for file in os.listdir(csv_dir) if '.csv' in file])

    for file in files:
        print(f"{file=}")
        gene = file.replace('.csv', '')
        tensor = np.loadtxt(csv_dir / file, delimiter=',')
        minima, maxima, minis, maxis = find_local_minima_maxima(tensor)
        # print(f"{minima=}")
        # print(f"{maxima=}")
        plot_minmax(tensor, minima, maxima, minis, maxis, gene, plot_dir)

print('End of script')
# %%
# def plot_minmax(tensor, minima, maxima, gene, plot_dir):
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Plot local minima as rectangles
#     for m in minima:
#         rect = patches.Rectangle((m[1] - 0.5, m[0] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
#         ax.add_patch(rect)

#     # Plot local maxima as rectangles
#     for m in maxima:
#         rect = patches.Rectangle((m[1] - 0.5, m[0] - 0.5), 1, 1, linewidth=1, edgecolor='b', facecolor='b', alpha=0.5)
#         ax.add_patch(rect)

#     # Set the limits based on the tensor shape
#     ax.set_xlim(0, tensor.shape[1])
#     ax.set_ylim(0, tensor.shape[0])

#     ax.set_title('Local Minima and Maxima')
#     ax.set_xlabel('Positions')
#     ax.set_ylabel('Latent Time')

#     # Add legend below the plot
#     minima_patch = patches.Patch(color='r', label='Local Minima')
#     maxima_patch = patches.Patch(color='b', label='Local Maxima')
#     ax.legend(handles=[minima_patch, maxima_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
#     file_name = plot_dir / f"{gene}.png"
#     plt.savefig(file_name, dpi=200)