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
folder_data = folder_root / "data"
dataset_name = "hspc_backup"
folder_data_preproc = folder_data / dataset_name
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

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
            if np.all(center < neighbors):
                local_minima.append((i, j))
                minis.append(tensor[i, j].item())
            elif np.all(center > neighbors):
                local_maxima.append((i, j))
                maxis.append(tensor[i, j].item())

    return local_minima, local_maxima, minis, maxis

def plot_minmax(tensor, minima, maxima, minis, maxis, gene, plot_dir):
    minima_y, minima_x = zip(*minima)
    maxima_y, maxima_x = zip(*maxima)

    minis = np.array(minis)
    maxis = np.array(maxis)
    minis = (minis - minis.min()) / (minis.max() - minis.min())
    maxis = (maxis - maxis.min()) / (maxis.max() - maxis.min())

    plt.figure(figsize=(5, 5))
    plt.scatter(minima_x, minima_y, c='red', s=1, linewidths=0.5, label='Minima')
    plt.scatter(maxima_x, maxima_y, c='blue', s=1, linewidths=0.5, label='Maxima')

    plt.title(str(plot_dir).split('/')[-1])
    plt.xlabel('Positions')
    plt.ylabel('Latent Time')
    plt.show()

    file_name = plot_dir / f"{gene}.png"
    plt.savefig(file_name, dpi=200)

csv_dir = folder_data_preproc / "likelihood_continuous_128_64_32_fold_0"
plot_dir = folder_data_preproc / "plots/likelihood_continuous_128_64_32_fold_0_>_not_>="
os.makedirs(plot_dir, exist_ok=True)

#%%
for x in range(promoters.shape[0]):
    print(f"{x=}")
    gene = promoters.index[x]
    tensor = np.loadtxt(csv_dir / f"{gene}.csv", delimiter=',')
    minima, maxima, minis, maxis = find_local_minima_maxima(tensor)
    print(f"{minima=}")
    print(f"{maxima=}")
    try:
        plot_minmax(tensor, minima, maxima, minis, maxis, gene, plot_dir)
    except ValueError:
        continue

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