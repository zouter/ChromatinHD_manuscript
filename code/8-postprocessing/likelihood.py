#%%
import os
import numpy as np
import pandas as pd
import chromatinhd as chd
import chromatinhd_manuscript.plot_functions as pf

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
