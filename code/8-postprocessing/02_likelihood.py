#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import chromatinhd_manuscript.plot_functions as pf

# %%
print('Start of likelihood.py')
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
specs = pickle.load(open(folder_root.parent / "code/8-postprocessing/specs.pkl", "rb"))

dataset_name = "myeloid"
dataset_name = "simulated"
dataset_name = specs['dataset_name']
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

nbins = specs['nbins']
pattern = f"likelihood_continuous_{dataset_name}_{'_'.join(str(n) for n in nbins)}_fold_"
directories = sorted([file for file in os.listdir(folder_data_preproc) if pattern in file])

#%%
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
for dir_sub in directories:
    dir_csv = folder_data_preproc / dir_sub
    dir_minmax = folder_data_preproc / "plots" / f"{dir_sub}_minmax"
    # dir_likelihood = folder_data_preproc / "plots" / f"{dir_sub}"

    files = sorted([file for file in os.listdir(dir_csv) if '.csv' in file])
    for file in files:
        gene = file.replace('.csv', '')
        filename = dir_csv / file

        pf.model_continuous(gene, dir_csv, show=False)
        df_minmax = find_minmax(filename)
        pf.minmax(gene, df_minmax, dir_minmax, show=False)

        print(dir_sub, gene)

print('End of likelihood.py')

# %%
