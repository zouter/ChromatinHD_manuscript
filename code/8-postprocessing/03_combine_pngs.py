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
print('Start of combine_pngs.py')
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
specs = pickle.load(open(folder_root.parent / "code/8-postprocessing/specs.pkl", "rb"))

dataset_name = specs['dataset_name']
nbins = specs['nbins']

# dirs, only do it for fold 0
dir1 = folder_data_preproc / f'plots/cutsites_{dataset_name}'
dir2 = folder_data_preproc / f'plots/likelihood_continuous_{dataset_name}_{"_".join(str(n) for n in nbins)}_fold_0'
dir3 = folder_data_preproc / f'plots/likelihood_continuous_{dataset_name}_{"_".join(str(n) for n in nbins)}_fold_0_minmax'
dir_out = folder_data_preproc / f'plots/likelihood_continuous_{dataset_name}_{"_".join(str(n) for n in nbins)}_fold_0_combined'

dirs = [dir1, dir2, dir3]

# %%
pf.combine_rows_cols(dirs, dir_out, 1, 3)

print('End of combine_pngs.py')

# %%