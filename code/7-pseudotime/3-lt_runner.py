# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import sys
import itertools
import subprocess
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

#%%
variables = {
    'dataset_name_sub': ["MV2"],
    'dataset_name': ["myeloid", "erythroid", "platelet"],
    'nbins': [(128, 64, 32,), (128,)],
    'model_type': ['sigmoid', 'linear']
}

combinations = [dict(zip(variables.keys(), values)) for values in itertools.product(*variables.values())]
for arg in combinations:
    print(arg)
    # subprocess.call([sys.executable, '3-lt_continuous_train.py', str(arg), 'external'])
    subprocess.call([sys.executable, '3-lt_continuous_infer.py', str(arg), 'external'])

print("3-lt_runner.py done!")

# %%
