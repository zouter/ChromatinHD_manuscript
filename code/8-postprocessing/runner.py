#%%
import os
import pickle
import pathlib
import itertools
import subprocess

dataset_name = ["simulated"]
nbins = [(128, 64, 32, ), (128, )]

combinations = [{'dataset_name': name, 'nbins': n} for name, n in itertools.product(dataset_name, nbins)]

directory1 = pathlib.Path(os.getcwd()).parent / "7-pseudotime"
directory2 = pathlib.Path(os.getcwd()).parent / "8-postprocessing"

# %%
for combination in combinations:
    pickle.dump(combination, open("./specs.pkl", "wb"))

    # subprocess.run(['python', directory1 / '3-lt_continuous_train.py'])
    # subprocess.run(['python', directory1 / '3-lt_continuous_infer.py'])
    # subprocess.run(['python', directory2 / 'likelihood.py'])
    subprocess.run(['python', directory2 / 'combine_pngs.py'])

# %%
