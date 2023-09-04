#%%
import os
import numpy as np
import pandas as pd
import chromatinhd as chd

import seaborn as sns
import matplotlib.pyplot as plt

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

data_dir = folder_data_preproc / f"{dataset_name_sub}_LC_gene"

files = sorted(os.listdir(data_dir))
models = set([file.split('_fold')[0] for file in files if 'latent_time' not in file])

#%%
file_dict = {}
for key in models:
    file_dict[key] = [file for file in files if file.startswith(key)]

df_dict = {}
for key in file_dict.keys():
    df_dict[key] = [pd.read_csv(data_dir / file, header=None) for file in file_dict[key]]

df_dict_mean = {}
for key in df_dict.keys():
    df_dict_mean[key] = pd.concat([df.mean(axis=1) for df in df_dict[key]], axis=1).mean(axis=1)

#%%
# Rename the columns and store the modified dataframes in a list
modified_dfs = [df.rename(columns={df.columns[0]: key}) for key, df in df_dict.items()]

# Concatenate the dataframes by columns
result_df = pd.concat(modified_dfs, axis=1)

#%%
lineage = 'myeloid'

key1 = f'MV2_{lineage}_sigmoid_128_64_32'
key2 = f'MV2_{lineage}_linear_128_64_32'

key1 = f'MV2_{lineage}_sigmoid_128_64_32'
key2 = f'MV2_{lineage}_sigmoid_128'

key1 = f'MV2_{lineage}_linear_128_64_32'
key2 = f'MV2_{lineage}_linear_128'

# df_dict_mean[key1].describe()
# df_dict_mean[key2].describe()

plt.scatter(df_dict_mean[key1], df_dict_mean[key2], color='blue') 
plt.show()

diff = df_dict_mean[key1] - df_dict_mean[key2]
diff = diff.sort_values().reset_index(drop=True)
# diff.describe()

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.8
ax.bar(diff.index, diff, width=bar_width, edgecolor='black')
ax.set_title(f"{key1} vs {key2}", fontsize=12)
ax.set_xlabel('Genes', fontsize=12)
ax.set_ylabel('Difference in Likelihood', fontsize=12)
fig.tight_layout()
plt.show()

#%%