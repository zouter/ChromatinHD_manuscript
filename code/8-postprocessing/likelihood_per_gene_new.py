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
files = [file for file in files if 'latent_time' not in file]
files = [file for file in files if '_pr_' not in file]

#%%
df_dict = {}
for key in files:
    df_temp = pd.read_csv(data_dir / key, header=None)
    df_temp = df_temp.rename(columns={df_temp.columns[0]: key.replace('_likelihood_per_gene.csv', '')})
    df_dict[key] = df_temp

#%%
df_main = pd.concat(df_dict.values(), axis=1)

#%%
lineage = 'myeloid'

cols = [col for col in df_main.columns if lineage in col]

df_lin = df_main[cols]

baseline_model = f'MV2_{lineage}_linear_128_fold'

# find 'baseline' columns
baseline_cols = [col for col in df_lin.columns if baseline_model in col]
# find 'dependent' columns
dependent_cols = [[col for col in df_lin.columns if baseline_model[-6:] in col] for baseline_model in baseline_cols]

#%%
df_lin_mod = df_lin.copy()
for index, bc in enumerate(baseline_cols):
    print( index, bc)
    for dc in dependent_cols[index]:
        print(dc)
        df_lin_mod[dc] = df_lin_mod[dc] - df_lin[bc]

#%%
patterns = ['linear_128_fold', 'linear_128_64_32_fold', 'sigmoid_128_fold', 'sigmoid_128_64_32_fold']

cols_ordered = []
for pat in patterns:
    cols = [col for col in df_lin_mod.columns if pat in col]
    cols = sorted(cols)
    cols_ordered.append(cols)

# %%
cols_final = []
for x in range(len(cols_ordered[0])):
    for sublist in cols_ordered:
        cols_final.append(sublist[x])
# %%
df_lin_mod = df_lin_mod[cols_final]

df_lin_mod = df_lin_mod.clip(-100, 100)

# %%
xlabels = [x.replace(f'{dataset_name_sub}_{lineage}_', '') for x in df_lin_mod.columns]

plt.imshow(df_lin_mod, cmap='RdBu', interpolation='none', aspect='auto')
plt.colorbar()
plt.title("Heatmap Example")
plt.xlabel("Model")
plt.xticks(ticks=range(len(df_lin_mod.columns)), labels=xlabels, rotation=270)
plt.ylabel("Genes")
plt.show()

# %%
# different aggregation 
df_lin_median = df_lin.median(axis=0).to_frame()
df_lin_median.reset_index(level=0, inplace=True)
df_lin_median.columns = ['model', 'median']

for x in enumerate(patterns):
    print(x)
    df_lin_median.loc[df_lin_median['model'].str.contains(x[1]), 'order'] = x[0]

# first extract fold number and store it in a new column
df_lin_median['fold'] = df_lin_median['model'].str.extract(r'fold_(\d+)')
df_lin_median['fold'] = pd.to_numeric(df_lin_median['fold'], errors='coerce')
df_lin_median['model'] = df_lin_median['model'].str.replace('_fold_\d+', '', regex=True)
df_lin_median = df_lin_median.sort_values(by=['order'])

# %%
plt.scatter(df_lin_median['model'], df_lin_median['median'], c=df_lin_median['fold'], cmap='viridis')
plt.xticks(rotation=270)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Scatter plot with categorical x-axis')
plt.show()

# %%
