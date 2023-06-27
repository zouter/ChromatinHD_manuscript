#%%
import os
import torch
import imageio
import numpy as np
import pandas as pd
import chromatinhd as chd

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# %%
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

promoter_name, window = "10k10k", np.array([-10000, 10000])
folds = pd.read_pickle(folder_data_preproc / "fragments_myeloid" / promoter_name / "folds.pkl")
lengths = [(len(fold['cells_train']), len(fold['cells_validation'])) for fold in folds]
ratios = [train_len / (train_len + val_len) for train_len, val_len in lengths]

def get_files(nbins):
    model_pattern = f"3-lt_continuous_{'_'.join(str(n) for n in nbins)}_fold_"
    files = sorted([file for file in os.listdir(folder_data_preproc) if model_pattern in file])
    return files

nbins1 = (128, 64, 32, )
nbins2 = (128, )

files1 = get_files(nbins1)
files2 = get_files(nbins2)

dfs1 = [pd.read_csv(folder_data_preproc / file, header=None) for file in files1]
dfs2 = [pd.read_csv(folder_data_preproc / file, header=None) for file in files2]

arr1 = np.squeeze(np.stack(dfs1, axis=2), axis=1)
arr2 = np.squeeze(np.stack(dfs2, axis=2), axis=1)

df1 = pd.DataFrame(arr1)
df2 = pd.DataFrame(arr2)

df1 = pd.concat([df1, df1.apply(lambda row: row.describe(), axis=1)], axis=1)
df2 = pd.concat([df2, df2.apply(lambda row: row.describe(), axis=1)], axis=1)

diff = pd.DataFrame(df1.iloc[:, :5] - df2.iloc[:, :5])
diff_mean = pd.DataFrame(diff.mean(axis=1))

# %%
h1 = df1.iloc[:, :5].div(df1['min'], axis=0)
h2 = df2.iloc[:, :5].div(df2['min'], axis=0)

# h1 = np.log10(h1)
# h2 = np.log10(h2)

vmin = np.min([h1.min(), h2.min()])
vmax = np.max([h1.max(), h2.max()])

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im1 = axs[0].imshow(h1, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
im2 = axs[1].imshow(h2, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
axs[0].set_title(str(nbins1))
axs[1].set_title(str(nbins2))
cax = fig.add_axes([1.05, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label('Colorbar')
plt.tight_layout()
plt.show()

# check order of cells during training
# check for uprageulated and downregulated genes

# %%
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.8
bar_color = 'steelblue'
ax.bar(diff_mean.index, diff_mean.iloc[:, 0], width=bar_width, color=bar_color, edgecolor='black')
ax.set_title('Bar Plot for df', fontsize=16)
ax.set_xlabel('Genes', fontsize=12)
ax.set_ylabel('Difference in Likelihood', fontsize=12)
fig.tight_layout()
plt.show()

# %%
sns.ecdfplot(diff.iloc[:, 0])
diff.iloc[:, 0].describe()

#%%