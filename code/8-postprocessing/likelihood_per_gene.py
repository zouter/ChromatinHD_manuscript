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

nbins = (256, )
nbins = (128, 64, 32, )
nbins = (128, )
model_pattern = f"3-lt_continuous_{'_'.join(str(n) for n in nbins)}_fold_"
models = sorted([file for file in os.listdir('.') if model_pattern in file])
models_name = [x.replace('3-lt_continuous_', '').replace('.pkl', '') for x in models]

l_1 = pd.read_csv(folder_data_preproc / '3-lt_continuous_128_64_32_likelihood_per_gene.csv', header=None)
l_2 = pd.read_csv(folder_data_preproc / '3-lt_continuous_128_likelihood_per_gene.csv', header=None)

# %%
sum(l_1 == l_2) # should be 0

# %%
df = l_1 - l_2

# %%
fig, ax = plt.subplots(figsize=(8, 6))

bar_width = 0.8
bar_color = 'steelblue'
ax.bar(df.index, df.iloc[:, 0], width=bar_width, color=bar_color, edgecolor='black')

ax.set_title('Bar Plot for df', fontsize=16)
ax.set_xlabel('Genes', fontsize=12)
ax.set_ylabel('Difference in Likelihood', fontsize=12)

fig.tight_layout()
# plt.savefig('bar_plot.png', dpi=300)
plt.show()

# %%
sns.ecdfplot(df.iloc[:, 0])
# %%
df.iloc[:, 0].describe()