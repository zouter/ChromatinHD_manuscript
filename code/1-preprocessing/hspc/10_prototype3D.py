#%%
import torch
import numpy as np
import pandas as pd
import chromatinhd as chd

import plotly.express as px

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name
promoter_name, window = "10k10k", np.array([-10000, 10000])

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)

# pick a gene
gene = promoters.index[6]

# fragments.mapping object specifies from cell and gene for each fragment
mapping = fragments.mapping

# fragments.coordinates object specifies cutsites for each fragment
coordinates = fragments.coordinates

# find out the gene_ix for chosen gene
gene_ix = fragments.var.loc[gene]['ix']

# use gene_ix to filter mapping and coordinates
mask = mapping[:,1] == gene_ix
mapping_sub = mapping[mask]
coordinates_sub = coordinates[mask]

# get latent time
latent_time = pd.read_csv(folder_data_preproc / 'MV2_latent_time_myeloid.csv')

# create df
tens = torch.cat((mapping_sub, coordinates_sub), dim=1)
df = pd.DataFrame(tens.numpy())
df.columns = ['cell_ix', 'gene_ix', 'cut_start', 'cut_end']
df['height'] = 1

# join latent time
df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)

# check latent time differences
df_lt = df.drop_duplicates(subset=['latent_time'])
df_lt.sort_values(by='latent_time', ascending=True, inplace=True)
df_lt['diff'] = df_lt['latent_time'] - df_lt['latent_time'].shift(1)
df_lt['diff'].hist(bins=200)

# reshape
df_long = pd.melt(df, id_vars=['cell_ix', 'gene_ix', 'cell', 'latent_time', 'height'], value_vars=['cut_start', 'cut_end'], var_name='cut_type', value_name='position')

#%%
# Create the 3D scatter plot with Plotly Express
fig = px.scatter_3d(df_long, x='position', y='latent_time', z='height', color='cut_type', template='plotly_white')
fig.update_traces(marker={'size': 2})

# Show the plot
fig.show()

# %%
tens = torch.tensor(df_long[['latent_time', 'position']].values)

torch.save(tens, folder_data_preproc / 'tens_lt.pt')

# %%
