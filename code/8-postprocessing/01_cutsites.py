#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import torch
import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import chromatinhd_manuscript.plot_functions as pf

# %%
# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
specs = pickle.load(open(folder_root.parent / "code/8-postprocessing/specs.pkl", "rb"))
dataset_name = specs['dataset_name']
dataset_name_sub = "MV2"
fragment_dir = folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}"
df_latent_file = folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv"

# load data
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
s_genes = info_genes_cells['s_genes'].dropna().tolist()
g2m_genes = info_genes_cells['g2m_genes'].dropna().tolist()
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()

# fragments and latent time are for 2-lt_discrete
fragments = chd.data.Fragments(fragment_dir / promoter_name)
fragments.window = window
fragments.create_cut_data()

latent_time = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv")
latent_time['rank_raw'] = latent_time['latent_time'].rank()
latent_time['rank'] = latent_time['rank_raw'] / latent_time.shape[0]

# %%
coordinates = fragments.coordinates
coordinates = (coordinates + 10000) / 20000

mapping = fragments.mapping

#%%
for x in range(promoters.shape[0]):
    gene = promoters.index[x]
    gene_ix = fragments.var.loc[gene]['ix']
    print(x, gene, gene_ix)

    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]

    tens1 = torch.cat((coordinates_sub[:, 0].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens2 = torch.cat((coordinates_sub[:, 1].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens = torch.cat((tens1, tens2), dim=0)

    df = pd.DataFrame(tens.numpy())
    df.columns = ['x', 'cell_ix']
    df = pd.merge(df, latent_time, left_on='cell_ix', right_index=True)
    df = df.rename(columns={'rank': 'y'})
    
    directory = folder_data_preproc / 'plots' / f"cutsites_{dataset_name}"
    pf.cutsites(gene, df, directory, show=False)