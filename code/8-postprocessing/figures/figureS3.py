# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import torch
import pickle
import itertools
import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
hspc_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

#%%
def create_color_gradient(df, input_column, color_gradient):
    df['gradient'] = np.nan 
    df.loc[df[input_column].isna(), 'gradient'] = 'lightgray'

    min_value = df[input_column].min()
    max_value = df[input_column].max()
    df.loc[df[input_column].notna(), 'gradient'] = \
        df.loc[df[input_column].notna(), input_column].apply(
            lambda x: plt.cm.get_cmap(color_gradient)((x - min_value) / (max_value - min_value))
        )
    
    return df['gradient']

def lt_vs_rank(df_latent):
    sub = df_latent.copy()
    sub = sub.sort_values(by=f'latent_time')

    sub['lt_shift'] = sub[f'latent_time'].diff()
    sub.loc[sub.index[0], 'lt_shift'] = sub.loc[sub.index[1], f'latent_time']

    sub['diff'] = sub['lt_shift'] - sub['rank'].iloc[1]
    sub = sub.sort_values(by='diff')
    sub['diff_rank'] = sub['diff'].rank() - 1

    count_negative = sub[sub['diff'] < 0]['diff'].count()
    count_positive = sub[sub['diff'] >= 0]['diff'].count()

    sub.loc[sub['diff'] < 0, 'diff_rank'] = (sub.loc[sub['diff'] < 0, 'diff_rank'] / count_negative) * 0.5
    sub.loc[sub['diff'] >= 0, 'diff_rank'] = (sub.loc[sub['diff'] >= 0, 'diff_rank'] - count_negative) / count_positive * 0.5 + 0.5

    sub['diff_rank_color'] = create_color_gradient(sub, 'diff_rank', 'RdBu')

    sub['y1'] = 0
    sub['y2'] = 1

    # switch x and y depending on if the plot should be vertical or horizontal
    y_values = np.vstack((sub[f'latent_time'], sub['rank']))
    x_values = np.vstack((sub['y1'], sub['y2']))
    colors = sub['diff_rank_color'].to_numpy()

    return x_values, y_values, colors

lineage_gene = {'myeloid': 'MPO', 'erythroid': 'HBB', 'platelet': 'CD74'}

for dataset_name, gene_name in lineage_gene.items():

    fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}/10k10k")
    fragments.window = window
    fragments.create_cut_data()
    gene_id = adata.var.loc[gene_name]['Accession']
    gene_ix = fragments.var.index.get_loc(gene_id)

    df_latent = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv", index_col = 0)
    df_latent['rank'] = (df_latent['latent_time'].rank() - 1) / (len(df_latent) - 1)

    mapping = fragments.mapping
    coordinates = fragments.coordinates
    coordinates = (coordinates + 10000) / 20000

    mask = mapping[:,1] == gene_ix
    mapping_sub = mapping[mask]
    coordinates_sub = coordinates[mask]

    tens1 = torch.cat((coordinates_sub[:, 0].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens2 = torch.cat((coordinates_sub[:, 1].unsqueeze(1), mapping_sub[:, 0].unsqueeze(1)), dim=1)
    tens = torch.cat((tens1, tens2), dim=0)

    df_latent['cell_ix'] = range(df_latent.shape[0])
    df_cutsites = pd.DataFrame(tens.numpy())
    df_cutsites.columns = ['x', 'cell_ix']
    df_cutsites = pd.merge(df_cutsites, df_latent, left_on='cell_ix', right_on='cell_ix')

    df_cutsites_lt = df_cutsites.rename(columns={'latent_time': 'y'})
    df_cutsites_pr = df_cutsites.rename(columns={'rank': 'y'})

    lineage_gene[dataset_name] = {
        'gene_name': gene_name,
        'df_latent': df_latent,
        'df_cutsites_lt': df_cutsites_lt[['x', 'y']],
        'df_cutsites_pr': df_cutsites_pr[['x', 'y']],
    }

#%%
def plot_cutsites(fig, gridspec, df, title):
    ax_object = fig.add_subplot(gridspec)
    ax_object.scatter(df['x'], df['y'], s=0.3, edgecolors='none', color='black')
    ax_object.set_xlim([0, 1])
    ax_object.set_ylim([0, 1])
    tick_positions = np.linspace(0, 1, 5)
    tick_labels = ['-10k', '-5k', '0', '5k', '10k']
    ax_object.set_xticks(tick_positions, tick_labels)
    ax_object.set_title(title)
    return ax_object

def plot_lt_vs_rank(fig, gridspec, df):
    x_values, y_values, colors = lt_vs_rank(df)
    ax_object = fig.add_subplot(gridspec)
    for i in range(len(x_values[0])):
        ax_object.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)
    ax_object.set_ylim(0, 1)
    ax_object.xaxis.set_visible(False)
    ax_object.yaxis.set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['bottom'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    return ax_object

# Create the grid layout
height, width = 17, 12
fig = plt.figure(figsize=(width, height))

rows, cols = 17, 12
grid = GridSpec(rows, cols, figure=fig)

lineages = [x for x in info_genes_cells.columns if 'lin' in x]
df = pd.DataFrame(lineages, columns=['lineage'])
df['group'] = df['lineage'].map({lineage: i for i, lineage in enumerate(lineages)})
df['index'] = df.groupby('group').cumcount()
df['plot_row'] = (df['group'] * 5) + 1 

col_1 = [grid[i:i+4, 1:5] for i in df['plot_row']]
col_2 = [grid[i:i+4, 5:8] for i in df['plot_row']]
col_3 = [grid[i:i+4, 8:12] for i in df['plot_row']]

for lin, c1, c2, c3 in zip(df['lineage'], col_1, col_2, col_3):
    lineage = lin.split('_')[1]
    gene_name = lineage_gene[lineage]['gene_name']
    df_cutsites_lt = lineage_gene[lineage]['df_cutsites_lt']
    df_cutsites_pr = lineage_gene[lineage]['df_cutsites_pr']

    ax1 = plot_cutsites(fig, c1, df_cutsites_lt, f"{gene_name} (latent time)")
    ax2 = plot_lt_vs_rank(fig, c2, lineage_gene[lineage]['df_latent'])
    ax3 = plot_cutsites(fig, c3, df_cutsites_pr, f"{gene_name} (percent rank)")
    # for ax3 show ticks and labels on the right side
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")


fig.savefig(folder_data_preproc / 'plots' / f"figS3.pdf", bbox_inches='tight', pad_inches=0.01)

# %%