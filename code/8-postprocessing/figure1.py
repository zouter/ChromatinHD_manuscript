# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

#%%
transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")

umap_coords = transcriptome.adata.obsm['X_umap']

obs = transcriptome.adata.obs
obs.index.name = 'cell'
obs['umap1'] = umap_coords[:, 0]
obs['umap2'] = umap_coords[:, 1]

#%%
df_myeloid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_myeloid.csv")
df_erythroid = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_erythroid.csv")
df_platelet = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_platelet.csv")

df_myeloid = df_myeloid.rename(columns={'latent_time': 'latent_time_myeloid'})
df_erythroid = df_erythroid.rename(columns={'latent_time': 'latent_time_erythroid'})
df_platelet = df_platelet.rename(columns={'latent_time': 'latent_time_platelet'})

obs = obs.merge(df_myeloid[['cell', 'latent_time_myeloid']], on='cell', how='left')
obs = obs.merge(df_erythroid[['cell', 'latent_time_erythroid']], on='cell', how='left')
obs = obs.merge(df_platelet[['cell', 'latent_time_platelet']], on='cell', how='left')

#%%
annotation = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

obs['color'] = [annotation[label] for label in obs['celltype']]

#%%
color_gradients = ['inferno', 'RdBu', 'PiYG', 'PRGn', 'PuOr', 'BrBG']

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

obs['color_gradient_myeloid'] = create_color_gradient(obs, 'latent_time_myeloid', 'inferno')
obs['color_gradient_erythroid'] = create_color_gradient(obs, 'latent_time_erythroid', 'inferno')
obs['color_gradient_platelet'] = create_color_gradient(obs, 'latent_time_platelet', 'inferno')

#%%
var = transcriptome.adata.var

genes = ['MPO', 'PF4', 'HBB', 'CD74']
genes_ix = [var.index.get_loc(gene) for gene in genes]
obs[genes] = transcriptome.adata.X[:, genes_ix]

#%%
for gene in genes:
    obs[f'color_gradient_{gene}'] = create_color_gradient(obs, gene, 'RdBu_r')
# reverse the gradient

def lt_vs_rank(obs, lineage):
    sub = obs[[f'color_gradient_{lineage}', f'latent_time_{lineage}']]
    sub = sub[sub[f'latent_time_{lineage}'].notna()]
    sub = sub.sort_values(by=f'latent_time_{lineage}')
    sub['rank'] = sub[f'latent_time_{lineage}'].rank() - 1
    sub['rank'] = sub['rank'] / sub['rank'].max()

    sub['lt_shift'] = sub[f'latent_time_{lineage}'].diff()
    sub.loc[sub.index[0], 'lt_shift'] = sub.loc[sub.index[1], f'latent_time_{lineage}']

    sub['diff'] = sub['lt_shift'] - sub['rank'].iloc[1]
    sub = sub.sort_values(by='diff')
    sub['diff_rank'] = sub['diff'].rank() - 1

    count_negative = sub[sub['diff'] < 0]['diff'].count()
    count_positive = sub[sub['diff'] >= 0]['diff'].count()

    sub.loc[sub['diff'] < 0, 'diff_rank'] = (sub.loc[sub['diff'] < 0, 'diff_rank'] / count_negative) * 0.5
    sub.loc[sub['diff'] >= 0, 'diff_rank'] = (sub.loc[sub['diff'] >= 0, 'diff_rank'] - count_negative) / count_positive * 0.5 + 0.5

    sub['diff_rank_color'] = create_color_gradient(sub, 'diff_rank', 'RdBu')

    sub['y1'] = 1
    sub['y2'] = 0

    x_values = np.vstack((sub[f'latent_time_{lineage}'], sub['rank']))
    y_values = np.vstack((sub['y1'], sub['y2']))
    colors = sub['diff_rank_color'].to_numpy()

    return x_values, y_values, colors

# %%
# Create the grid layout
rows, cols = 68, 90

rowA_y1 = 0
rowA_y2 = 31

rowB_y1 = 37
rowB_y2 = 58

rowC_y1 = 64
rowC_y2 = 68

fig = plt.figure(figsize=(24, 15))
grid = GridSpec(rows, cols, figure=fig)

ax_cellt = fig.add_subplot(grid[rowA_y1:rowA_y2, int((0/4)*cols):int((2/4)*cols)])
ax_gene1 = fig.add_subplot(grid[rowA_y1:int(rowA_y2/2), int((2/4)*cols):int((3/4)*cols)])
ax_gene2 = fig.add_subplot(grid[rowA_y1:int(rowA_y2/2), int((3/4)*cols)+1:int((4/4)*cols)])
ax_gene3 = fig.add_subplot(grid[int(rowA_y2/2)+1:rowA_y2, int((2/4)*cols):int((3/4)*cols)])
ax_gene4 = fig.add_subplot(grid[int(rowA_y2/2)+1:rowA_y2, int((3/4)*cols)+1:int((4/4)*cols)])

ax_mye = fig.add_subplot(grid[rowB_y1:rowB_y2, int((0/3)*cols):int((1/3)*cols)])
ax_ery = fig.add_subplot(grid[rowB_y1:rowB_y2, int((1/3)*cols):int((2/3)*cols)])
ax_pla = fig.add_subplot(grid[rowB_y1:rowB_y2, int((2/3)*cols):int((3/3)*cols)])

ax_mye_rank = fig.add_subplot(grid[rowC_y1:rowC_y2, int((0/3)*cols)+3:int((1/3)*cols)-3])
ax_ery_rank = fig.add_subplot(grid[rowC_y1:rowC_y2, int((1/3)*cols)+3:int((2/3)*cols)-3])
ax_pla_rank = fig.add_subplot(grid[rowC_y1:rowC_y2, int((2/3)*cols)+3:int((3/3)*cols)-3])

# plots
ax_cellt.scatter(obs['umap1'], obs['umap2'], s=3, c=obs['color'])
ax_cellt.set_title(f"All cells (n = {len(obs)})")
ax_cellt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for cluster in obs['celltype'].unique():
    cluster_obs = obs[obs['celltype'] == cluster]
    cluster_center_umap1 = cluster_obs['umap1'].mean()
    cluster_center_umap2 = cluster_obs['umap2'].mean()
    annotation = ax_cellt.annotate(cluster, (cluster_center_umap1, cluster_center_umap2), fontsize=11, fontweight='bold', color='black', ha='center', va='center')
    annotation.set_bbox({'boxstyle': 'round', 'fc': 'white', 'alpha': 0.6})

gene = 'MPO'
ax_gene1.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{gene}'])
ax_gene1.set_title(f"{gene}")
ax_gene1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

gene = 'PF4'
ax_gene2.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{gene}'])
ax_gene2.set_title(f"{gene}")
ax_gene2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

gene = 'HBB'
ax_gene3.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{gene}'])
ax_gene3.set_title(f"{gene}")
ax_gene3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

gene = 'CD74'
ax_gene4.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{gene}'])
ax_gene4.set_title(f"{gene}")
ax_gene4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

x = 'myeloid'
title = f"{x[0].upper() + x[1:]} lineage (n = {(obs[f'color_gradient_{x}'] != 'lightgray').sum()})"
ax_mye.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax_mye.set_title(title)
ax_mye.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
arrow = FancyArrowPatch(posA=(-2.3, -0.9), posB=(-0.3, 10.5), arrowstyle='->, head_width=0.3', connectionstyle=f'arc3, rad=-0.5', mutation_scale=10, lw=1, color='black')
ax_mye.add_patch(arrow)

x = 'erythroid'
title = f"{x[0].upper() + x[1:]} lineage (n = {(obs[f'color_gradient_{x}'] != 'lightgray').sum()})"
ax_ery.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax_ery.set_title(title)
ax_ery.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
arrow = FancyArrowPatch(posA=(-0.1, -0.9), posB=(7, -0.1), arrowstyle='->, head_width=0.3', connectionstyle=f'arc3, rad=-0.5', mutation_scale=10, lw=1, color='black')
ax_ery.add_patch(arrow)

x = 'platelet'
title = f"{x[0].upper() + x[1:]} lineage (n = {(obs[f'color_gradient_{x}'] != 'lightgray').sum()})"
ax_pla.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax_pla.set_title(title)
ax_pla.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
arrow = FancyArrowPatch(posA=(-3, 1.5), posB=(6.8, 10.5), arrowstyle='->, head_width=0.3', connectionstyle=f'arc3, rad=-0.1', mutation_scale=10, lw=1, color='black')
ax_pla.add_patch(arrow)

x_values, y_values, colors = lt_vs_rank(obs, 'myeloid')
ax_mye_rank.set_xlim(0, 1)
ax_mye_rank.set_xlabel('Percent Rank (uniform distribution)')
ax_mye_rank.spines['left'].set_visible(False)
ax_mye_rank.spines['right'].set_visible(False)
ax_mye_rank.yaxis.set_visible(False)
for i in range(len(x_values[0])):
    ax_mye_rank.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)
ax_mye_rank2 = ax_mye_rank.twiny()
ax_mye_rank2.set_xlabel('Latent Time (non uniform distribution)')
ax_mye_rank2.set_xlim(0, 1)

x_values, y_values, colors = lt_vs_rank(obs, 'erythroid')
ax_ery_rank.set_xlim(0, 1)
ax_ery_rank.set_xlabel('Percent Rank (uniform distribution)')
ax_ery_rank.spines['left'].set_visible(False)
ax_ery_rank.spines['right'].set_visible(False)
ax_ery_rank.yaxis.set_visible(False)
for i in range(len(x_values[0])):
    ax_ery_rank.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)
ax_ery_rank2 = ax_ery_rank.twiny()
ax_ery_rank2.set_xlabel('Latent Time (non uniform distribution)')
ax_ery_rank2.set_xlim(0, 1)

x_values, y_values, colors = lt_vs_rank(obs, 'platelet')
ax_pla_rank.set_xlim(0, 1)
ax_pla_rank.set_xlabel('Percent Rank (uniform distribution)')
ax_pla_rank.spines['left'].set_visible(False)
ax_pla_rank.spines['right'].set_visible(False)
ax_pla_rank.yaxis.set_visible(False)
for i in range(len(x_values[0])):
    ax_pla_rank.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)
ax_pla_rank2 = ax_pla_rank.twiny()
ax_pla_rank2.set_xlabel('Latent Time (non uniform distribution)')
ax_pla_rank2.set_xlim(0, 1)

# TODO
# maybe add histograms of latent time distributions as new row between rowB and rowC
# add colorbar for latent time
# add colorbar for expression


for ax_name in [ax_cellt, ax_gene1, ax_gene2, ax_gene3, ax_gene4, ax_mye, ax_ery, ax_pla, ax_mye_rank, ax_ery_rank, ax_pla_rank]:
    for spine in ax_name.spines.values():
        spine.set_visible(False)

fig.text(0.14, 0.9, 'A', fontsize=16, fontweight='bold', va='top')
fig.text(0.52, 0.9, 'B', fontsize=16, fontweight='bold', va='top')
fig.text(0.14, 0.5, 'C', fontsize=16, fontweight='bold', va='top')
fig.text(0.14, 0.2, 'D', fontsize=16, fontweight='bold', va='top')

plt.savefig(folder_data_preproc / 'plots' / "fig1.pdf", bbox_inches='tight', pad_inches=0)

############################################################################################################################################################################
"End of figure 1"
############################################################################################################################################################################
#%%
# delete this block
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs['color'])
ax.set_title(f"All cells (n = {len(obs)})")
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

clusters = obs['celltype'].unique()
for cluster in clusters:
    cluster_obs = obs[obs['celltype'] == cluster]
    cluster_center_umap1 = cluster_obs['umap1'].mean()
    cluster_center_umap2 = cluster_obs['umap2'].mean()

    annotation = ax.annotate(cluster, (cluster_center_umap1, cluster_center_umap2), fontsize=11, fontweight='bold', color='black', ha='center', va='center')
    annotation.set_bbox({'boxstyle': 'round', 'fc': 'white', 'alpha': 0.6})

#%%
# delete this block
gene = 'MPO'
for gene in genes:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{gene}'])
    ax.set_title(f"{gene}")
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

#%%
# delete this block
x = 'myeloid'
title = f"{x[0].upper() + x[1:]} lineage (n = {(obs[f'color_gradient_{x}'] != 'lightgray').sum()})"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax.set_title(title)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

arrow = FancyArrowPatch(posA=(-2.3, -0.9), posB=(-0.3, 10.5), arrowstyle='->, head_width=0.3', connectionstyle=f'arc3, rad=-0.5', mutation_scale=10, lw=1, color='black')
ax.add_patch(arrow)

# df_colorbar = obs[[f'color_gradient_{x}', f'latent_time_{x}']].sort_values(by=f'latent_time_{x}')
# cmap = ListedColormap(df_colorbar[f'color_gradient_{x}'].unique())
# cax = fig.add_axes([0.2, 0.8, 0.1, 0.03])
# cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
# cb.set_label('latent time', labelpad=5)
# cb.ax.yaxis.set_tick_params(labelleft=True, labelright=False)

#%%
# delete this block
x = 'erythroid'
title = f"{x[0].upper() + x[1:]} lineage (n = {(obs[f'color_gradient_{x}'] != 'lightgray').sum()})"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax.set_title(title)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

arrow = FancyArrowPatch(posA=(-0.1, -0.9), posB=(7, -0.1), arrowstyle='->, head_width=0.3', connectionstyle=f'arc3, rad=-0.5', mutation_scale=10, lw=1, color='black')
ax.add_patch(arrow)

#%%
# delete this block
x = 'platelet'
title = f"{x[0].upper() + x[1:]} lineage (n = {(obs[f'color_gradient_{x}'] != 'lightgray').sum()})"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax.set_title(title)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

arrow = FancyArrowPatch(posA=(-3, 1.5), posB=(6.8, 10.5), arrowstyle='->, head_width=0.3', connectionstyle=f'arc3, rad=-0.1', mutation_scale=10, lw=1, color='black')
ax.add_patch(arrow)

# %%
fig, ax = plt.subplots(figsize=(8, 2))
lineage = 'myeloid'
x_values, y_values, colors = lt_vs_rank(obs, lineage)
ax.set_xlim(0, 1)
ax.set_xlabel('Percent Rank (uniform distribution)')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_visible(False)

for i in range(len(x_values[0])):
    ax.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)

ax2 = ax.twiny()
ax2.set_xlabel('Latent Time (non uniform distribution)')
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, 1)

plt.tight_layout()
# plt.savefig(folder_data_preproc / 'plots' / f"fig1_{lineage}_rank.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(8, 2))
lineage = 'erythroid'
x_values, y_values, colors = lt_vs_rank(obs, lineage)
ax.set_xlim(0, 1)
ax.set_xlabel('Percent Rank (uniform distribution)')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_visible(False)

for i in range(len(x_values[0])):
    ax.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)

ax2 = ax.twiny()
ax2.set_xlabel('Latent Time (non uniform distribution)')
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, 1)

plt.tight_layout()
# plt.savefig(folder_data_preproc / 'plots' / f"fig1_{lineage}_rank.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(8, 2))
lineage = 'platelet'
x_values, y_values, colors = lt_vs_rank(obs, lineage)
ax.set_xlim(0, 1)
ax.set_xlabel('Percent Rank (uniform distribution)')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_visible(False)

for i in range(len(x_values[0])):
    ax.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)

ax2 = ax.twiny()
ax2.set_xlabel('Latent Time (non uniform distribution)')
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, 1)

plt.tight_layout()
# plt.savefig(folder_data_preproc / 'plots' / f"fig1_{lineage}_rank.pdf")
plt.show()

#%%
# Set up the figure and subplots
fig = plt.figure(figsize=(16, 15))

# Plot 1: All cells
ax1 = fig.add_subplot(3, 2, 1)
ax1.scatter(obs['umap1'], obs['umap2'], s=1, c=obs['color'])
ax1.set_title(f"All cells ({len(obs)} cells)")
ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in ax1.spines.values():
    spine.set_visible(False)

clusters = obs['celltype'].unique()
for cluster in clusters:
    cluster_obs = obs[obs['celltype'] == cluster]
    cluster_center_umap1 = cluster_obs['umap1'].mean()
    cluster_center_umap2 = cluster_obs['umap2'].mean()

    annotation = ax1.annotate(
        cluster,
        (cluster_center_umap1, cluster_center_umap2),
        fontsize=11,
        fontweight='bold',
        color='black',
        ha='center',
        va='center'
    )

    annotation.set_bbox({
        'boxstyle': 'round',
        'fc': 'white',
        'alpha': 0.6 
    })

# Plot 2: Myeloid
ax2 = fig.add_subplot(3, 2, 2)
x = 'myeloid'
n = (obs[f'color_gradient_{x}'] != 'lightgray').sum()
lineage = x[0].upper() + x[1:]
title = f"{lineage} lineage ({n} cells)"
ax2.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax2.set_title(title)
ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in ax2.spines.values():
    spine.set_visible(False)

arrow2 = FancyArrowPatch(
    posA=(-2.3, -0.9),
    posB=(-0.3, 10.5),
    arrowstyle='->, head_width=0.3',
    connectionstyle=f'arc3, rad=-0.5',
    mutation_scale=10,
    lw=1,
    color='black'
)

ax2.add_patch(arrow2)

# Plot 3: Erythroid
ax3 = fig.add_subplot(3, 2, 3)
x = 'erythroid'
n = (obs[f'color_gradient_{x}'] != 'lightgray').sum()
lineage = x[0].upper() + x[1:]
title = f"{lineage} lineage ({n} cells)"
ax3.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax3.set_title(title)
ax3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in ax3.spines.values():
    spine.set_visible(False)

arrow3 = FancyArrowPatch(
    posA=(-0.1, -0.9),
    posB=(7, -0.1),
    arrowstyle='->, head_width=0.3',
    connectionstyle=f'arc3, rad=-0.5',
    mutation_scale=10,
    lw=1,
    color='black'
)

ax3.add_patch(arrow3)

# Plot 4: Platelet
ax4 = fig.add_subplot(3, 2, 4)
x = 'platelet'
n = (obs[f'color_gradient_{x}'] != 'lightgray').sum()
lineage = x[0].upper() + x[1:]
title = f"{lineage} lineage ({n} cells)"
ax4.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax4.set_title(title)
ax4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in ax4.spines.values():
    spine.set_visible(False)

arrow4 = FancyArrowPatch(
    posA=(-3, 1.5),
    posB=(6.8, 10.5),
    arrowstyle='->, head_width=0.3',
    connectionstyle=f'arc3, rad=-0.1',
    mutation_scale=10,
    lw=1,
    color='black'
)

ax4.add_patch(arrow4)

ax5 = fig.add_axes([0.1, 0.05, 0.35, 0.25])
colors = ['darkorange', 'mediumorchid', 'cornflowerblue']
df_myeloid['latent_time_myeloid'].hist(bins=50, alpha=0.7, label='Myeloid', color=colors[0], ax=ax5)
df_erythroid['latent_time_erythroid'].hist(bins=50, alpha=0.7, label='Erythroid', color=colors[1], ax=ax5)
df_platelet['latent_time_platelet'].hist(bins=50, alpha=0.7, label='Platelet', color=colors[2], ax=ax5)
ax5.set_xlabel('Latent Time')
ax5.set_ylabel('Frequency')
ax5.set_title('Latent Time Distribution')
ax5.legend()
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.grid(False)
ax5.tick_params(axis='both', which='both', direction='out', top=False, right=False)

# %%
obs_hsc = obs[obs['celltype'] == 'HSC']
obs_hsc = obs_hsc.sort_values(by='latent_time_myeloid')
line_colors = obs_hsc['color_gradient_myeloid'].tolist()
obs_hsc = obs_hsc[['latent_time_myeloid', 'latent_time_erythroid', 'latent_time_platelet']]
obs_hsc = obs_hsc.rename(columns={'latent_time_myeloid': 'Myeloid', 'latent_time_erythroid': 'Erythroid', 'latent_time_platelet': 'Platelet'})
obs_hsc['class'] = range(len(obs_hsc))

ax6 = fig.add_axes([0.6, 0.05, 0.30, 0.25])
pc = pd.plotting.parallel_coordinates(obs_hsc, 'class', color=line_colors, lw=0.1, ax=ax6)
pc.set_ylim(0, 0.35)
pc.set_ylabel('Latent Time')
pc.tick_params(axis='y', which='both', labelleft=True, labelright=True)
pc.legend().remove()
pc.yaxis.grid(False)
pc.yaxis.set_ticks_position('both')
pc.spines['top'].set_visible(False)
pc.spines['bottom'].set_visible(False)
plt.title(f'Latent Time for HSCs across lineages ({len(obs_hsc)} cells)')

x = 'myeloid'
df_colorbar = obs[[f'color_gradient_{x}', f'latent_time_{x}']].sort_values(by=f'latent_time_{x}')
cmap = ListedColormap(df_colorbar[f'color_gradient_{x}'].unique())
cax = fig.add_axes([0.5, 0.4, 0.08, 0.01])
cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
cb.set_label('latent time', labelpad=5)
cb.ax.yaxis.set_tick_params(labelleft=True, labelright=False)

plt.tight_layout()
plt.savefig(folder_data_preproc / 'plots' / f"fig1_old.pdf")
plt.show()
