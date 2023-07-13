# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
dataset_name_sub = "MV2"
folder_data_preproc = folder_data / dataset_name

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
annotation = {
    'LMPP': '134, 94, 173',
    'HSC': '213, 47, 47',
    'MEP': '140, 86, 76',
    'MPP': '222, 117, 189',
    'Erythrocyte': '31, 119, 180',
    'GMP': '255, 127, 15',
    'Prog MK': '21, 190, 207',
    'Granulocyte': '43, 160, 43',
    'Prog DC': '188, 189, 33',
    'Prog B': '127, 127, 127'
 }

annotation = {key: tuple([int(x) / 255 for x in value.split(", ")]) for key, value in annotation.items()}

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

# %%
obs_hsc = obs[obs['celltype'] == 'HSC']
obs_hsc = obs_hsc.sort_values(by='latent_time_myeloid')
line_colors = obs_hsc['color_gradient_myeloid'].tolist()
obs_hsc = obs_hsc[['latent_time_myeloid', 'latent_time_erythroid', 'latent_time_platelet']]
obs_hsc = obs_hsc.rename(columns={'latent_time_myeloid': 'Myeloid', 'latent_time_erythroid': 'Erythroid', 'latent_time_platelet': 'Platelet'})
obs_hsc['class'] = range(len(obs_hsc))

#%%
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs['color'])
ax.set_title(f"All cells ({len(obs)} cells)")
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

clusters = obs['celltype'].unique()
for cluster in clusters:
    cluster_obs = obs[obs['celltype'] == cluster]
    cluster_center_umap1 = cluster_obs['umap1'].mean()
    cluster_center_umap2 = cluster_obs['umap2'].mean()

    annotation = ax.annotate(
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

#%%
x = 'myeloid'
n = (obs[f'color_gradient_{x}'] != 'lightgray').sum()
lineage = x[0].upper() + x[1:]
title = f"{lineage} lineage ({n} cells)"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax.set_title(title)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

arrow = FancyArrowPatch(
    posA=(-2.3, -0.9),
    posB=(-0.3, 10.5),
    arrowstyle='->, head_width=0.3',
    connectionstyle=f'arc3, rad=-0.5',
    mutation_scale=10,
    lw=1,
    color='black'
)

ax.add_patch(arrow)

df_colorbar = obs[[f'color_gradient_{x}', f'latent_time_{x}']].sort_values(by=f'latent_time_{x}')
cmap = ListedColormap(df_colorbar[f'color_gradient_{x}'].unique())
cax = fig.add_axes([0.2, 0.8, 0.1, 0.03])
cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
cb.set_label('latent time', labelpad=5)
cb.ax.yaxis.set_tick_params(labelleft=True, labelright=False)

#%%
x = 'erythroid'
n = (obs[f'color_gradient_{x}'] != 'lightgray').sum()
lineage = x[0].upper() + x[1:]
title = f"{lineage} lineage ({n} cells)"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax.set_title(title)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

arrow = FancyArrowPatch(
    posA=(-0.1, -0.9),
    posB=(7, -0.1),
    arrowstyle='->, head_width=0.3',
    connectionstyle=f'arc3, rad=-0.5',
    mutation_scale=10,
    lw=1,
    color='black'
)

ax.add_patch(arrow)

#%%
x = 'platelet'
n = (obs[f'color_gradient_{x}'] != 'lightgray').sum()
lineage = x[0].upper() + x[1:]
title = f"{lineage} lineage ({n} cells)"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(obs['umap1'], obs['umap2'], s=1, c=obs[f'color_gradient_{x}'])
ax.set_title(title)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

arrow = FancyArrowPatch(
    posA=(-3, 1.5),
    posB=(6.8, 10.5),
    arrowstyle='->, head_width=0.3',
    connectionstyle=f'arc3, rad=-0.1',
    mutation_scale=10,
    lw=1,
    color='black'
)

ax.add_patch(arrow)

#%%
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['darkorange', 'mediumorchid', 'cornflowerblue']
df_myeloid['latent_time_myeloid'].hist(bins=50, alpha=0.7, label='Myeloid', color=colors[0])
df_erythroid['latent_time_erythroid'].hist(bins=50, alpha=0.7, label='Erythroid', color=colors[1])
df_platelet['latent_time_platelet'].hist(bins=50, alpha=0.7, label='Platelet', color=colors[2])
ax.set_xlabel('Latent Time')
ax.set_ylabel('Frequency')
ax.set_title('Latent Time Distribution')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.tick_params(axis='both', which='both', direction='out', top=False, right=False)

#%%
plt.figure(figsize=(8, 5))
ax = pd.plotting.parallel_coordinates(obs_hsc, 'class', color=line_colors, lw=0.1)
ax.set_ylim(0, 0.35)
ax.set_ylabel('Latent Time')
ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
ax.legend().remove()
ax.yaxis.grid(False)
ax.yaxis.set_ticks_position('both')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title(f'Latent Time for HSCs across lineages ({len(obs_hsc)} cells)')

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
plt.savefig(folder_data_preproc / 'plots' / f"fig1.pdf")
plt.show()

# %%
x = 'myeloid'
x = 'erythroid'
x = 'platelet'

for x in ['myeloid', 'erythroid', 'platelet']:

    sub = obs[[f'color_gradient_{x}', f'latent_time_{x}']]
    sub = sub[sub[f'latent_time_{x}'].notna()]
    sub = sub.sort_values(by=f'latent_time_{x}')
    sub['rank'] = sub[f'latent_time_{x}'].rank() - 1
    sub['rank'] = sub['rank'] / sub['rank'].max()

    sub['lt_shift'] = sub[f'latent_time_{x}'].diff()
    sub.loc[sub.index[0], 'lt_shift'] = sub.loc[sub.index[1], f'latent_time_{x}']

    sub['diff'] = sub['lt_shift'] - sub['rank'].iloc[1]
    sub = sub.sort_values(by='diff')
    sub['diff_rank'] = sub['diff'].rank() - 1

    count_negative = sub[sub['diff'] < 0]['diff'].count()
    count_positive = sub[sub['diff'] >= 0]['diff'].count()

    sub.loc[sub['diff'] < 0, 'diff_rank'] = (sub.loc[sub['diff'] < 0, 'diff_rank'] / count_negative) * 0.5
    sub.loc[sub['diff'] >= 0, 'diff_rank'] = (sub.loc[sub['diff'] >= 0, 'diff_rank'] - count_negative) / count_positive * 0.5 + 0.5

    # sub['diff_rank'].hist(bins=50)

    sub['diff_rank_color'] = create_color_gradient(sub, 'diff_rank', 'RdBu')

    sub['y1'] = 1
    sub['y2'] = 0

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlabel('Percent Rank (uniform distribution)')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(0, 1)

    x_values = np.vstack((sub[f'latent_time_{x}'], sub['rank']))
    y_values = np.vstack((sub['y1'], sub['y2']))
    colors = sub['diff_rank_color'].to_numpy()
    for i in range(len(x_values[0])):
        ax.plot(x_values[:,i], y_values[:,i], c=colors[i], linewidth=0.1)

    ax2 = ax.twiny()
    ax2.set_xlabel('Latent Time (non uniform distribution)')
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(folder_data_preproc / 'plots' / f"fig1_{x}_rank.pdf")
    plt.show()

# %%
fig = chd.grid.Figure(main = chd.grid.Grid())

# first row
row1_grid = fig.main[0, 0] = chd.grid.Grid()

# cell type
panel, ax = row1_grid[0, 0] = chd.grid.Panel((3, 3))
# you can use ax to plot your cell type stuff

# genes_grid
genes_grid = row1_grid[0, 1] = chd.grid.Wrap(ncol = 2)

genes = ["A", "B", "C", "D"]
for gene in genes:
    panel, ax = genes_grid.add(chd.grid.Panel((1.5, 1.5)))
    # you can use ax to plot your gene

# 
row23_grid = fig.main[1, 0] = chd.grid.Grid(ncol = 3)

lineages = ["a", "b", "c"]

for i, lineage in enumerate(lineages):
    panel, ax_lt = row23_grid[0, i] = chd.grid.Panel((2, 2))
    panel, ax_rank = row23_grid[1, i] = chd.grid.Panel((2, 0.5))
fig.plot()
# %%
