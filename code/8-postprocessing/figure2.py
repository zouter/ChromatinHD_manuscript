# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import torch
import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
annotation = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

#%%
lineage_gene = {'lin_myeloid': 'MPO', 'lin_erythroid': 'HBB', 'lin_platelet': 'PF4'}
lineage_objects = {}

for lineage_name, gene_name in lineage_gene.items():

    dataset_name = lineage_name.replace('lin_', '')
    gene_name = gene_name

    fragments = chd.data.Fragments(folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}/10k10k")
    fragments.window = window
    fragments.create_cut_data()
    gene_id = adata.var.loc[gene_name]['Accession']
    gene_ix = fragments.var.index.get_loc(gene_id)

    lineage = info_genes_cells[f'lin_{dataset_name}'].dropna().tolist()
    df_latent = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv", index_col = 0)
    latent = pd.get_dummies(df_latent['celltype'])
    latent = latent[lineage]
    latent_torch = torch.from_numpy(latent.values).to(torch.float)
    adata_oi = adata[list(df_latent.index), gene_name]
    df_latent[gene_name] = pd.DataFrame(adata_oi.X, index=adata_oi.obs.index, columns=[gene_name])
    exp_xmin, exp_xmax = df_latent[gene_name].min(), df_latent[gene_name].max()

    dir_likelihood = folder_data_preproc / f"{dataset_name_sub}_LCT/lct_{dataset_name}_128_64_32_fold_0"
    probs = pd.read_csv(dir_likelihood / (gene_id + '.csv'), index_col=0)

    lineage_objects[lineage_name] = {
        'df_latent': df_latent,
        'latent_torch': latent_torch,
        'fragments': fragments,
        'celltypes': lineage,
        'gene_name': gene_name,
        'gene_ix': gene_ix,
        'probs': probs,
        'exp_xmin': exp_xmin,
        'exp_xmax': exp_xmax,
    }

# %%
bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]
pseudocoordinates = torch.linspace(0, 1, 1000)

# Create the grid layout
height, width = 10, 10
fig = plt.figure(figsize=(height, width))

rows, cols = 12, 20
grid = GridSpec(rows, cols, figure=fig)

lineages = [x for x in info_genes_cells.columns if 'lin' in x]
lineages_dict = {x: info_genes_cells[x].dropna().tolist() for x in lineages}
df = pd.DataFrame([(lineage, cell_type) for lineage, cell_types in lineages_dict.items() for cell_type in cell_types], columns=['lineage', 'cell type'])
df['group'] = df['lineage'].map({lineage: i for i, lineage in enumerate(lineages)})
df['index'] = df.groupby('group').cumcount()
df['plot_row'] = range(df.shape[0]) + df['group']
df['plot_row'] = df.groupby('group')['plot_row'].transform(lambda x: x.sort_values(ascending=False).values)
df = df.sort_values('plot_row').reset_index(drop=True)
# cumcount for each group

col_accessibility = [grid[i, 1:10] for i in df['plot_row']]
col_expression = [grid[i, 11:15] for i in df['plot_row']]
col_lt = [grid[i, 16:20] for i in df['plot_row']]

def plot_accessibility(fig, gridspec, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation):
    ax_object = fig.add_subplot(gridspec)
    ax_object.bar(binmids, bincounts / n_cells * len(bins), width=binsize, color="#888888", lw=0)
    ax_object.plot(pseudocoordinates.numpy(), data['probs'].iloc[index, :], label=index, color=annotation[data['celltypes'][index]], lw=2, zorder=20)
    ax_object.plot(pseudocoordinates.numpy(), data['probs'].iloc[index, :], label=index, color="#FFFFFF", lw=3, zorder=10)
    ax_object.set_ylabel(f"{data['probs'].index[index]}  \n n={int(n_cells)}  ", rotation=0, ha="right", va="center")
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    return ax_object

def plot_expression(fig, gridspec, expression, celltype, annotation, exp_xmin, exp_xmax):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(exp_xmin * 1.05, exp_xmax * 1.05)
    return ax_object

def plot_lt(fig, gridspec, lt, celltype, annotation):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(lt, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(-0.05, 1.05)
    return ax_object

for lineage, celltype, index, plot_row, c1, c2, c3 in zip(df['lineage'], df['cell type'], df['index'], df['plot_row'], col_accessibility, col_expression, col_lt):
    print(lineage, celltype, index, plot_row, c1, c2, c3)

    data = lineage_objects[lineage]
    print(data['gene_name'], data['celltypes'])

    # 1. data for accessibility
    fragments_oi = (data['latent_torch'][data['fragments'].cut_local_cell_ix, index] != 0) & (data['fragments'].cut_local_gene_ix == data['gene_ix'])
    bincounts, _ = np.histogram(data['fragments'].cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
    n_cells = data['latent_torch'][:, index].sum()

    # 2. data for expression
    expression = data['df_latent'].loc[data['df_latent']['celltype'] == data['celltypes'][index], data['gene_name']].values
    
    # 3. data for latent time
    lt = df_latent.loc[df_latent['celltype'] == celltype, 'latent_time'].values

    ax_1 = plot_accessibility(fig, c1, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation)
    ax_2 = plot_expression(fig, c2, expression, data['celltypes'][index], annotation, data['exp_xmin'], data['exp_xmax'])
    ax_3 = plot_lt(fig, c3, lt, data['celltypes'][index], annotation)

#%%
subplot_positions = {}
for index, lineage in enumerate(lineages):
    print(index, lineage)
    subplot_positions[lineage] = {
        f"ax_{1 + len(lineages)*index}": [grid[i, 1:10] for i in df.loc[df['lineage'] == lineage, 'plot_row']][::-1],
        f"ax_{2 + len(lineages)*index}": [grid[i, 11:15] for i in df.loc[df['lineage'] == lineage, 'plot_row']][::-1],
        f"ax_{3 + len(lineages)*index}": [grid[i, 16:20] for i in df.loc[df['lineage'] == lineage, 'plot_row']][::-1],
    }

ax_1 = None
ax_4 = None
ax_7 = None

for lin_type, ax_dict in subplot_positions.items():
    print(lin_type)
    data = lineage_objects[lin_type]
    print(data['gene_name'], data['celltypes'])
    for ax, gridspec_list in ax_dict.items():
        for index, gridspec in enumerate(gridspec_list):
            if ax in ['ax_1', 'ax_4', 'ax_7']:
                print(ax, gridspec, index, data['celltypes'][index])

                fragments_oi = (data['latent_torch'][data['fragments'].cut_local_cell_ix, index] != 0) & (data['fragments'].cut_local_gene_ix == data['gene_ix'])
                bincounts, _ = np.histogram(data['fragments'].cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
                n_cells = data['latent_torch'][:, index].sum()

                if ax == 'ax_1':
                    ax_1 = plot_accessibility(fig, gridspec, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation)
                if ax == 'ax_4':
                    ax_4 = plot_accessibility(fig, gridspec, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation)
                if ax == 'ax_7':
                    ax_7 = plot_accessibility(fig, gridspec, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation)

            if ax in ['ax_2', 'ax_5', 'ax_8']:
                print(ax, gridspec, index, data['celltypes'][index])

                expression = data['df_latent'].loc[data['df_latent']['celltype'] == data['celltypes'][index], data['gene_name']].values

                if ax == 'ax_2':
                    ax_2 = plot_expression(fig, gridspec, expression, data['celltypes'][index], annotation, data['exp_xmin'], data['exp_xmax'])
                if ax == 'ax_5':
                    ax_5 = plot_expression(fig, gridspec, expression, data['celltypes'][index], annotation, data['exp_xmin'], data['exp_xmax'])
                if ax == 'ax_8':
                    ax_8 = plot_expression(fig, gridspec, expression, data['celltypes'][index], annotation, data['exp_xmin'], data['exp_xmax'])

            if ax in ['ax_3', 'ax_6', 'ax_9']:
                print(ax, gridspec, index, data['celltypes'][index])


                if ax == 'ax_3':
                    ax_3 = fig.add_subplot(gridspec)
                if ax == 'ax_6':
                    ax_6 = fig.add_subplot(gridspec)
                if ax == 'ax_9':
                    ax_9 = fig.add_subplot(gridspec)

                ax_3.boxplot([11, 12, 13], vert=False, widths=0.5)
    
        print('---')

fig.show()

#%%
# Create the shared ax_1 outside the loop
ax_1 = None
ax_4 = None
ax_7 = None

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]
pseudocoordinates = torch.linspace(0, 1, 1000)

num_plots_mye = len(lineage)
subplot_positions = {
    "ax_1": [grid[i, 1:10] for i in range(num_plots_mye)],
    "ax_2": [grid[i, 11:15] for i in range(num_plots_mye)],
    "ax_3": [grid[i, 16:20] for i in range(num_plots_mye)],
}

for i, celltype in enumerate(probs.index):
    print(i, celltype)

    if i == 0:
        ax_1 = fig.add_subplot(subplot_positions["ax_1"][num_plots_mye - 1 - i])
    else:
        ax_1 = fig.add_subplot(subplot_positions["ax_1"][num_plots_mye - 1 - i], sharey=ax_1)
    ax_2 = fig.add_subplot(subplot_positions["ax_2"][num_plots_mye - 1 - i])
    ax_3 = fig.add_subplot(subplot_positions["ax_3"][num_plots_mye - 1 - i])

    fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_ix)
    bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
    n_cells = latent_torch[:, i].sum()
    ax_1.bar(binmids, bincounts / n_cells * len(bins), width=binsize, color="#888888", lw=0)
    ax_1.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label=i, color=annotation[celltype], lw=2, zorder=20)
    ax_1.plot(pseudocoordinates.numpy(), probs.iloc[i, :], label=i, color="#FFFFFF", lw=3, zorder=10)
    ax_1.set_ylabel(f"{probs.index[i]}  \n n={int(n_cells)}  ", rotation=0, ha="right", va="center")
    ax_1.spines['top'].set_visible(False)
    ax_1.spines['right'].set_visible(False)

    medianprops = dict(color=annotation[celltype], linewidth=1)

    expression = df_latent.loc[df_latent['celltype'] == celltype, gene_name].values
    ax_2.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_2.set_xlim(exp_xmin*1.05, exp_xmax*1.05)

    lt = df_latent.loc[df_latent['celltype'] == celltype, 'latent_time'].values
    ax_3.boxplot(lt, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_3.set_xlim(-0.05, 1.05)

    for ax in [ax_2, ax_3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)

    if i > 0:
        ax_1.set_xticklabels([])
        ax_2.set_xticklabels([])
        ax_3.set_xticklabels([])
    
    if i == num_plots_mye - 1:
        ax_1.set_title(f"{gene_name}: Accessibility")
        ax_2.set_title(f"{gene_name}: Expression")
        ax_3.set_title("Latent Time")

fig.text(0.15, 0.95, 'A', fontsize=16, fontweight='bold', va='top')
fig.text(0.55, 0.95, 'B', fontsize=16, fontweight='bold', va='top')
fig.text(0.74, 0.95, 'C', fontsize=16, fontweight='bold', va='top')

fig.savefig(folder_data_preproc / 'plots' / "fig2.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()

# %%
