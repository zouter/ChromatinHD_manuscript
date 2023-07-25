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

from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
annotation_original = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]
pseudocoordinates = torch.linspace(0, 1, 1000)

#%%
lineage_gene = {'lin_myeloid': 'MPO', 'lin_erythroid': 'HBB', 'lin_platelet': 'CD74'}
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
    df_latent['quantile'] = pd.qcut(df_latent['latent_time'], 10, labels = False) + 1
    latent = pd.get_dummies(df_latent['quantile'])
    latent_torch = torch.from_numpy(latent.values).to(torch.float)
    adata_oi = adata[list(df_latent.index), gene_name]
    df_latent[gene_name] = pd.DataFrame(adata_oi.X, index=adata_oi.obs.index, columns=[gene_name])
    exp_xmin, exp_xmax = df_latent[gene_name].min(), df_latent[gene_name].max()

    dir_likelihood = folder_data_preproc / f"{dataset_name_sub}_LQ/lq_{dataset_name}_128_64_32_fold_0"
    probs = pd.read_csv(dir_likelihood / (gene_id + '.csv'), index_col=0)

    df_bincounts = pd.DataFrame()
    for i, celltype in enumerate(latent.columns):
        fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_ix)
        bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
        n_cells = latent_torch[:, i].sum()
        bincounts = bincounts / n_cells * len(bins)

        df_bincounts[celltype] = bincounts

    print(lineage_name)
    lineage_objects[lineage_name] = {
        'df_latent': df_latent,
        'latent_torch': latent_torch,
        'fragments': fragments,
        'celltypes': latent.columns,
        'gene_name': gene_name,
        'gene_ix': gene_ix,
        'probs': probs,
        'df_bincounts': df_bincounts,
        'exp_xmin': exp_xmin,
        'exp_xmax': exp_xmax,
    }

# %%
def gradient(colors):
    points = np.linspace(0, 1, len(colors))
    red = interp1d(points, [color[0] for color in colors])
    green = interp1d(points, [color[1] for color in colors])
    blue = interp1d(points, [color[2] for color in colors])
    result = [(float(red(i)), float(green(i)), float(blue(i))) for i in np.linspace(0, 1, 10)]
    result = {i+1: result[i] for i in range(len(result))}
    return result

def plot_accessibility(fig, gridspec, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation, ymax):
    ax_object = fig.add_subplot(gridspec)
    ax_object.bar(binmids, bincounts, width=binsize, color="#888888", lw=0)
    ax_object.plot(pseudocoordinates.numpy(), data['probs'].iloc[index, :], label=index, color=annotation[data['celltypes'][index]], lw=2, zorder=20)
    ax_object.plot(pseudocoordinates.numpy(), data['probs'].iloc[index, :], label=index, color="#FFFFFF", lw=3, zorder=10)
    ax_object.set_ylabel(f"quantile {data['probs'].index[index]}  \n n={int(n_cells)}  ", rotation=0, ha="right", va="center")
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.set_ylim(0, ymax)
    new_ticks = [-10000, -5000, 0, 5000, 10000]
    old_ticks = [x/20000 + 0.5 for x in new_ticks]
    ax_object.set_xticks(old_ticks)
    ax_object.set_xticklabels(new_ticks)
    return ax_object

def plot_expression(fig, gridspec, expression, celltype, annotation, exp_xmin, exp_xmax):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(exp_xmin * 1.05, exp_xmax * 1.05)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    ax_object.get_yaxis().set_visible(False)
    return ax_object

def plot_lt(fig, gridspec, lt, celltype, annotation):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(lt, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(-0.05, 1.05)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    ax_object.get_yaxis().set_visible(False)
    return ax_object

# %%
# Create the grid layout
height, width = 10, 10
rows, cols = 10, 20

lineages = [x for x in info_genes_cells.columns if 'lin' in x]
lineages_dict = {x: lineage_objects[x]['celltypes'] for x in lineages}

for lineage in lineages:
    print(lineage, lineage_objects[lineage]['celltypes'])

    fig = plt.figure(figsize=(width, height))
    grid = GridSpec(rows, cols, figure=fig)

    lineages_dict = {x: lineage_objects[x]['celltypes'] for x in [lineage]}

    df = pd.DataFrame([(lineage, cell_type) for lineage, cell_types in lineages_dict.items() for cell_type in cell_types], columns=['lineage', 'cell type'])
    df['group'] = 0
    df['index'] = df.groupby('group').cumcount()
    df['plot_row'] = range(df.shape[0]) + df['group']
    df['plot_row'] = df.groupby('group')['plot_row'].transform(lambda x: x.sort_values(ascending=False).values)
    df = df.sort_values('plot_row').reset_index(drop=True)
    df['xaxis'] = df.groupby('group')['index'].transform(lambda x: x == x.min())
    df['title'] = df.groupby('group')['index'].transform(lambda x: x == x.max())

    col_accessibility = [grid[i, 1:10] for i in df['plot_row']]
    col_expression = [grid[i, 11:15] for i in df['plot_row']]
    col_lt = [grid[i, 16:20] for i in df['plot_row']]

    lineage = info_genes_cells[lineage].dropna().tolist()
    colors = [annotation_original[x] for x in lineage]
    annotation = gradient(colors)

    for lineage, celltype, index, plot_row, xaxis, title, c1, c2, c3 in zip(df['lineage'], df['cell type'], df['index'], df['plot_row'], df['xaxis'], df['title'], col_accessibility, col_expression, col_lt):
        print(lineage, celltype, index, plot_row, c1, c2, c3)

        data = lineage_objects[lineage]
        print(data['gene_name'], data['celltypes'])

        # 1. data for accessibility
        bincounts = data['df_bincounts'][celltype]
        ymax = data['df_bincounts'].max().max()
        print(ymax)
        n_cells = data['latent_torch'][:, index].sum()

        # 2. data for expression
        expression = data['df_latent'].loc[data['df_latent']['quantile'] == celltype, data['gene_name']].values
        
        # 3. data for latent time
        lt = data['df_latent'].loc[data['df_latent']['quantile'] == celltype, 'latent_time'].values

        ax_1 = plot_accessibility(fig, c1, binmids, bincounts, n_cells, bins, binsize, pseudocoordinates, data, index, annotation, ymax)
        ax_2 = plot_expression(fig, c2, expression, data['celltypes'][index], annotation, data['exp_xmin'], data['exp_xmax'])
        ax_3 = plot_lt(fig, c3, lt, data['celltypes'][index], annotation)

        if title == True:
            ax_1.set_title(f"{lineage_gene[lineage]}: Accessibility")
            ax_2.set_title(f"{lineage_gene[lineage]}: Expression")
            ax_3.set_title("Latent Time")
        
        if xaxis == False:
            ax_1.set_xticklabels([])
            ax_2.set_xticklabels([])
            ax_3.set_xticklabels([])

    x1, x2, x3 = 0.05, 0.52, 0.73
    y1, y2, y3 = 0.905, 0.58, 0.32

    fig.text(x1, y1, 'A', fontsize=16, fontweight='bold', va='top')
    fig.text(x2, y1, 'B', fontsize=16, fontweight='bold', va='top')
    fig.text(x3, y1, 'C', fontsize=16, fontweight='bold', va='top')

    fig.savefig(folder_data_preproc / 'plots' / f"fig3_{lineage}_{lineage_gene[lineage]}.pdf", bbox_inches='tight', pad_inches=0.01)
    fig.show()

############################################################################################################################################################################
"End of figure 3"
############################################################################################################################################################################

# %%
# alternative for third panel, fraction of cell types in each quantile instead of boxplot of latent time
lin = 'lin_myeloid'
quantiles = sorted(lineage_objects[lin]['df_latent']['quantile'].unique())[::-1]
lineage = info_genes_cells[lin].dropna().tolist()

for quant in quantiles:
    print(quant)
    df_oi = lineage_objects[lin]['df_latent'][lineage_objects[lin]['df_latent']['quantile'] == quant]
    vc = df_oi['celltype'].value_counts(normalize=True)

    df_vc = pd.DataFrame(vc).reset_index()
    df_vc.columns = ['celltype', 'fraction']
    df_vc['color'] = df_vc['celltype'].apply(lambda x: annotation_original[x])
    df_vc['order'] = df_vc['celltype'].apply(lambda x: lineage.index(x))
    df_vc = df_vc.sort_values('order')

    fig, ax = plt.subplots(figsize=(5, 0.5))
    bottom = 0
    for value, color in zip(df_vc['fraction'], df_vc['color']):
        ax.barh('a', value, left=bottom, color=color)
        bottom += value
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
