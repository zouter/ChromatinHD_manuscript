# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"
model_type = 'quantile'

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
if len(sys.argv) > 2 and sys.argv[2] == 'from_fig_runner':
    lineage_gene = sys.argv[1]
    lineage_gene = dict([eval(lineage_gene)])
    fig_runner = True
else:
    fig_runner = False

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
    df_latent['reads'] = pd.DataFrame(adata_oi.layers['matrix'].todense()).values

    whisker_limits = {}
    for quantile in df_latent['quantile'].unique():
        subset = df_latent[df_latent['quantile'] == quantile]
        Q1 = np.percentile(subset[gene_name], 25)
        Q3 = np.percentile(subset[gene_name], 75)
        IQR = Q3 - Q1
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR
        whisker_limits[quantile] = (lower_whisker, upper_whisker)

    w_min = min([x[0] for x in whisker_limits.values()])
    w_max = max([x[1] for x in whisker_limits.values()])

    exp_xmin = max(w_min, exp_xmin)
    exp_xmax = min(w_max, exp_xmax)

    dir_likelihood = folder_data_preproc / f"{dataset_name_sub}_LQ/{dataset_name_sub}_{dataset_name}_{model_type}_128_64_32"
    probs = pd.read_csv(dir_likelihood / (gene_id + '.csv.gz'), index_col=0)
    probs = probs.T
    probs.index = probs.index.astype(int)

    df_bincounts = pd.DataFrame()
    n_fr = []
    for i, celltype in enumerate(latent.columns):
        fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_ix)
        n_frags = fragments_oi.sum().item()
        bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=bins)
        n_cells = latent_torch[:, i].sum()
        bincounts = bincounts / n_cells * len(bins)

        df_bincounts[celltype] = bincounts
        n_fr.append(n_frags)

    probs.index.name = "cluster"
    probs.columns.name = "position"

    plotdata = pd.DataFrame({"prob": probs.unstack()})

    plotdata["prob_baseline"] = probs.iloc[0].reindex(plotdata.index, level=0).values
    plotdata['prob_diff'] = plotdata['prob'] / plotdata['prob_baseline']

    plotdata['prob_baseline_consecutive'] = plotdata.groupby('position')['prob'].apply(lambda x: x.shift(1).fillna(x.iloc[0]))
    plotdata['prob_diff_consecutive'] = plotdata['prob'] / plotdata['prob_baseline_consecutive']

    plotdata['baseline'] = plotdata['prob_baseline']
    plotdata['gradient'] = plotdata['prob_diff']

    plotdata.loc[plotdata['gradient'] < 1, 'gradient'] = -1 / plotdata['gradient'] + 1
    plotdata.loc[plotdata['gradient'] >= 1, 'gradient'] = plotdata['gradient'] - 1

    plotdata.reset_index(level='cluster', inplace=True)
    plotdata.reset_index(level='position', inplace=True)

    print(lineage_name)
    lineage_objects[lineage_name] = {
        'df_latent': df_latent,
        'latent_torch': latent_torch,
        'fragments': fragments,
        'celltypes': latent.columns,
        'gene_name': gene_name,
        'gene_ix': gene_ix,
        'probs': probs,
        'plotdata': plotdata,
        'df_bincounts': df_bincounts,
        'n_fr': n_fr,
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

def plot_differential(fig, gridspec, data, ymax, celltype, n_cells, n_fr, n_exp):
    ax_object = fig.add_subplot(gridspec)
    ax_object.plot(data["position"], (data["prob"]), color="gray", lw=0.5, zorder=1, linestyle="solid")
    ax_object.plot(data["position"], (data["baseline"]), color="black", lw=0.5, zorder=1, linestyle="solid")
    ax_object.set_ylabel(f"q{celltype}, n={int(n_cells)} \n n_fr.={n_fr} \n n_exp.={n_exp} ", rotation=0, ha="right", va="center")
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)  
    ax_object.set_ylim(0, ymax)
    ax_object.set_xlim(0, 1)
    new_ticks = [-10000, -5000, 0, 5000, 10000]
    old_ticks = [(x/20000 + 0.5) for x in new_ticks]
    ax_object.set_xticks(old_ticks)
    ax_object.set_xticklabels(new_ticks)

    # up/down gradient
    polygon = ax_object.fill_between(data["position"], (data["prob"]), (data["baseline"]), color="black", zorder=0)
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    c = data["gradient"].values
    c[c == np.inf] = 0.0
    c[c == -np.inf] = -10.0
    gradient = ax_object.imshow(c.reshape(1, -1), cmap=cmap_atac_diff, aspect="auto", extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], zorder=25, norm=norm_atac_diff)
    gradient.set_clip_path(polygon.get_paths()[0], transform=ax_object.transData)
    polygon.set_alpha(0)
    return ax_object

def plot_expression(fig, gridspec, expression, celltype, annotation, exp_xmin, exp_xmax):
    medianprops = dict(color=annotation[celltype], linewidth=1)
    ax_object = fig.add_subplot(gridspec)
    ax_object.boxplot(expression, vert=False, widths=0.5, showfliers=False, medianprops=medianprops)
    ax_object.set_xlim(exp_xmin - abs(exp_xmin * 0.1), exp_xmax + abs(exp_xmax * 0.1))
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

for lineage in lineage_objects.keys():
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

    lineages = info_genes_cells[lineage].dropna().tolist()
    colors = [annotation_original[x] for x in lineages]
    annotation = gradient(colors)

    cmap_atac_diff=mpl.cm.RdBu_r
    norm_atac_diff=mpl.colors.Normalize(-lineage_objects[lineage]['plotdata']['gradient'].abs().max(), lineage_objects[lineage]['plotdata']['gradient'].abs().max())

    for lineage, celltype, index, plot_row, xaxis, title, c1, c2, c3 in zip(df['lineage'], df['cell type'], df['index'], df['plot_row'], df['xaxis'], df['title'], col_accessibility, col_expression, col_lt):
        print(lineage, celltype, index, plot_row, c1, c2, c3)

        data = lineage_objects[lineage]
        print(data['gene_name'], data['celltypes'])

        # 1. data for accessibility
        bincounts = data['df_bincounts'][celltype]
        ymax = data['df_bincounts'].max().max()
        n_cells = data['latent_torch'][:, index].sum()

        # 1. data for differential accessibility
        diff = data['plotdata'].loc[data['plotdata']['cluster'] == celltype]
        ymax = data['plotdata']['prob'].max()
        n_cells = data['latent_torch'][:, index].sum()

        # 2. data for expression
        expression = data['df_latent'].loc[data['df_latent']['quantile'] == celltype, data['gene_name']].values
        n_exp = (data['df_latent'].loc[data['df_latent']['quantile'] == celltype, 'reads'] != 0).sum(axis=0)

        # 3. data for latent time
        lt = data['df_latent'].loc[data['df_latent']['quantile'] == celltype, 'latent_time'].values

        ax_1 = plot_differential(fig, c1, diff, ymax, celltype, n_cells, data['n_fr'][index], n_exp)
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

    x1, x2, x3 = 0.03, 0.52, 0.73
    y1, y2, y3 = 0.905, 0.58, 0.32

    fig.text(x1, y1, 'A', fontsize=16, fontweight='bold', va='top')
    fig.text(x2, y1, 'B', fontsize=16, fontweight='bold', va='top')
    fig.text(x3, y1, 'C', fontsize=16, fontweight='bold', va='top')

    arrow = FancyArrowPatch(posA=(0.03, 0.87), posB=(0.03, 0.12), arrowstyle='<-, head_width=0.3', connectionstyle=f'arc3, rad=0', mutation_scale=10, lw=1, color='gray')
    fig.add_artist(arrow)
    fig.text(0.02, 0.495, 'Direction of differentiation', fontsize=8, va='center', rotation=90)

    if fig_runner:
        os.makedirs(folder_data_preproc / 'plots' / lineage, exist_ok=True)
        fig.savefig(folder_data_preproc / 'plots' / lineage / f"fig3diff_{lineage}_{lineage_gene[lineage]}.pdf", bbox_inches='tight', pad_inches=0.01)
    else:
        fig.savefig(folder_data_preproc / 'plots' / f"fig3diff_{lineage}_{lineage_gene[lineage]}.pdf", bbox_inches='tight', pad_inches=0.01)
        fig.show()

############################################################################################################################################################################
"End of figure 3"
############################################################################################################################################################################
# %%
