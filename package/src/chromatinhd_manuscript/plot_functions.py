import os
import re
import torch
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

default_figsize = (15, 15)
default_dpi = 100
default_titlesize = 20
default_labelsize = 18
default_xlabel = 'Normalized Genomic Position (0.5 = TSS)'
default_ylabel = 'Normalized Pseudotemporal Ordering'
NF_range = [0, 1]

def extract_model_fold(input_string):
    pattern = r"(\d+(?:_\d+)*_fold_\d+)"
    result = re.search(pattern, input_string)
    if result:
        return result.group(0)
    else:
        return None
    
def cutsites(gene, df, directory, show=False):
    fig, ax = plt.subplots(figsize=default_figsize)
    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {len(df)})", fontsize=default_titlesize) #check if len(df) is correct
    ax.set_xlabel(default_xlabel, fontsize=default_labelsize)
    ax.set_ylabel(default_ylabel, fontsize=default_labelsize)
    ax.set_xlim(NF_range)
    ax.set_ylim(NF_range)
    ax.set_facecolor('white')

    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory / f'{gene}.png', dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def cutsites_histo(gene, df, df_long, n_fragments, directory, show=False):
    fig, axs = plt.subplots(figsize=default_figsize, ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    ax_hist = axs[0]
    ax_hist.hist(df['rank'], bins=100, orientation='horizontal')
    ax_hist.set_xlabel('n cells')
    ax_hist.set_ylabel('Rank')
    ax_hist.set_ylim(NF_range)
    ax_hist.invert_xaxis()

    ax_scatter = axs[1]
    ax_scatter.scatter(df_long['x'], df_long['y'], s=1, marker='s', color='black')
    ax_scatter.set_xlabel(default_xlabel)
    ax_scatter.set_ylabel(default_ylabel)
    ax_scatter.set_xlim(NF_range)
    ax_scatter.set_ylim(NF_range)
    ax_scatter.set_facecolor('white')

    fig.suptitle(f"{gene} (cut sites = {2 * n_fragments})", fontsize=20)

    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory / f'{gene}.png', dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def model_continuous(gene, directory, show=False):
    file_name = directory / f"{gene}.csv"
    probsx = np.loadtxt(file_name, delimiter=',')

    dir_plot_full = directory.parent / 'plots' / directory.name
    file_name = dir_plot_full / f"{gene}.png"

    fig, ax = plt.subplots(figsize=default_figsize)
    heatmap = ax.imshow(probsx, cmap='RdBu_r', aspect='auto')
    cbar = plt.colorbar(heatmap)
    ax.set_title(f"{gene} (likelihood = {extract_model_fold(directory.name)})", fontsize=default_titlesize)
    ax.set_xlabel(default_xlabel, fontsize=default_labelsize)
    ax.set_ylabel(default_ylabel, fontsize=default_labelsize)

    os.makedirs(dir_plot_full, exist_ok=True)
    plt.savefig(file_name, dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def minmax(gene, df, directory, show=False):
    minima = df[df['type'] == 'minima']
    maxima = df[df['type'] == 'maxima']
    
    plt.figure(figsize=default_figsize)
    plt.scatter(minima['x'], minima['y'], c='red', s=1, linewidths=0.5, label='Minima')
    plt.scatter(maxima['x'], maxima['y'], c='blue', s=1, linewidths=0.5, label='Maxima')
    plt.title(f"{gene} (minmax: {extract_model_fold(directory.name)})", fontsize=default_titlesize)
    plt.xlabel(default_xlabel, fontsize=default_labelsize)
    plt.ylabel(default_ylabel, fontsize=default_labelsize)
    plt.xlim(NF_range)
    plt.ylim(NF_range)

    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory / f"{gene}.png", dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close()

def model_quantile(gene_oi, latent_torch, fragments, directory, show=False):
    gene_id = fragments.var.index[gene_oi]

    file_name = directory / f"{gene_id}.csv"
    probs = pd.read_csv(file_name)

    dir_plot_full = directory.parent / 'plots' / directory.name
    file_name = dir_plot_full / f"{gene_id}.png"

    bins = np.linspace(0, 1, 500)
    binmids = (bins[1:] + bins[:-1])/2
    binsize = binmids[1] - binmids[0]
    pseudocoordinates = torch.linspace(0, 1, 1000)

    fig, axes = plt.subplots(probs.shape[0], 1, figsize=(20, 1*probs.shape[0]), sharex = True, sharey = True)
    for i, ax in zip(reversed(range(probs.shape[0])), axes):
        n_cells = latent_torch[:, i].sum()

        fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi)
        bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins)
        freq = round((fragments_oi.sum()/n_cells).item(), 3)

        ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0)
        ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#0000FF", lw = 2, zorder = 20)
        ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#FFFFFF", lw = 3, zorder = 10)
        ax.set_ylabel(f"{probs.iloc[i]['cluster']}\n freq={freq}", rotation = 0, ha = "right", va = "center")

    fig.suptitle(f'Probs for gene_oi = {gene_id}', y=0.92, fontsize=default_titlesize)
    fig.text(0.5, 0.03, default_xlabel, ha="center", fontsize=default_labelsize)
    fig.text(0.04, 0.5, "Likelihood", va="center", rotation="vertical", fontsize=default_labelsize)

    os.makedirs(dir_plot_full, exist_ok=True)
    plt.savefig(file_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

def combine_1rows_3cols(dirs, output_dir, nrows, ncols):
    """
    fixed: height per image in row
    fixed: width per image in column
    flexible: width per image in row
    flexible: height per image in column
    """

    # Get all files in the directories
    files = [sorted([f for f in os.listdir(directory) if f.endswith(".png")]) for directory in dirs]

    # Check if all file lists have the same length
    if not all(len(file) == len(files[0]) for file in files):
        raise ValueError("All file lists must have the same length.")

    # Create grid positions
    combinations = list(itertools.product(range(1, nrows+1), range(1, ncols+1)))

    for i, files_in_row in enumerate(zip(*files)):
        # Check if all files in the row have the same name
        if not all(file == files_in_row[0] for file in files_in_row):
            raise ValueError("All files in the row must have the same name.")

        # Open images from each directory
        images = [Image.open(os.path.join(dirs[j], file)) for j, file in enumerate(files_in_row)]

        # Get image dimensions
        widths = [img.width for img in images]
        heights = [img.height for img in images]

        # Create dataframe with grid positions and image dimensions
        row_positions, col_positions = zip(*combinations)
        df = pd.DataFrame({'Row': row_positions, 'Col': col_positions, 'Widths': widths, 'Heights': heights})

        # Calculate the combined image grid positions
        df['x'] = df.groupby('Row')['Widths'].cumsum() - df['Widths']
        df['y'] = df.groupby('Col')['Heights'].cumsum() - df['Heights']

        # Calculate the combined image grid dimensions
        width_sum = df.loc[df['Row'] == 1, 'Widths'].sum()
        height_sum = df.loc[df['Col'] == 1, 'Heights'].sum()

        # Create a blank canvas for the combined image grid
        combined_img = Image.new("RGB", (width_sum, height_sum))

        # Paste images into the combined image grid
        for j, img in enumerate(images):
            combined_img.paste(img, (df.loc[j, 'x'], df.loc[j, 'y']))

        # Save the combined image
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, files_in_row[0])
        combined_img.save(output_file, "PNG")

        print(output_file)

